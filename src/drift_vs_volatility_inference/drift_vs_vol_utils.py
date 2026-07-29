"""Shared data loading, nested-sampling inference, and plotting utilities for the
GBM drift-vs-volatility notebooks."""
import dynesty
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from dynesty import plotting as dyplot
from dynesty.utils import quantile

RETURN_WINDOW = 5
MU_MIN, MU_MAX = -2.0, 2.0
SIGMA_MIN, SIGMA_MAX = 0.01, 3.0
NLIVE = 300
DLOGZ = 0.05
RANDOM_SEED = 123


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def download_prices(ticker, start_year, end_year):
    """Daily close prices for `ticker` from Jan 1 `start_year` through Dec 31 `end_year`."""
    downloaded = yf.download(
        ticker,
        start=f"{start_year}-01-01",
        end=f"{end_year + 1}-01-01",
        auto_adjust=False,
        progress=False,
    )
    prices = downloaded["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    return prices.dropna()


def get_year_returns(prices, year, return_window=RETURN_WINDOW):
    """Prices and log-returns for `year`, sub-sampled every `return_window` trading days."""
    year_prices = prices.loc[prices.index.year == year].dropna()
    sampled_prices = year_prices.iloc[::return_window]
    log_returns = np.log(sampled_prices / sampled_prices.shift(1)).dropna().to_numpy()
    return year_prices, log_returns


# ---------------------------------------------------------------------------
# GBM model: priors and log-likelihoods
# ---------------------------------------------------------------------------

def make_prior_transform_full(mu_min=MU_MIN, mu_max=MU_MAX, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX):
    def prior_transform(u):
        x = np.array(u)
        x[0] = mu_min + x[0] * (mu_max - mu_min)
        x[1] = sigma_min + x[1] * (sigma_max - sigma_min)
        return x

    return prior_transform


def make_prior_transform_mu0(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX):
    def prior_transform(u):
        x = np.array(u)
        x[0] = sigma_min + x[0] * (sigma_max - sigma_min)
        return x

    return prior_transform


def make_loglike_full(log_returns, dt):
    def loglike(theta):
        mu, sigma = theta
        mean = (mu - 0.5 * sigma**2) * dt
        std = sigma * np.sqrt(dt)
        residual = (log_returns - mean) / std
        return -0.5 * np.sum(residual**2 + np.log(2 * np.pi * std**2))

    return loglike


def make_loglike_mu0(log_returns, dt):
    def loglike(theta):
        sigma = theta[0]
        mean = -0.5 * sigma**2 * dt
        std = sigma * np.sqrt(dt)
        residual = (log_returns - mean) / std
        return -0.5 * np.sum(residual**2 + np.log(2 * np.pi * std**2))

    return loglike


# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------

def _weighted_quantile_summary(values, weights):
    low, median, high = quantile(values, [0.025, 0.5, 0.975], weights=weights)
    mean = np.average(values, weights=weights)
    std = np.sqrt(np.average((values - mean) ** 2, weights=weights))
    return mean, std, low, median, high


def posterior_summary_full(results):
    samples = results.samples
    weights = results.importance_weights()
    row = {}
    for name, values in [("mu", samples[:, 0]), ("sigma", samples[:, 1])]:
        mean, std, low, median, high = _weighted_quantile_summary(values, weights)
        row[f"{name}_mean"] = mean
        row[f"{name}_std"] = std
        row[f"{name}_q025"] = low
        row[f"{name}_median"] = median
        row[f"{name}_q975"] = high
    return row


def posterior_summary_mu0(results):
    sigma = results.samples[:, 0]
    weights = results.importance_weights()
    mean, std, low, median, high = _weighted_quantile_summary(sigma, weights)
    return {
        "mu0_sigma_mean": mean,
        "mu0_sigma_std": std,
        "mu0_sigma_q025": low,
        "mu0_sigma_median": median,
        "mu0_sigma_q975": high,
    }


# ---------------------------------------------------------------------------
# Nested sampling runs
# ---------------------------------------------------------------------------

def run_nested_for_year(
    prices, year, return_window=RETURN_WINDOW,
    mu_min=MU_MIN, mu_max=MU_MAX, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
    nlive=NLIVE, dlogz=DLOGZ, random_seed=RANDOM_SEED,
):
    """Fit the free-mu and mu=0 GBM models to one year of log-returns via nested sampling."""
    dt = return_window / 252
    year_prices, log_returns = get_year_returns(prices, year, return_window)

    sampler_full = dynesty.NestedSampler(
        make_loglike_full(log_returns, dt),
        make_prior_transform_full(mu_min, mu_max, sigma_min, sigma_max),
        ndim=2,
        nlive=nlive,
        rstate=np.random.default_rng(random_seed + year),
    )
    sampler_full.run_nested(dlogz=dlogz, print_progress=False)

    sampler_mu0 = dynesty.NestedSampler(
        make_loglike_mu0(log_returns, dt),
        make_prior_transform_mu0(sigma_min, sigma_max),
        ndim=1,
        nlive=nlive,
        rstate=np.random.default_rng(random_seed + 10000 + year),
    )
    sampler_mu0.run_nested(dlogz=dlogz, print_progress=False)

    results_full = sampler_full.results
    results_mu0 = sampler_mu0.results

    row = posterior_summary_full(results_full)
    row.update(posterior_summary_mu0(results_mu0))
    row["year"] = year
    row["price_observations"] = len(year_prices)
    row["return_observations"] = len(log_returns)
    row["logz_full"] = results_full.logz[-1]
    row["logzerr_full"] = results_full.logzerr[-1]
    row["logz_mu0"] = results_mu0.logz[-1]
    row["logzerr_mu0"] = results_mu0.logzerr[-1]
    row["log_bayes_factor_full_vs_mu0"] = row["logz_full"] - row["logz_mu0"]

    return row, results_full, results_mu0


def run_nested_for_years(prices, years, verbose=True, **kwargs):
    """Run `run_nested_for_year` over `years`, returning a summary DataFrame sorted by
    year plus dicts of per-year nested-sampling results for the full and mu=0 models."""
    rows = []
    results_full_by_year = {}
    results_mu0_by_year = {}

    for year in years:
        if verbose:
            print(f"Running {year}")
        row, results_full, results_mu0 = run_nested_for_year(prices, year, **kwargs)
        rows.append(row)
        results_full_by_year[year] = results_full
        results_mu0_by_year[year] = results_mu0

    yearly_summary = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return yearly_summary, results_full_by_year, results_mu0_by_year


def run_nested_for_tickers(tickers, years, min_return_observations=10, verbose=True, **kwargs):
    """Run `run_nested_for_year` for every (ticker, year) pair, pooling results into one
    DataFrame with a `ticker` column. Skips pairs with too few return observations (e.g.
    a ticker's IPO year)."""
    rows = []
    results_full_by_ticker_year = {}
    results_mu0_by_ticker_year = {}
    return_window = kwargs.get("return_window", RETURN_WINDOW)

    for ticker in tickers:
        prices = download_prices(ticker, min(years), max(years))
        for year in years:
            _, log_returns = get_year_returns(prices, year, return_window)
            if len(log_returns) < min_return_observations:
                if verbose:
                    print(f"Skipping {ticker} {year}: {len(log_returns)} return observations")
                continue
            if verbose:
                print(f"Running {ticker} {year}")
            row, results_full, results_mu0 = run_nested_for_year(prices, year, **kwargs)
            row["ticker"] = ticker
            rows.append(row)
            results_full_by_ticker_year[(ticker, year)] = results_full
            results_mu0_by_ticker_year[(ticker, year)] = results_mu0

    pooled_summary = pd.DataFrame(rows).sort_values(["ticker", "year"]).reset_index(drop=True)
    return pooled_summary, results_full_by_ticker_year, results_mu0_by_ticker_year


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_corner(results, labels=(r"$\mu$", r"$\sigma$")):
    return dyplot.cornerplot(results, labels=list(labels), show_titles=True)


def plot_drift_volatility_evidence(yearly_summary, stock_name):
    """3-panel time series: drift posterior, volatility posterior (full vs. mu=0 model),
    and the log Bayes factor comparing the free-mu model to the mu=0 model."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    mu_yerr = [
        yearly_summary["mu_median"] - yearly_summary["mu_q025"],
        yearly_summary["mu_q975"] - yearly_summary["mu_median"],
    ]
    axes[0].errorbar(yearly_summary["year"], yearly_summary["mu_median"], yerr=mu_yerr, fmt="o-", capsize=3)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("annualized drift")
    axes[0].set_title(f"{stock_name}: full model posterior for mu")

    sigma_yerr = [
        yearly_summary["sigma_median"] - yearly_summary["sigma_q025"],
        yearly_summary["sigma_q975"] - yearly_summary["sigma_median"],
    ]
    axes[1].errorbar(
        yearly_summary["year"], yearly_summary["sigma_median"], yerr=sigma_yerr,
        fmt="o-", capsize=3, label="full model",
    )
    axes[1].plot(yearly_summary["year"], yearly_summary["mu0_sigma_median"], "s--", label="mu = 0 model")
    axes[1].set_ylabel("annualized volatility")
    axes[1].set_title(f"{stock_name}: posterior for sigma")
    axes[1].legend()

    axes[2].bar(yearly_summary["year"], yearly_summary["log_bayes_factor_full_vs_mu0"])
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set_xlabel("year")
    axes[2].set_ylabel("logZ_full - logZ_mu0")
    axes[2].set_title("Evidence comparison")

    fig.tight_layout()
    return fig, axes


def plot_bayes_factor_scatter_hist(pooled_summary):
    """Scatter of the log Bayes factor (full vs. mu=0) by year, one dot per (ticker, year),
    plus a pooled histogram of the same values across all stock-years."""
    n_tickers = pooled_summary["ticker"].nunique()
    bayes_factor = pooled_summary["log_bayes_factor_full_vs_mu0"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(pooled_summary["year"], bayes_factor, s=18, alpha=0.5)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("year")
    axes[0].set_ylabel("logZ_full - logZ_mu0")
    axes[0].set_title(f"Bayes factor by year ({n_tickers} stocks, {len(pooled_summary)} stock-years)")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(bayes_factor, bins=40, alpha=0.75)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("logZ_full - logZ_mu0")
    axes[1].set_ylabel("count")
    axes[1].set_title("Pooled distribution across stock-years")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    return fig, axes
