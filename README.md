# Volatility-Forecast-and-Inference

## Setup
```bash
conda env create -f environment.yml
conda activate pymc-env
```

## Identifiability of drift vs. volatility
A stock's price is modeled as Geometric Brownian Motion, governed by two parameters: drift
`mu` (the expected return) and volatility `sigma`. For each stock and year, a Bayesian
nested-sampling fit estimates both from historical returns, then a Bayes factor asks
whether a model that allows `mu` to be nonzero actually fits the data better than a
restricted model that assumes `mu = 0`. Repeated across many stocks and years, the answer
is almost always "no" — the Bayes factor rarely shows real support for nonzero drift, so a
year's worth (or more) of returns usually isn't enough to tell an actual trend apart from
what pure noise would produce, even though volatility itself is pinned down with much more
confidence. See the last plot in
[`gbm_nested_sampling_drift_volatility.ipynb`](notebooks/drift_vs_volatility_inference/gbm_nested_sampling_drift_volatility.ipynb):
a scatter and histogram of this Bayes factor pooled across all stock-years.

## Volatility forecasting from Optiver order-book data
Given a snapshot of a stock's limit order book and trades over a 10-minute window (the
Optiver Kaggle dataset), the task is to predict the realized volatility of the *next*
10-minute window. Order-book/trade features (bid-ask spread, order imbalance, trade
volume, sub-window realized volatility, ...) feed into linear regression, LightGBM, and
XGBoost, benchmarked against a naive baseline that just reuses the current window's
realized volatility as its forecast. The models cut error by ~30% versus that baseline
(test RMSPE 0.24 vs. 0.34) — but linear regression alone gets almost all the way there;
LightGBM and XGBoost barely improve on it. That suggests volatility is close to linearly
predictable from these features at this horizon, and that most of the remaining error is
irreducible noise rather than something a more flexible model would fix. See the last plot
in
[`optiver_vol_modeling.ipynb`](notebooks/optiver_volatility/optiver_vol_modeling.ipynb):
train (CV) vs. test (bootstrap) RMSPE for all four models.
