"""Shared data loading, feature engineering, and evaluation utilities for the
Optiver realized volatility notebooks."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

WINDOW_SECONDS = 600
BOOK_BASE_FEATURES = [
    "rv_wap1", "rv_wap2", "spread_mean", "spread_std",
    "imbalance_mean", "imbalance_std", "n_updates", "wap_range",
]
TRADE_BASE_FEATURES = ["trade_count", "trade_volume", "rv_trade"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmspe(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def realized_volatility(log_returns):
    return np.sqrt(np.sum(np.square(log_returns)))


def log_return(prices):
    return np.log(prices).diff()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_book(stock_id, data_dir):
    return pd.read_parquet(Path(data_dir) / "book_train.parquet" / f"stock_id={stock_id}")


def load_trade(stock_id, data_dir):
    return pd.read_parquet(Path(data_dir) / "trade_train.parquet" / f"stock_id={stock_id}")


def compute_wap(df, level=1):
    bid_price, ask_price = df[f"bid_price{level}"], df[f"ask_price{level}"]
    bid_size, ask_size = df[f"bid_size{level}"], df[f"ask_size{level}"]
    return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)


def split_time_ids(time_ids, frac=0.5):
    """Chronological train/test split of unique time_ids at the `frac` point."""
    sorted_ids = np.array(sorted(set(time_ids)))
    split_idx = int(len(sorted_ids) * frac)
    return sorted_ids[:split_idx], sorted_ids[split_idx:]


# ---------------------------------------------------------------------------
# Baseline: whole-bucket realized volatility of wap1
# ---------------------------------------------------------------------------

def naive_rv_baseline(stock_ids, data_dir, train):
    """Whole 10-minute-bucket realized volatility of wap1 for every (stock_id, time_id)."""
    frames = []
    for stock_id in stock_ids:
        book = load_book(stock_id, data_dir)
        book["wap1"] = compute_wap(book, level=1)
        book["log_ret"] = book.groupby("time_id")["wap1"].transform(log_return)
        book = book.dropna(subset=["log_ret"])
        rv = book.groupby("time_id")["log_ret"].apply(realized_volatility)
        frames.append(pd.DataFrame({"stock_id": stock_id, "time_id": rv.index, "pred": rv.to_numpy()}))
    pred = pd.concat(frames, ignore_index=True)
    return pred.merge(train[["stock_id", "time_id", "target"]], on=["stock_id", "time_id"], how="inner")


# ---------------------------------------------------------------------------
# Windowed feature engineering
# ---------------------------------------------------------------------------

def compute_stock_features(stock_id, data_dir, n_windows=20, window_seconds=WINDOW_SECONDS):
    """Book/trade features for one stock, computed independently within `n_windows`
    equal sub-windows of each 10-minute bucket."""
    b = load_book(stock_id, data_dir)
    b["wap1"] = compute_wap(b, level=1)
    b["wap2"] = compute_wap(b, level=2)
    b["spread"] = (b["ask_price1"] - b["bid_price1"]) / b["wap1"]
    b["imbalance"] = (b["bid_size1"] - b["ask_size1"]) / (b["bid_size1"] + b["ask_size1"])
    b["ret1"] = b.groupby("time_id")["wap1"].transform(log_return)
    b["ret2"] = b.groupby("time_id")["wap2"].transform(log_return)
    b["window"] = np.minimum((b["seconds_in_bucket"] * n_windows // window_seconds).astype(int), n_windows - 1)

    book_g = b.groupby(["time_id", "window"]).agg(
        rv_wap1=("ret1", lambda x: np.sqrt(np.nansum(x.to_numpy() ** 2))),
        rv_wap2=("ret2", lambda x: np.sqrt(np.nansum(x.to_numpy() ** 2))),
        spread_mean=("spread", "mean"),
        spread_std=("spread", "std"),
        imbalance_mean=("imbalance", "mean"),
        imbalance_std=("imbalance", "std"),
        n_updates=("wap1", "size"),
        wap1_min=("wap1", "min"),
        wap1_max=("wap1", "max"),
        wap1_mean=("wap1", "mean"),
    )
    book_g["wap_range"] = (book_g["wap1_max"] - book_g["wap1_min"]) / book_g["wap1_mean"]

    book_feats = book_g[BOOK_BASE_FEATURES].unstack("window")
    book_feats.columns = [f"{feat}_w{w}" for feat, w in book_feats.columns]
    n_updates_cols = [f"n_updates_w{w}" for w in range(n_windows)]
    book_feats[n_updates_cols] = book_feats[n_updates_cols].fillna(0)
    book_feats = book_feats.reset_index()

    t = load_trade(stock_id, data_dir)
    t["trade_ret"] = t.groupby("time_id")["price"].transform(log_return)
    t["window"] = np.minimum((t["seconds_in_bucket"] * n_windows // window_seconds).astype(int), n_windows - 1)

    trade_g = t.groupby(["time_id", "window"]).agg(
        trade_count=("price", "size"),
        trade_volume=("size", "sum"),
        rv_trade=("trade_ret", lambda x: np.sqrt(np.nansum(x.to_numpy() ** 2))),
    )
    trade_feats = trade_g[TRADE_BASE_FEATURES].unstack("window")
    trade_feats.columns = [f"{feat}_w{w}" for feat, w in trade_feats.columns]
    trade_feats = trade_feats.reset_index()

    feats = book_feats.merge(trade_feats, on="time_id", how="left")
    trade_cols = [f"{feat}_w{w}" for feat in TRADE_BASE_FEATURES for w in range(n_windows)]
    feats[trade_cols] = feats[trade_cols].fillna(0.0)
    feats["stock_id"] = stock_id
    return feats


def add_market_features(all_features, n_windows=20):
    """Cross-sectional mean/std of rv_wap1_w{i} across all stocks at the same time_id."""
    rv_cols = [f"rv_wap1_w{w}" for w in range(n_windows)]
    mkt_mean = all_features.groupby("time_id")[rv_cols].transform("mean").rename(columns=lambda c: f"mkt_{c}_mean")
    mkt_std = all_features.groupby("time_id")[rv_cols].transform("std").rename(columns=lambda c: f"mkt_{c}_std")
    mkt_cols = list(mkt_mean.columns) + list(mkt_std.columns)
    return pd.concat([all_features, mkt_mean, mkt_std], axis=1), mkt_cols


def build_feature_matrix(stock_ids, data_dir, n_windows=20, window_seconds=WINDOW_SECONDS):
    """Full feature matrix (per-window book/trade features + cross-sectional market
    features) for all stocks, plus the list of feature column names."""
    all_features = pd.concat(
        [compute_stock_features(sid, data_dir, n_windows, window_seconds) for sid in stock_ids],
        ignore_index=True,
    )
    feature_cols = [f"{feat}_w{w}" for w in range(n_windows) for feat in (BOOK_BASE_FEATURES + TRADE_BASE_FEATURES)]
    all_features, mkt_cols = add_market_features(all_features, n_windows)
    feature_cols = feature_cols + mkt_cols
    return all_features, feature_cols


# ---------------------------------------------------------------------------
# Evaluation: blocked CV for train error bars, bootstrap for test error bars
# ---------------------------------------------------------------------------

def time_blocked_folds(time_ids, n_splits=5):
    """Chronological KFold over sorted unique time_ids (contiguous blocks, no shuffling)."""
    sorted_ids = np.array(sorted(set(time_ids)))
    kf = KFold(n_splits=n_splits, shuffle=False)
    for train_idx, val_idx in kf.split(sorted_ids):
        yield sorted_ids[train_idx], sorted_ids[val_idx]


def cv_scores_baseline(pred_df, time_ids, n_splits=5, metric_fn=rmspe):
    """Fold-level metric of a fit-free baseline over blocked CV folds of time_ids."""
    scores = []
    for _, val_ids in time_blocked_folds(time_ids, n_splits):
        fold = pred_df[pred_df["time_id"].isin(val_ids)]
        scores.append(metric_fn(fold["target"].to_numpy(), fold["pred"].to_numpy()))
    return np.array(scores)


def cv_scores_model(model_factory, data, feature_cols, time_ids, n_splits=5, metric_fn=rmspe, weighted=True):
    """Fit a fresh model_factory() on each CV fold's training time_ids, score on the held-out fold."""
    scores = []
    for fold_train_ids, fold_val_ids in time_blocked_folds(time_ids, n_splits):
        train_fold = data[data["time_id"].isin(fold_train_ids)]
        val_fold = data[data["time_id"].isin(fold_val_ids)]
        model = model_factory()
        sample_weight = 1.0 / train_fold["target"].to_numpy() ** 2 if weighted else None
        model.fit(train_fold[feature_cols], train_fold["target"], sample_weight=sample_weight)
        pred = model.predict(val_fold[feature_cols])
        scores.append(metric_fn(val_fold["target"].to_numpy(), pred))
    return np.array(scores)


def fit_predict(model_factory, train_data, test_data, feature_cols, weighted=True):
    """Fit model_factory() on train_data, predict on test_data."""
    model = model_factory()
    sample_weight = 1.0 / train_data["target"].to_numpy() ** 2 if weighted else None
    model.fit(train_data[feature_cols], train_data["target"], sample_weight=sample_weight)
    return model.predict(test_data[feature_cols])


def bootstrap_scores(df, y_true_col, y_pred_col, metric_fn=rmspe, n_boot=100, seed=0):
    """Bootstrap-resample rows of df and evaluate metric_fn on each resample."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = df[y_true_col].to_numpy(), df[y_pred_col].to_numpy()
    n = len(df)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])
    return scores


def summarize_scores(model, train_scores, test_scores):
    return {
        "model": model,
        "train_rmspe_mean": train_scores.mean(),
        "train_rmspe_std": train_scores.std(),
        "test_rmspe_mean": test_scores.mean(),
        "test_rmspe_std": test_scores.std(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_prediction_diagnostics(target, pred, title=None, color="tab:blue"):
    """Scatter of target vs. prediction, and a relative-error histogram."""
    target, pred = np.asarray(target), np.asarray(pred)
    rel_error = (target - pred) / target
    limit = max(np.quantile(target, 0.995), np.quantile(pred, 0.995))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(target, pred, s=6, alpha=0.25, color=color)
    axes[0].plot([0, limit], [0, limit], color="red", linewidth=1.5)
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_xlabel("target")
    axes[0].set_ylabel("prediction")
    axes[0].set_title(title or "target vs prediction")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(np.clip(rel_error, -3, 3), bins=100, alpha=0.7, color=color)
    axes[1].axvline(0, color="red", linewidth=1.5)
    axes[1].set_xlabel("relative error: (target - pred) / target")
    axes[1].set_ylabel("count")
    axes[1].set_title("relative error")
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    return fig, axes
