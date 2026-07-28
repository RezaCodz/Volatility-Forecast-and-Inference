import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score


def rmspe(y_true, y_pred):
    p_error = ((y_true - y_pred) / y_true)**2
    return np.sqrt(np.mean(p_error))

def RV(r): 
    return np.sqrt(np.sum((r) **2))

def return_calc(prices):
    return np.log(prices).diff()


def baseline_pred(stock_id, time_inds):

    book = pd.read_parquet(data_dir/"book_train.parquet"/f"stock_id={stock_id}")
    book = book[book["time_id"].isin(time_inds)].copy()
    book["wap_1"] = (book["bid_price1"] * book["ask_size1"] + book["ask_price1"] * book["bid_size1"])/(book["bid_size1"]+book["ask_size1"])
    book["return_1"] = book.groupby("time_id")["wap_1"].transform(return_calc)
    book_clean = book.dropna(subset = ["return_1"])


    realized_vol = book_clean.groupby("time_id")["return_1"].agg(RV)


    base_line_pred = pd.DataFrame({"time_id" : realized_vol.index, "current_vl": realized_vol.values})
    base_line_pred = base_line_pred.merge(train[train["stock_id"] == stock_id][["stock_id","time_id", "target"]], on="time_id")

    return base_line_pred


def bootstrap(prediction_data, column_target, column_pred):
    rng = np.random.default_rng(0)
    rmspe_list = []
    n = prediction_data.shape[0]
    for _ in range(100):
        idx = rng.choice(n, n, replace=True)
        sample = prediction_data.iloc[idx]
        rmspe_list.append(rmspe(sample[column_target],sample[column_pred]))
    return rmspe_list