# Volatility-Forcast-and-Inference

## Setup
```bash
conda env create -f environment.yml
conda activate pymc-env
```

## Drift vs. volatility identifiability
For each stock and year, fits a GBM (`mu`, `sigma`) to weekly log-returns via nested
sampling, and compares it against a restricted `mu = 0` model using the Bayes factor
(`logZ_full - logZ_mu0`). Pooled across many stocks and years, the Bayes factor rarely
gives consistent evidence for nonzero drift — drift is often indistinguishable from noise,
while volatility is comparatively well identified. See the last plot in
[`gbm_nested_sampling_drift_volatility.ipynb`](notebooks/drift_vs_volatility_inference/gbm_nested_sampling_drift_volatility.ipynb)
(pooled Bayes factor scatter + histogram across stock-years).

## Optiver realized volatility
Compares a naive realized-volatility baseline against linear regression, LightGBM, and
XGBoost trained on windowed order-book/trade features, using blocked time-series CV
(train) and bootstrap (test) RMSPE. See the last plot in
[`optiver_vol_modeling.ipynb`](notebooks/optiver_volatility/optiver_vol_modeling.ipynb)
(model comparison: train vs. test RMSPE).
