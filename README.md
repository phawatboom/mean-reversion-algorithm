# MR-PROD — Mean‑Reversion Research → Production (Headless)

A fast, leak‑aware pipeline that produces **per‑stock, per‑date** mean‑reversion likelihoods and **expected move sizes** for horizons **1, 3, 7, 15, 30 trading days**. It supports **walk‑forward backtests**, **next‑day forecasts**, **probability calibration**, and **conformal prediction intervals**.

---

## 1) Quick Start

**Install**

```bash
pip install pandas numpy scikit-learn openpyxl joblib matplotlib tqdm
# Optional tree models
pip install xgboost lightgbm yfinance fredapi
```

**Backtest**

```bash
python main_model.py --mode backtest --excel MXUS_Data.xlsx --sheet Returns --out results
```

**Forecast next day**

```bash
python main_model.py --mode forecast --excel MXUS_Data.xlsx --sheet Returns --out results
```

**Classifier choice**

```bash
# Logistic regression (default)
python main_model.py --model logit
# or gradient‑boosted trees
python main_model.py --model xgb
python main_model.py --model lgb
```

---

## 2) Data Requirements

**Excel workbook**

* **Returns** sheet (default name can be auto‑detected):

  * Column **Date**
  * One column per ticker, cells may be numeric or strings like `0.56%`
* **Market** sheet (optional):

  * Either `MKT_RET` daily return, or a single price column. If absent, the script can fetch S&P 500 and derive features.

**Header auto‑discovery**

* The loader searches for the first row whose first cell equals **Date**. All rows above are ignored.

**Cleaning**

* Strings like `0.56%` are parsed to decimals.
* Returns are **winsorized** to ±10 percent by default.
* Tickers with less than 70 percent valid rows are dropped.

---

## 3) What the model predicts

**Classification head**

* **Target**: whether the next h‑day move **reverses** the most recent momentum sign, **only** when the future move magnitude exceeds a **vol‑scaled gate**.
* **Label rule** for horizon `h` and momentum lookback `k`:

  * Compute forward return `fwd_h` and recent momentum `mom_k`.
  * Gate `G = vol_k × rolling_std_20d × sqrt(h)`.
  * If `|fwd_h| > G` and `sign(fwd_h) = − sign(mom_k)`, label is **1** (mean‑reversion).
  * If `|fwd_h| > G` and `sign(fwd_h) = + sign(mom_k)`, label is **0** (momentum).
  * Else unlabeled and excluded from training.

**Magnitude head**

* Two regressors estimate **|move|** conditional on regime: one for MR, one for momentum.
* Expected signed move mixes regime magnitudes with predicted probabilities and regime signs.

**Calibration and intervals**

* **Isotonic regression** calibrates probabilities using **out‑of‑sample** predictions on the outer‑train slice.
* **Conformal intervals** from calibration residuals give **90 percent** prediction bands around the expected move.

---

## 4) Features

**From returns only**

* AR(1) rolling autocorrelation: windows 10, 20, 60
* Sign‑persistence rate: windows 20, 60
* Short‑history returns: 1‑day lag, 5‑day, 20‑day compounded
* Rolling volatility: 10, 20
* Price‑index z‑score vs rolling MA: 20
* Cross‑sectional ranks for selected features per Date

**Optional market context**

* Shifted 1‑day and 5‑day market returns, 20‑day market volatility, 20‑day market z‑score.

---

## 5) Walk‑Forward Protocol

**Outer windows**

* Train: **3 years**, Validation: **1 year** (configurable), step by validation size.
* **Horizon‑aware purge**: all train samples within `h` business days before the validation start are removed.
* **Embargo**: predictions start `embargo_days` after validation start to avoid spillover.

**Inner CV**

* Purged **TimeSeriesSplit** folds to select hyperparameters.
* Selection score combines AUC and PR‑AUC. If a minimum decision coverage is requested, F1 on the covered subset is also considered.

**Calibration split**

* A recent slice of outer‑train (default 20 percent of dates, minimum 30) is used for conformal residuals and reporting.

---

## 6) Decisioning and Coverage Control

**Objective**

* Make decisions only when confident, otherwise abstain.

**Rules**

* Probability filter: decide MR if `p ≥ τ`, momentum if `p ≤ 1 − τ`.
* **Coverage‑driven τ selection**: choose the largest τ whose symmetric coverage meets `min_coverage`.
* Optional **magnitude filter**: require `|expected_move| ≥ min_exp_move_abs`.

**Outputs**

* `y_pred ∈ {1, 0, −1}` where −1 means **abstain**.
* Coverage and accuracy are reported for the covered subset.

---

## 7) Modes and CLI

**Backtest**

* Produces calibrated probabilities and magnitude bands for all dates in each validation window.
* Saves per‑horizon predictions, per‑ticker metrics, and a run summary.

**Forecast**

* Trains on the most recent window and emits next‑day forecasts for all tickers at the latest feature date.
* Produces a single `forecast_YYYYMMDD.csv` plus metadata.

**Common flags**

* `--mode {backtest,forecast}`
* `--excel PATH`, `--sheet NAME`, `--out DIR`
* `--model {logit,xgb,lgb}`
* `--train_years`, `--val_years`, `--embargo`
* `--with_market/--no_market`, `--market_sheet`, `--market_source`, `--market_start`
* Decisioning: `--min_coverage`, `--threshold_fixed`, `--threshold_rule {coverage,none}`, `--min_exp_move`
* Performance and stability: `--n_jobs`, `--random_state`, `--min_valid_frac`

**Examples**

```bash
# Coverage‑driven decisions with 40 percent target coverage
python main_model.py --mode backtest --min_coverage 0.40

# Fixed threshold at 0.75 and minimum 1 percent expected move
python main_model.py --mode forecast --threshold_fixed 0.75 --min_exp_move 0.01

# Use LightGBM with market features disabled
python main_model.py --mode backtest --model lgb --no_market
```

---

## 8) Outputs and Schema

**Backtest artifacts**

* `predictions_{horizon}.csv` per horizon

  * `Date`, `ticker`, `horizon`, `y_true` (0 or 1), `fwd_ret`
  * `mom_sign`, `y_prob` (raw), `y_prob_cal` (calibrated), `mr_score` in [−1, 1]
  * `confidence = max(p, 1 − p)`, `exp_move`, `pi_low`, `pi_high`
  * `threshold` used, `y_pred` in {1, 0, −1}, `abstained` boolean
* `metrics_{horizon}.csv`

  * Per‑ticker: `support`, `covered`, `coverage`, `accuracy`, `precision`, `recall`, `f1`, `auc`
* `summary.csv`

  * Per horizon: rows, unique tickers, unique dates, `auc`, `coverage`, `accuracy_full`, `accuracy_covered`, `covered_rows`
* `feature_columns.txt` and `run_meta.json`

**Forecast artifacts**

* `forecast_YYYYMMDD.csv` with one row per ticker per horizon

  * Core columns: `date`, `ticker`, `horizon`, `raw_probability`, `calibrated_probability`, `confidence`, `expected_move`, `prediction_interval_low`, `prediction_interval_high`, `momentum_sign`, `decision_threshold`, `prediction`, `abstained`
  * Plus all model features as additional columns
* `run_meta_forecast.json`

---

## 9) Configuration Reference (selected)

* `train_years`, `validation_years`, `trading_days_per_year`: outer walk‑forward geometry
* `embargo_days`: gap after validation start before scoring begins
* `vol_window`, `vol_k`: label gate strength via rolling volatility
* `alpha_grid`, `reg_strengths_C`, `inner_cv_folds`: logistic regression search space and CV
* `xgb_param_grid`, `lgb_param_grid`: optional tree model grids
* `horizons`: list of `(name, forward_h, momentum_lookback_k)` tuples
* `ar_windows`, `sign_persist_windows`, `vol_feat_windows`, `z_score_windows`: feature windows
* `winsor_abs_limit`: absolute clip for daily returns
* `calib_frac`: fraction of outer‑train dates for conformal residuals and reporting
* `conformal_alpha`: 0.10 gives 90 percent prediction intervals
* `threshold_grid`, `min_coverage`, `threshold_fixed`, `min_exp_move_abs`, `threshold_rule`

---

## 10) Example Walkthrough

**Goal**: forecast next‑day moves for the latest date using logistic regression, decide only when `|expected_move| ≥ 1 percent` and coverage‑driven thresholding meets 40 percent coverage.

```bash
python main_model.py \
  --mode forecast \
  --excel MXUS_Data.xlsx --sheet Returns \
  --model logit \
  --min_coverage 0.40 \
  --threshold_rule coverage \
  --min_exp_move 0.01
```

**Interpretation**

* `calibrated_probability` is the likelihood of **mean‑reversion**. Values near 1 imply a reversal is likely.
* `expected_move` is the signed expectation that mixes regime magnitudes with the calibrated probability and the recent momentum sign.
* `prediction_interval_low` and `prediction_interval_high` give a calibrated move band.
* `prediction` equals 1 for MR, 0 for momentum, or −1 for abstain.

---

## 11) Reproducibility and Performance

* Deterministic seeds via `random_state`.
* Parallel scoring of hyperparameters with `joblib` using `--n_jobs`.
* Logging to `results/run.log` with INFO level by default.

---

## 12) Troubleshooting

**Could not find a header row with 'Date'**

* Ensure the first column that marks the real header row contains the literal `Date` then real column headers on that row.

**No predictions for a horizon**

* Not enough labeled examples after gating, or class imbalance collapses a fold. Try reducing `vol_k`, lengthening `train_years`, or enabling market features.

**Market fetch failed**

* Network issues or missing packages. Use a local `Market` sheet or run with `--no_market`.

**Single‑class fold errors**

* The code guards AUC and calibration against single‑class validation folds. If it still happens, increase data, tweak gates, or widen windows.

**Autodiscovery cannot find a workbook**

* Provide `--excel` explicitly or place the workbook in the CWD named `MXUS_Data.xlsx` or `MXUS_Data_*.xlsx`.

---

## 13) Design Notes

* **Leak‑aware**: horizon‑aware purge, embargo, purged inner CV, OOS calibration.
* **Interpretable**: returns‑only features with optional market context, logistic baseline.
* **Production‑ready**: consistent schemas, artifacts, and metadata for downstream consumers.

---

## 14) License and Attribution

Internal research tooling. If reused externally, add appropriate license text and credit the MR‑PROD authors.
