# filepath: stable_2_lastestv3.py
#!/usr/bin/env python3
"""
MR-PROD — Mean-Reversion Research → Production (headless)
=========================================================

Purpose
-------
Produce *per-stock*, *per-date* mean-reversion likelihoods and expected
move sizes for the next 1, 3, 7, 15, and 30 trading days — with leak-aware
validation, probability calibration, and conformal magnitude bands.

Modes
-----
1) backtest  → Full walk-forward backtest with calibrated probabilities, magnitude
               head, conformal bands, and per-ticker metrics (existing pipeline).
2) forecast  → Train on most-recent window and emit next-day forecasts for all
               tickers (probability, confidence, exp_move, prediction bands).

Key features
------------
• Robust Excel loader for daily simple returns (supports strings like "0.56%")
• Returns hygiene: winsorization (±10% default)
• Features from returns only (AR(1), sign persistence, momentum, vol, z-score)
• Labels: mean-reversion vs momentum with a vol-scaled gate per horizon
• Walk-forward by trading-day counts with horizon-aware purge + optional embargo
• Inner CV hyperparameter tuning (AUC); OOS isotonic calibration (no leakage)
• Magnitude head (|move|) per regime + conformal prediction intervals
• Clean per-horizon outputs (predictions & per-ticker metrics) + run_meta.json

Requirements
------------
  pip install pandas numpy scikit-learn openpyxl joblib matplotlib tqdm
(Optional tree models):
  pip install xgboost lightgbm
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingRegressor

# Optional tqdm
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable

ABSTAIN = -1
THIS_DIR = Path(__file__).resolve().parent


# =========================
# Config
# =========================
@dataclass
class Config:
    # I/O
    mode: str = "backtest"  # "backtest" or "forecast"
    excel_path: str | Path = "MXUS_Data.xlsx"
    sheet_name: str = "Returns"
    output_dir: str | Path = "results"

    # Market (optional)
    with_market: bool = True
    market_sheet: str = "Market"
    market_source: str = "yahoo_price"  # ['yahoo_price','yahoo_tr','stooq','fred']
    market_start: str = "1990-01-01"

    # Labeling (vol-scaled gate)
    vol_window: int = 20
    vol_k: float = 0.5  # gate = vol_k * rolling_std * sqrt(h)

    # Walk-forward
    train_years: int = 3
    validation_years: int = 1
    trading_days_per_year: int = 252
    embargo_days: int = 5

    # Model grids (logit by default)
    alpha_grid: List[float] = field(default_factory=lambda: [0.0, 1.0])
    reg_strengths_C: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    inner_cv_folds: int = 3

    # Optional tree grids
    xgb_param_grid: List[Dict[str, Union[int, float]]] = field(
        default_factory=lambda: [
            {"max_depth": 3, "n_estimators": 400, "learning_rate": 0.05},
        ]
    )
    lgb_param_grid: List[Dict[str, Union[int, float]]] = field(
        default_factory=lambda: [
            {"num_leaves": 31, "n_estimators": 400, "learning_rate": 0.05},
        ]
    )

    # Trading-day horizons: (name, forward_h, momentum_lookback_k)
    horizons: List[Tuple[str, int, int]] = field(
        default_factory=lambda: [
            ("d1", 1, 1),
            ("d3", 3, 3),
            ("d7", 7, 7),
            ("d15", 15, 15),
            ("d30", 30, 30),
        ]
    )

    # Feature windows
    ar_windows: List[int] = field(default_factory=lambda: [10, 20, 60])
    sign_persist_windows: List[int] = field(default_factory=lambda: [20, 60])
    vol_feat_windows: List[int] = field(default_factory=lambda: [10, 20])
    z_score_windows: List[int] = field(default_factory=lambda: [20])

    winsor_abs_limit: float = 0.10
    random_state: int = 42

    # Thresholding / coverage control
    threshold_grid: List[float] = field(default_factory=lambda: [round(x, 2) for x in np.arange(0.55, 0.91, 0.01)])
    min_coverage: Optional[float] = 0.40     # e.g., 0.40 for 40% minimum coverage (for decisions)
    threshold_fixed: Optional[float] = None  # fixed τ if provided (for decisions)
    min_exp_move_abs: Optional[float] = 0.01 # e.g., 0.01 for 1% minimum move
    threshold_rule: str = "coverage"         # 'coverage' or 'none'
    n_jobs: int = -1                         # parallelism for param scoring
    min_valid_frac: float = 0.70             # min non-NaN fraction per ticker

    # Calibration & magnitude
    calib_frac: float = 0.20      # held-out slice inside outer-train (by date)
    conformal_alpha: float = 0.10 # 90% prediction intervals

    # Magnitude model (fast, stable defaults)
    hgb_max_depth: int = 3
    hgb_max_iter: int = 300
    hgb_lr: float = 0.05
    hgb_l2: float = 1.0

    # Verbosity
    verbose: bool = True
    show_progress: bool = True


# =========================
# Logging
# =========================
def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mr_prod")
    logger.setLevel(logging.INFO)
    # Clear old handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(out_dir / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def get_unique_filepath(filepath: Path) -> Path:
    """If a file path exists, appends a version number (_v1, _v2, etc.)."""
    if not filepath.exists():
        return filepath
    parent = filepath.parent
    stem = filepath.stem
    suffix = filepath.suffix
    version = 1
    while True:
        new_filepath = parent / f"{stem}_v{version}{suffix}"
        if not new_filepath.exists():
            return new_filepath
        version += 1

# =========================
# Data utils
# =========================
def _parse_percentish(series: pd.Series) -> pd.Series:
    """Convert strings like '0.56%' or '0.56' to decimals; robust to mixed cells.
    Heuristic: if 90th abs-quantile suggests values are in [0, 100], divide by 100.
    """
    s = pd.to_numeric(series.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")
    abs_q90 = s.abs().quantile(0.90)
    abs_q95 = s.abs().quantile(0.95)
    abs_q99 = s.abs().quantile(0.99)
    if (0.20 < abs_q90 <= 100) or (abs_q95 > 1.0 and abs_q99 <= 100):
        s = s / 100.0
    return s


def load_returns_wide(excel_path: str | Path, sheet_name: str) -> pd.DataFrame:
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Read raw (no header); your file has junk rows above the real header
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, engine="openpyxl")

    # Find the row whose first cell is 'Date' (case-insensitive)
    col0 = raw.iloc[:, 0].astype(str).str.strip().str.lower()
    header_idx_candidates = col0[col0.eq("date")].index.tolist()
    if not header_idx_candidates:
        raise ValueError("Could not find a header row with 'Date' in the first column.")
    header_idx = header_idx_candidates[0]

    # Promote that row to headers and drop rows above it
    headers = raw.iloc[header_idx].astype(str).str.strip().tolist()
    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = headers

    # Normalize/ensure 'Date' column name, then parse dates
    if "Date" not in df.columns:
        # just in case the case/spacing differs
        date_col = [c for c in df.columns if str(c).strip().lower() == "date"]
        if date_col:
            df = df.rename(columns={date_col[0]: "Date"})
        else:
            # if somehow missing, force first column to Date
            df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].sort_values("Date").reset_index(drop=True)

    # Convert all ticker columns to numeric simple returns (handles "0.56%" too)
    for col in [c for c in df.columns if c != "Date"]:
        df[col] = _parse_percentish(df[col])

    # Drop completely-empty ticker columns (if any)
    keep = ["Date"] + [c for c in df.columns if c != "Date" and df[c].notna().sum() > 0]
    df = df[keep]

    return df


def validate_returns_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Basic QC: duplicate/non-monotone dates, all-NaN tickers."""
    issues = []
    dup = int(df["Date"].duplicated().sum())
    if dup:
        issues.append(dict(issue="duplicate_dates", count=dup))
    if not df["Date"].is_monotonic_increasing:
        issues.append(dict(issue="non_monotone_dates", count=int(len(df))))
    bad = [c for c in df.columns if c != "Date" and df[c].notna().sum() == 0]
    if bad:
        issues.append(dict(issue="all_nan_tickers", tickers=",".join(bad), count=len(bad)))
    return pd.DataFrame(issues)


def winsorize_simple_returns(series: pd.Series, limit_abs: float) -> pd.Series:
    return series.clip(lower=-limit_abs, upper=limit_abs)


def compound_forward_simple(simple_returns: pd.Series, horizon: int) -> pd.Series:
    cp = (1.0 + simple_returns).cumprod()
    out = cp.shift(-horizon) / cp - 1.0
    if horizon > 0:
        out.iloc[-horizon:] = np.nan
    return out


def compound_past_simple(simple_returns: pd.Series, window: int) -> pd.Series:
    past = (1.0 + simple_returns).rolling(window).apply(np.prod, raw=True) - 1.0
    return past.shift(1)


def rolling_autocorr(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    x_prev = series.shift(lag)
    corr = series.rolling(window).corr(x_prev)
    return corr.shift(1)


def rolling_sign_persistence(series: pd.Series, window: int) -> pd.Series:
    same_sign = (np.sign(series) * np.sign(series.shift(1))) > 0
    rate = same_sign.rolling(window).mean()
    return rate.shift(1)


# =========================
# Market context (optional S&P 500)
# =========================
def load_market_from_excel(excel_path: str | Path, sheet_name: str = "Market", logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    except Exception:
        return None
    if df.empty or df.shape[1] < 2:
        return None
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].sort_values("Date").reset_index(drop=True)

    cols = [c for c in df.columns if c != "Date"]
    ret = None
    if "MKT_RET" in df.columns:
        ret = pd.to_numeric(df["MKT_RET"], errors="coerce")
    else:
        candidates = [c for c in cols if c.lower() in ("mkt_px","px","close","adj close","price","value")]
        if not candidates and len(cols) == 1:
            candidates = [cols[0]]
        if candidates:
            px = pd.to_numeric(df[candidates[0]], errors="coerce")
            ret = px.pct_change()
    if ret is None:
        if logger: logger.warning("Market sheet present but couldn't infer MKT_RET — skipping market features.")
        return None
    out = pd.DataFrame({"Date": df["Date"], "MKT_RET": ret})
    return out.dropna()


def get_sp500_market(source: str = "yahoo_price", start: str = "1990-01-01", end: Optional[str] = None, logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    try:
        if end is None:
            end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        if source == "yahoo_price":
            import yfinance as yf
            s = yf.download("^GSPC", start=start, end=end, progress=False)["Adj Close"].rename("MKT_PX")
        elif source == "yahoo_tr":
            import yfinance as yf
            s = yf.download("^SP500TR", start=start, end=end, progress=False)["Adj Close"].rename("MKT_PX")
        elif source == "stooq":
            url = "https://stooq.com/q/d/l/?s=%5Espx&i=d"
            df = pd.read_csv(url)
            df["Date"] = pd.to_datetime(df["Date"])
            s = df.set_index("Date")["Close"].rename("MKT_PX").sort_index()
        elif source == "fred":
            from fredapi import Fred
            fred = Fred(api_key=os.getenv("FRED_API_KEY"))
            s = fred.get_series("SP500", observation_start=start, observation_end=end).rename("MKT_PX")
            s.index = pd.to_datetime(s.index)
        else:
            if logger: logger.warning(f"Unknown market source '{source}' — skipping market features.")
            return None
    except Exception as e:
        if logger: logger.warning(f"Failed to fetch market data via {source}: {e}")
        return None
    mkt = s.dropna().to_frame()
    mkt["MKT_RET"] = mkt["MKT_PX"].pct_change()
    out = mkt.reset_index().rename(columns={"index": "Date"})
    return out.dropna()


def compute_market_features(market_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = market_df.copy().sort_values("Date")
    df["MKT_RET_1D"] = df["MKT_RET"]
    df["MKT_RET_5D"] = (1.0 + df["MKT_RET"]).rolling(5).apply(lambda x: np.prod(x) - 1.0, raw=True)
    df["MKT_VOL_20D"] = df["MKT_RET"].rolling(20).std()
    px_index = (1.0 + df["MKT_RET"].fillna(0)).cumprod()
    w = 20
    ma = px_index.rolling(w, min_periods=w).mean()
    sd = px_index.rolling(w, min_periods=w).std()
    df["MKT_Z_MA_20D"] = (px_index - ma) / sd
    for c in ["MKT_RET_1D","MKT_RET_5D","MKT_VOL_20D","MKT_Z_MA_20D"]:
        df[c] = df[c].shift(1)
    return df[["Date","MKT_RET_1D","MKT_RET_5D","MKT_VOL_20D","MKT_Z_MA_20D"]].dropna()


# =========================
# Feature panel + labels (vol-scaled gate)
# =========================
def build_feature_panel(returns_wide: pd.DataFrame, cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    tickers = [c for c in returns_wide.columns if c != "Date"]
    frames: List[pd.DataFrame] = []
    for ticker in tqdm(tickers, desc="Building features (tickers)", disable=not (cfg.verbose and cfg.show_progress)):
        r = returns_wide[ticker].astype(float)
        if r.notna().mean() < cfg.min_valid_frac:  # drop sparse series
            continue
        r = winsorize_simple_returns(r, cfg.winsor_abs_limit)
        base = pd.DataFrame({"Date": returns_wide["Date"], "ticker": ticker})
        feats: Dict[str, pd.Series] = {}
        # AR(1)
        for w in cfg.ar_windows:
            feats[f"ar1_corr_{w}d"] = rolling_autocorr(r, window=w)
        # Sign persistence
        for w in cfg.sign_persist_windows:
            feats[f"sign_persist_{w}d"] = rolling_sign_persistence(r, window=w)
        # Momentum & short history returns
        feats["ret_1d"] = r.shift(1)
        feats["ret_5d"] = compound_past_simple(r, 5)
        feats["ret_20d"] = compound_past_simple(r, 20)
        # Volatility features
        for w in cfg.vol_feat_windows:
            feats[f"vol_{w}d"] = r.rolling(w).std().shift(1)
        # Price-index z-score vs rolling MA
        price_index = (1.0 + r.fillna(0)).cumprod()
        for w in cfg.z_score_windows:
            ma = price_index.rolling(w, min_periods=w).mean()
            sd = price_index.rolling(w, min_periods=w).std()
            feats[f"z_ma_{w}d"] = ((price_index - ma) / sd).shift(1)
        feats_df = pd.DataFrame(feats)
        # Forward returns for all horizons
        fwd = {f"fwd_{h}d": compound_forward_simple(r, h) for _, h, _ in cfg.horizons}
        fwd_df = pd.DataFrame(fwd)
        # Recent momentum windows (k per horizon)
        mom = {f"mom_{k}d": compound_past_simple(r, k) for _, _, k in cfg.horizons}
        mom_df = pd.DataFrame(mom)
        frames.append(pd.concat([base, feats_df, fwd_df, mom_df], axis=1))

    if not frames:
        raise RuntimeError("No usable tickers after cleaning.")

    panel = pd.concat(frames, ignore_index=True)

    # Cross-sectional ranks (for interpretability/features)
    for col in ["ret_1d", "ret_5d", "ret_20d"] + [f"z_ma_{w}d" for w in cfg.z_score_windows]:
        if col in panel.columns:
            panel[f"cs_rank_{col}"] = panel.groupby("Date")[col].rank(pct=True)

    # Pre-compute per-ticker daily vol (for gate)
    panel["daily_vol"] = panel.groupby("ticker")["ret_1d"].transform(lambda s: s.rolling(cfg.vol_window).std())

    # Vol-scaled label gate and labels
    for name, h, k in cfg.horizons:
        fwd_col = f"fwd_{h}d"
        mom_col = f"mom_{k}d"
        # Project daily vol to horizon and fill within ticker
        gate = cfg.vol_k * panel["daily_vol"] * np.sqrt(max(h, 1))
        gate = gate.groupby(panel["ticker"]).transform(lambda s: s.fillna(s.median()))
        sign_future = np.sign(panel[fwd_col])
        sign_recent = np.sign(panel[mom_col])
        mask_big = panel[fwd_col].abs() > gate
        label_mr = np.where(
            (mask_big) & (sign_recent != 0) & (sign_future == -sign_recent), 1,
            np.where((mask_big) & (sign_recent != 0), 0, np.nan),
        )
        panel[f"label_mr_{name}"] = label_mr

    # Final feature list: drop targets/labels/mom/fwd/meta
    feature_cols = [
        c for c in panel.columns
        if c not in ("Date", "ticker", "daily_vol")
        and not c.startswith("fwd_") and not c.startswith("mom_") and not c.startswith("label_")
    ]

    # Drop rows with NaN features (model inputs must be clean)
    panel = panel.dropna(subset=feature_cols).reset_index(drop=True)

    logger.info(
        f"Panel: {panel.shape[0]:,} rows × {panel.shape[1]} cols  "
        f"({panel['Date'].min().date()} to {panel['Date'].max().date()})"
    )
    return panel


# =========================
# Splits
# =========================
def walk_forward_windows(dates: pd.Series, train_years: int, val_years: int, tdy: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    unique_dates = pd.to_datetime(pd.Series(dates.unique())).sort_values().to_numpy()
    n_train = int(train_years * tdy)
    n_val = int(val_years * tdy)
    starts = np.arange(0, len(unique_dates) - (n_train + n_val) + 1, n_val, dtype=int)
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for s in starts:
        tr = unique_dates[s : s + n_train]
        va = unique_dates[s + n_train : s + n_train + n_val]
        if tr.size == 0 or va.size == 0:
            continue
        windows.append((pd.Timestamp(tr[0]), pd.Timestamp(tr[-1]), pd.Timestamp(va[0]), pd.Timestamp(va[-1])))
    return windows


def purged_inner_splits(dates: np.ndarray, n_splits: int, purge_days: int):
    """TimeSeriesSplit with a *pre-validation purge* of `purge_days` on the train side."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    dates = pd.to_datetime(dates)
    for tr_idx, val_idx in tscv.split(dates):
        if val_idx.size == 0 or tr_idx.size == 0:
            continue
        val_start = pd.Timestamp(dates[val_idx].min())
        purge_cut = val_start - BDay(purge_days)
        tr_dates = dates[tr_idx]
        keep = (tr_dates <= purge_cut) & (tr_dates < val_start)
        tr_idx_purged = tr_idx[keep]
        if tr_idx_purged.size == 0:
            continue
        yield tr_idx_purged, val_idx


# =========================
# Models & calibration
# =========================
def make_logit(alpha: float, C: float, cfg: Config) -> Pipeline:
    if alpha == 0.0:
        logit = LogisticRegression(penalty="l2", solver="lbfgs", C=C, class_weight="balanced", max_iter=10_000, random_state=cfg.random_state)
    elif alpha == 1.0:
        logit = LogisticRegression(penalty="l1", solver="saga", C=C, class_weight="balanced", max_iter=10_000, random_state=cfg.random_state)
    else:
        logit = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=alpha, C=C, class_weight="balanced", max_iter=10_000, random_state=cfg.random_state)
    return Pipeline([("scale", StandardScaler()), ("logit", logit)])


def make_tree(model_name: str, params: Dict[str, Union[int, float]], cfg: Config):
    if model_name == "xgb":
        import xgboost as xgb
        defaults = dict(objective="binary:logistic", eval_metric="logloss", n_jobs=-1, tree_method="hist",
                        random_state=cfg.random_state, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
        defaults.update(params)
        return xgb.XGBClassifier(**defaults)
    if model_name == "lgb":
        import lightgbm as lgb
        defaults = dict(objective="binary", n_jobs=-1, random_state=cfg.random_state, subsample=0.8,
                        colsample_bytree=0.8, reg_lambda=1.0, learning_rate=0.05, is_unbalance=True)
        defaults.update(params)
        return lgb.LGBMClassifier(**defaults)
    raise ValueError("Unknown tree model")


def build_model(name: str, params: Dict[str, Union[int, float]], cfg: Config):
    if name == "logit":
        return make_logit(params.get("alpha", 0.0), params.get("C", 1.0), cfg)
    if name in ("xgb", "lgb"):
        return make_tree(name, params, cfg)
    raise ValueError("Unknown model name")


def auc_safe(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    keep = np.isin(y, [0, 1]) & np.isfinite(p)
    y = y[keep]
    p = p[keep]
    if y.size == 0 or np.unique(y).size < 2 or np.allclose(np.nanstd(p), 0.0):
        return 0.5
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return 0.5


def fit_isotonic_safe(probs: np.ndarray, y_true: np.ndarray) -> Optional[IsotonicRegression]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_true)
    keep = np.isfinite(p) & np.isin(y, [0, 1])
    p = p[keep]
    y = y[keep]
    if p.size < 30 or np.unique(y).size < 2:
        return None
    try:
        iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return iso
    except Exception:
        return None


def apply_calibrator(iso: Optional[IsotonicRegression], probs: np.ndarray) -> np.ndarray:
    if iso is None:
        return np.asarray(probs, dtype=float)
    try:
        return iso.predict(np.asarray(probs, dtype=float))
    except Exception:
        return np.asarray(probs, dtype=float)


# =========================
# Threshold selection (coverage-driven)
# =========================
def choose_threshold_by_coverage(p_mr: np.ndarray, thresholds: List[float], min_coverage: float) -> Tuple[float, float]:
    """Pick the *largest* τ whose symmetric coverage meets/exceeds `min_coverage`.
    Coverage(τ) = mean( p>=τ or p<=1-τ ). If none meet, return τ=0.5 (coverage=1.0).
    """
    p = np.asarray(p_mr, dtype=float)
    for threshold in sorted([t for t in thresholds if t > 0.5], reverse=True):
        covered = (p >= threshold) | (p <= (1.0 - threshold))
        cov = float(np.mean(covered))
        if cov + 1e-12 >= float(min_coverage):
            return float(threshold), cov
    return 0.5, 1.0


# =========================
# Magnitude head + conformal
# =========================
def make_hgbr(cfg: Config) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=cfg.hgb_max_depth,
        max_iter=cfg.hgb_max_iter,
        learning_rate=cfg.hgb_lr,
        l2_regularization=cfg.hgb_l2,
        random_state=cfg.random_state,
    )


def conformal_q(residuals: np.ndarray, alpha: float) -> float:
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    n = res.size
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(res, k - 1)[k - 1])


# =========================
# Backtest: core run (one horizon) — existing pipeline
# =========================
def run_one_horizon(panel: pd.DataFrame, horizon_name: str, cfg: Config, model_name: str, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_col = f"label_mr_{horizon_name}"
    feat_cols = [c for c in panel.columns if c not in ("Date","ticker","daily_vol") and not c.startswith("fwd_") and not c.startswith("mom_") and not c.startswith("label_")]

    df = panel.dropna(subset=[label_col]).copy()
    X_all = df[feat_cols].values
    y_all = df[label_col].astype(int).values
    dates_all = df["Date"].to_numpy()

    windows = walk_forward_windows(df["Date"], cfg.train_years, cfg.validation_years, cfg.trading_days_per_year)
    h_days, k_days = next((h, k) for n,h,k in cfg.horizons if n==horizon_name)
    fwd_col = f"fwd_{h_days}d"; mom_col = f"mom_{k_days}d"

    all_preds: List[pd.DataFrame] = []

    def iter_param_grid(name: str):
        if name == "logit":
            for a in cfg.alpha_grid:
                for C in cfg.reg_strengths_C:
                    yield {"alpha": a, "C": float(C)}
        elif name == "xgb":
            for p in cfg.xgb_param_grid:
                yield dict(p)
        elif name == "lgb":
            for p in cfg.lgb_param_grid:
                yield dict(p)
        else:
            raise ValueError("Unknown model name")

    for w_idx, (tr_start, tr_end, va_start, va_end) in enumerate(tqdm(windows, desc=f"{horizon_name}: windows", disable=not (cfg.verbose and cfg.show_progress)), start=1):
        # Horizon-aware purge & optional embargo
        train_cutoff = tr_end - BDay(h_days)
        test_start_adj = va_start + BDay(cfg.embargo_days)
        test_cutoff = va_end - BDay(h_days)

        in_tr = (dates_all >= tr_start) & (dates_all <= train_cutoff)
        in_te = (dates_all >= test_start_adj) & (dates_all <= test_cutoff)

        if not in_tr.any() or not in_te.any():
            logger.info(f"{horizon_name} W{w_idx:02d} skipped (empty split).")
            continue

        X_tr, y_tr = X_all[in_tr], y_all[in_tr]
        X_te, y_te = X_all[in_te], y_all[in_te]
        dates_tr = dates_all[in_tr]; dates_te = df.loc[in_te, "Date"].values
        tickers_te = df.loc[in_te, "ticker"].values
        fwd_tr = df.loc[in_tr, fwd_col].values; fwd_te = df.loc[in_te, fwd_col].values
        mom_tr = np.sign(df.loc[in_tr, mom_col].values); mom_te = np.sign(df.loc[in_te, mom_col].values)

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            logger.info(f"{horizon_name} W{w_idx:02d} skipped (need both classes).")
            continue

        # 1) Inner CV to pick params (AUC) using purged folds
        best_params = None; best_auc = -np.inf
        for params in iter_param_grid(model_name):
            y_prob_all = []; y_true_all = []
            for tr_i, va_i in purged_inner_splits(dates_tr, cfg.inner_cv_folds, purge_days=h_days):
                model = build_model(model_name, params, cfg)
                model.fit(X_tr[tr_i], y_tr[tr_i])
                y_prob_all.append(model.predict_proba(X_tr[va_i])[:,1])
                y_true_all.append(y_tr[va_i])
            if not y_true_all:
                continue
            y_prob_cv = np.concatenate(y_prob_all)
            y_true_cv = np.concatenate(y_true_all)
            if y_true_cv.size == 0 or np.unique(y_true_cv).size < 2:
                continue
            auc_cv = auc_safe(y_true_cv, y_prob_cv)
            if auc_cv > best_auc:
                best_auc = auc_cv; best_params = params
        if best_params is None:
            best_params = {"alpha":0.0, "C":1.0} if model_name=="logit" else (cfg.xgb_param_grid[0] if model_name=="xgb" else cfg.lgb_param_grid[0])

        # 1b) Parallel param scoring (AUC + PR-AUC, and F1@coverage if requested) — overrides best_params
        def evaluate_params(params):
            aucs, prs, f1s = [], [], []
            for tr_i, va_i in purged_inner_splits(dates_tr, cfg.inner_cv_folds, purge_days=h_days):
                mdl = build_model(model_name, params, cfg)
                mdl.fit(X_tr[tr_i], y_tr[tr_i])
                pro = mdl.predict_proba(X_tr[va_i])[:, 1]
                yv  = y_tr[va_i]
                aucs.append(auc_safe(yv, pro))
                try:
                    prs.append(float(average_precision_score(yv, pro)))
                except Exception:
                    prs.append(0.0)
                if cfg.min_coverage is not None:
                    threshold_cv, _ = choose_threshold_by_coverage(pro, cfg.threshold_grid, cfg.min_coverage)
                    hi = pro >= threshold_cv; lo = pro <= (1.0 - threshold_cv)
                    mask = (hi | lo)
                    if mask.any() and np.unique(yv[mask]).size == 2:
                        yp = np.full_like(pro, ABSTAIN, int)
                        yp[hi], yp[lo] = 1, 0
                        from sklearn.metrics import f1_score as _f1
                        f1s.append(float(_f1(yv[mask], yp[mask], zero_division=0)))
            if cfg.min_coverage is not None and f1s:
                auc_score = np.nanmean(aucs) if aucs else 0.0
                f1_score = np.nanmean(f1s) if f1s else 0.0
                score = auc_score * 0.5 + f1_score * 0.5
            else:
                auc_score = np.nanmean(aucs) if aucs else 0.0
                pr_score = np.nanmean(prs) if prs else 0.0
                score = auc_score * 0.3 + pr_score * 0.7
            return score, params

        candidates = list(iter_param_grid(model_name))
        if candidates:
            scored = Parallel(n_jobs=cfg.n_jobs, prefer="threads")(delayed(evaluate_params)(p) for p in candidates)
            try:
                best_score, best_params = max(scored, key=lambda t: (t[0], ))
            except Exception:
                pass

        # 2) Build OOS probabilities for OUTER-TRAIN using purged inner folds (no leakage)
        oos_prob = np.full(y_tr.shape[0], np.nan, dtype=float)
        for tr_i, va_i in purged_inner_splits(dates_tr, cfg.inner_cv_folds, purge_days=h_days):
            mdl = build_model(model_name, best_params, cfg)
            mdl.fit(X_tr[tr_i], y_tr[tr_i])
            oos_prob[va_i] = mdl.predict_proba(X_tr[va_i])[:, 1]

        # 3) Isotonic on OOS predictions
        iso = fit_isotonic_safe(oos_prob, y_tr)

        # 4) Temporal calibration split inside OUTER-TRAIN (for magnitude head + reporting)
        d_tr = pd.to_datetime(pd.Series(dates_tr)); uniq = d_tr.drop_duplicates().sort_values()
        n_cal = max(int(len(uniq) * cfg.calib_frac), 30)
        cutoff = uniq.iloc[-n_cal] if len(uniq) > n_cal else uniq.iloc[-1]
        calib_mask = (d_tr >= cutoff).to_numpy(); model_mask = ~calib_mask

        X_tr_model, y_tr_model = X_tr[model_mask], y_tr[model_mask]
        X_tr_calib, y_tr_calib = X_tr[calib_mask], y_tr[calib_mask]
        fwd_tr = df.loc[in_tr, fwd_col].values
        fwd_tr_calib = fwd_tr[calib_mask]

        # 5) Final classifier fit on FULL OUTER-TRAIN; calibrated probs on TEST
        clf_full = build_model(model_name, best_params, cfg)
        clf_full.fit(X_tr, y_tr)
        p_raw = clf_full.predict_proba(X_te)[:, 1]
        p = apply_calibrator(iso, p_raw)

        # 6) Magnitude head (two regressors on MODEL-SLICE)
        y_abs_model = np.abs(fwd_tr[model_mask])
        mr_mask_model = y_tr_model == 1; mo_mask_model = y_tr_model == 0

        def fit_reg(Xm, ym):
            if Xm.shape[0] < 50 or np.sum(np.isfinite(ym)) < 30 or float(np.nanstd(ym)) == 0.0:
                return None, float(np.nan)
            reg = make_hgbr(cfg); reg.fit(Xm, ym); return reg, float(np.nanmedian(ym))

        reg_mr, med_mr = fit_reg(X_tr_model[mr_mask_model], y_abs_model[mr_mask_model])
        reg_mo, med_mo = fit_reg(X_tr_model[mo_mask_model], y_abs_model[mo_mask_model])

        def predict_or_const(model, X, const):
            if model is None:
                return np.full(X.shape[0], const if np.isfinite(const) else 0.0)
            try:
                return model.predict(X)
            except Exception:
                return np.full(X.shape[0], const if np.isfinite(const) else 0.0)

        pred_mr_abs = predict_or_const(reg_mr, X_te, med_mr)
        pred_mo_abs = predict_or_const(reg_mo, X_te, med_mo)

        # Expected signed move = mixture of regime magnitudes with regime signs
        exp_move = p * (-np.sign(mom_te)) * pred_mr_abs + (1.0-p) * (np.sign(mom_te)) * pred_mo_abs

        # 7) Conformal bands from CALIB slice residuals (conservative)
        res_mr = np.abs(
            np.abs(fwd_tr_calib[y_tr_calib==1]) -
            predict_or_const(reg_mr, X_tr_calib[y_tr_calib==1], med_mr)
        )
        res_mo = np.abs(
            np.abs(fwd_tr_calib[y_tr_calib==0]) -
            predict_or_const(reg_mo, X_tr_calib[y_tr_calib==0], med_mo)
        )
        q_mr = conformal_q(res_mr, cfg.conformal_alpha)
        q_mo = conformal_q(res_mo, cfg.conformal_alpha)

        low_move  = p * (-np.sign(mom_te)) * (pred_mr_abs - q_mr) + (1.0-p) * ( np.sign(mom_te)) * (pred_mo_abs - q_mo)
        high_move = p * (-np.sign(mom_te)) * (pred_mr_abs + q_mr) + (1.0-p) * ( np.sign(mom_te)) * (pred_mo_abs + q_mo)

        # Package predictions for this window
        mr_score = 1.0 - 2.0*p  # [-1,1] (negative implies MR direction expected)
        action_conf = np.maximum(p, 1.0-p)

        # Optional coverage-driven decisioning
        threshold_used = np.nan
        y_pred = np.full(p.shape[0], ABSTAIN, dtype=int)
        abstained = np.ones(p.shape[0], dtype=bool)
        if cfg.threshold_fixed is not None:
            threshold_used = float(cfg.threshold_fixed)
            hi = p >= threshold_used
            lo = p <= (1.0 - threshold_used)
        elif cfg.min_coverage is not None and cfg.threshold_rule == "coverage":
            threshold_used, _ = choose_threshold_by_coverage(p, cfg.threshold_grid, cfg.min_coverage)
            hi = p >= threshold_used
            lo = p <= (1.0 - threshold_used)
        else:
            hi = lo = np.zeros_like(p, dtype=bool)
        if hi.any() or lo.any():
            y_pred[hi] = 1  # MR
            y_pred[lo] = 0  # Momentum
            abstained = ~(hi | lo)

        preds = pd.DataFrame({
            "Date": dates_te,
            "ticker": tickers_te,
            "horizon": horizon_name,
            "y_true": y_te,
            "fwd_ret": fwd_te,
            "mom_sign": np.sign(mom_te),
            "y_prob": p_raw,
            "y_prob_cal": p,
            "mr_score": mr_score,
            "confidence": action_conf,
            "exp_move": exp_move,
            "pi_low": low_move,
            "pi_high": high_move,
            "threshold": threshold_used,
            "y_pred": y_pred,
            "abstained": abstained,
        })
        all_preds.append(preds)

    if not all_preds:
        logger.warning(f"No predictions for horizon {horizon_name} (data/splits insufficient).")
        return pd.DataFrame(), pd.DataFrame()

    preds_all = pd.concat(all_preds, ignore_index=True)

    # Per-ticker metrics
    rows = []
    for tkr, g in preds_all.groupby("ticker", sort=False):
        yt = g["y_true"].to_numpy()
        pp = g["y_prob_cal"].to_numpy()
        aucv = float(auc_safe(yt, pp)) if yt.size > 0 else np.nan
        if cfg.min_coverage is not None and "abstained" in g.columns:
            gg = g[~g["abstained"]]
            yt2 = gg["y_true"].to_numpy()
            yp2 = gg["y_pred"].to_numpy() if not gg.empty else np.array([])
            cov = 1.0 - float(g["abstained"].mean()) if not g.empty else np.nan
            if yt2.size == 0 or np.unique(yt2).size < 2:
                rows.append(dict(ticker=tkr, support=int(yt.size), covered=int(yt2.size), coverage=cov, accuracy=np.nan, precision=np.nan, recall=np.nan, f1=np.nan, auc=aucv))
                continue
            rows.append(dict(
                ticker=tkr,
                support=int(yt.size),
                covered=int(yt2.size),
                coverage=cov,
                accuracy=float((yt2 == yp2).mean()),
                precision=float(precision_score(yt2, yp2, zero_division=0)),
                recall=float(recall_score(yt2, yp2, zero_division=0)),
                f1=float(f1_score(yt2, yp2, zero_division=0)),
                auc=aucv,
            ))
        else:
            if yt.size == 0 or np.unique(yt).size < 2:
                rows.append(dict(ticker=tkr, support=int(yt.size), accuracy=np.nan, precision=np.nan, recall=np.nan, f1=np.nan, auc=aucv))
                continue
            yp = (pp >= 0.5).astype(int)
            rows.append(dict(
                ticker=tkr,
                support=int(yt.size),
                accuracy=float((yt == yp).mean()),
                precision=float(precision_score(yt, yp, zero_division=0)),
                recall=float(recall_score(yt, yp, zero_division=0)),
                f1=float(f1_score(yt, yp, zero_division=0)),
                auc=aucv,
            ))
    metrics = pd.DataFrame(rows).sort_values("ticker")

    return preds_all, metrics


# =========================
# I/O autodiscovery
# =========================
def autodiscover_excel_and_sheet(excel_arg: Optional[Union[str, Path]], sheet_arg: Optional[str], logger: Optional[logging.Logger] = None) -> Tuple[Path, str]:
    """Pick a sensible Excel file and sheet when not fully specified."""
    def choose_sheet(xls: pd.ExcelFile, preferred: Optional[str]) -> str:
        names = list(xls.sheet_names)
        if preferred and preferred in names:
            return preferred
        if "Returns" in names:
            return "Returns"
        for n in names:
            if "return" in n.lower():
                return n
        return names[0]

    # Case 1: user provided a path
    if excel_arg:
        p = Path(excel_arg)
        if p.exists():
            try:
                xls = pd.ExcelFile(p)
                chosen = choose_sheet(xls, sheet_arg)
                if logger:
                    logger.info(f"Using Excel: {p.name} | sheet: {chosen}")
                return p, chosen
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to open {p}: {e}")
        else:
            if logger:
                logger.warning(f"Excel path not found: {p}. Will try autodiscovery.")

    # Case 2: autodiscover in CWD
    patterns = ["MXUS_Data_*.xlsx", "MXUS_Data.xlsx", "*.xlsx"]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(sorted(Path.cwd().glob(pat), key=lambda q: q.stat().st_mtime, reverse=True))
        if candidates:
            break
    for p in candidates:
        try:
            xls = pd.ExcelFile(p)
            chosen = choose_sheet(xls, sheet_arg)
            if logger:
                logger.info(f"Autodiscovered Excel: {p.name} | sheet: {chosen}")
            return p, chosen
        except Exception:
            continue

    raise FileNotFoundError("Excel file not specified and not found (looked for MXUS_Data_*.xlsx, MXUS_Data.xlsx, or any .xlsx). Provide --excel or place the file in the working directory.")


# =========================
# Backtest pipeline (existing)
# =========================
def run_pipeline(cfg: Config, model_name: str) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir)

    # Deterministic seeds
    np.random.seed(cfg.random_state)

    logger.info("Resolving input Excel + sheet…")
    excel_path, sheet_name = autodiscover_excel_and_sheet(cfg.excel_path, cfg.sheet_name, logger)
    cfg.excel_path, cfg.sheet_name = excel_path, sheet_name

    logger.info("Loading returns (wide)...")
    returns_wide = load_returns_wide(cfg.excel_path, cfg.sheet_name)

    # QC + enforce monotone/no-dup
    qc = validate_returns_wide(returns_wide)
    if not qc.empty:
        qc.to_csv(out_dir / "data_qc.csv", index=False)
    returns_wide = returns_wide.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

    # Optional market context
    market_feats = None
    if cfg.with_market:
        logger.info("Loading market context…")
        mkt_df = load_market_from_excel(cfg.excel_path, sheet_name=cfg.market_sheet, logger=logger)
        if mkt_df is None:
            logger.info(f"No usable '{cfg.market_sheet}' sheet. Fetching S&P 500 via {cfg.market_source}…")
            mkt_df = get_sp500_market(cfg.market_source, start=cfg.market_start, logger=logger)
        if mkt_df is not None and not mkt_df.empty:
            market_feats = compute_market_features(mkt_df, cfg)
            logger.info(f"Market features ready: {list(market_feats.columns.drop('Date'))}")
        else:
            logger.warning("Proceeding without market features (fetch or sheet failed).")

    logger.info("Building feature panel + labels...")
    t0 = time.time()
    panel = build_feature_panel(returns_wide, cfg, logger)
    if market_feats is not None:
        panel = panel.merge(market_feats, on="Date", how="left")
        logger.info("Merged market features into panel.")
    logger.info(f"Feature panel built in {time.time()-t0:.2f}s")

    # Save feature column list for reference
    feat_cols = [
        c for c in panel.columns
        if c not in ("Date","ticker","daily_vol") and not c.startswith("fwd_") and not c.startswith("mom_") and not c.startswith("label_")
    ]
    (out_dir / "feature_columns.txt").write_text("\n".join(feat_cols), encoding="utf-8")

    # Run each horizon
    summary_rows = []
    for name, _, _ in cfg.horizons:
        logger.info(f"=== Horizon: {name} ===")
        preds, metrics = run_one_horizon(panel, name, cfg, model_name, logger)
        if preds.empty:
            logger.warning(f"No results for horizon {name} — skipping saves.")
            continue
        preds_path = get_unique_filepath(out_dir / f"predictions_{name}.csv")
        metrics_path = get_unique_filepath(out_dir / f"metrics_{name}.csv")
        preds.sort_values(["Date","ticker"]).to_csv(preds_path, index=False)
        metrics.to_csv(metrics_path, index=False)
        logger.info(f"Saved: {preds_path.name} ({len(preds):,} rows), {metrics_path.name} ({len(metrics):,} rows)")
        
        # Quick horizon summary
                # Quick horizon summary with both accuracies + counts
        yt = preds["y_true"].to_numpy()
        pp = preds["y_prob_cal"].to_numpy()
        finite = np.isfinite(pp)

        # AUC over all finite probs
        auc_all = auc_safe(yt[finite], pp[finite]) if finite.any() else float("nan")

        # Full-sample accuracy (0.5 cutoff on calibrated prob)
        if finite.any():
            yp_full = (pp[finite] >= 0.5).astype(int)
            accuracy_full = float((yt[finite] == yp_full).mean())
            coverage_full = float(finite.mean())
        else:
            accuracy_full = float("nan")
            coverage_full = 0.0

        # Covered/decision accuracy (only when decisions exist)
        accuracy_covered = float("nan")
        coverage = coverage_full
        covered_rows = 0

        if "abstained" in preds.columns:
            covered_mask = (~preds["abstained"]).to_numpy() & finite
            coverage = float(covered_mask.mean())
            covered_rows = int(covered_mask.sum())
            if covered_rows > 0:
                yt_cov = preds.loc[covered_mask, "y_true"].to_numpy()
                yp_cov = preds.loc[covered_mask, "y_pred"].to_numpy()
                accuracy_covered = float((yt_cov == yp_cov).mean())

        # Extra counts
        n_unique_tickers = preds["ticker"].nunique() if "ticker" in preds.columns else 0
        n_unique_dates = preds["Date"].nunique() if "Date" in preds.columns else 0

        summary_rows.append(dict(
            horizon=name,
            rows=int(len(preds)),
            n_unique_tickers=n_unique_tickers,
            n_unique_dates=n_unique_dates,
            auc=auc_all,
            coverage=coverage,
            accuracy_full=accuracy_full,
            accuracy_covered=accuracy_covered,
            covered_rows=covered_rows,
        ))


    if summary_rows:
        summary_path = get_unique_filepath(out_dir / "summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    # Run meta
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "excel": str(cfg.excel_path),
        "sheet": cfg.sheet_name,
        "model": model_name,
        "mode": "backtest",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
    }
    meta_path = get_unique_filepath(out_dir / "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info("Done. Artifacts: predictions_*.csv, metrics_*.csv, summary.csv, run_meta.json")


# =========================
# NEW: Forecast mode (train on most recent window; predict next day)
# =========================
def _iter_param_grid_forecast(model_name: str, cfg: Config):
    if model_name == "logit":
        for a in cfg.alpha_grid:
            for C in cfg.reg_strengths_C:
                yield {"alpha": a, "C": float(C)}
    elif model_name == "xgb":
        for p in cfg.xgb_param_grid:
            yield dict(p)
    elif model_name == "lgb":
        for p in cfg.lgb_param_grid:
            yield dict(p)
    else:
        raise ValueError("Unknown model name")


def forecast_one_horizon(
    train_panel: pd.DataFrame,
    forecast_panel: pd.DataFrame,
    horizon_name: str,
    cfg: Config,
    model_name: str,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Train on most-recent window; produce calibrated probs, exp_move, bands for the latest date."""
    label_col = f"label_mr_{horizon_name}"
    feat_cols = [c for c in train_panel.columns if c not in ("Date","ticker","daily_vol") and not c.startswith("fwd_") and not c.startswith("mom_") and not c.startswith("label_")]

    df_tr = train_panel.dropna(subset=[label_col]).copy()
    if len(df_tr) < 100 or df_tr[label_col].nunique() < 2:
        logger.warning(f"[{horizon_name}] Skipping: insufficient training data.")
        return pd.DataFrame(), {}

    X_tr, y_tr, dates_tr = df_tr[feat_cols].values, df_tr[label_col].astype(int).values, df_tr["Date"].to_numpy()
    h_days, k_days = next((h, k) for n,h,k in cfg.horizons if n==horizon_name)
    fwd_col, mom_col = f"fwd_{h_days}d", f"mom_{k_days}d"
    fwd_tr = df_tr[fwd_col].values

    X_fc, tickers_fc = forecast_panel[feat_cols].values, forecast_panel["ticker"].values
    mom_fc_sign = np.sign(forecast_panel[mom_col].values)

    # Param search (AUC + PR-AUC) with purged inner splits
    def evaluate_params(params):
        aucs, prs = [], []
        for tr_i, va_i in purged_inner_splits(dates_tr, cfg.inner_cv_folds, purge_days=h_days):
            # Skip if validation fold has insufficient data
            if len(va_i) == 0 or len(tr_i) == 0:
                continue
            if len(np.unique(y_tr[va_i])) < 2:  # Single class in validation
                continue
                
            mdl = build_model(model_name, params, cfg)
            mdl.fit(X_tr[tr_i], y_tr[tr_i])
            pro = mdl.predict_proba(X_tr[va_i])[:, 1]
            aucs.append(auc_safe(y_tr[va_i], pro))
            try:
                prs.append(float(average_precision_score(y_tr[va_i], pro)))
            except Exception:
                prs.append(0.0)
        
        # Guard against empty metric lists
        if not aucs or not prs:
            return -float("inf"), params
            
        score = np.nanmean(aucs) * 0.3 + np.nanmean(prs) * 0.7
        return score, params

    candidates = list(_iter_param_grid_forecast(model_name, cfg))
    best_params = candidates[0] if candidates else {}
    if len(candidates) > 1:
        try:
            scored = Parallel(n_jobs=cfg.n_jobs, prefer="threads")(delayed(evaluate_params)(p) for p in candidates)
            if scored:
                valid_scores = [(score, params) for score, params in scored if score > -float("inf")]
                if valid_scores:
                    _, best_params = max(valid_scores, key=lambda t: t[0])
                    logger.info(f"[{horizon_name}] Best params: {best_params}")
        except Exception as e:
            logger.warning(f"[{horizon_name}] Param search failed, using default. Error: {e}")

    # OOS probs for train slice → isotonic
    oos_prob = np.full(y_tr.shape[0], np.nan, dtype=float)
    for tr_i, va_i in purged_inner_splits(dates_tr, cfg.inner_cv_folds, purge_days=h_days):
        if len(va_i) == 0 or len(tr_i) == 0:
            continue
        mdl = build_model(model_name, best_params, cfg)
        mdl.fit(X_tr[tr_i], y_tr[tr_i])
        oos_prob[va_i] = mdl.predict_proba(X_tr[va_i])[:, 1]
    iso = fit_isotonic_safe(oos_prob, y_tr)

    # Temporal calibration split inside train (for magnitudes + conformal)
    d_tr = pd.to_datetime(pd.Series(dates_tr)); uniq = d_tr.drop_duplicates().sort_values()
    n_cal = max(int(len(uniq) * cfg.calib_frac), 30)
    cutoff = uniq.iloc[-n_cal] if len(uniq) > n_cal else uniq.iloc[-1]
    calib_mask = (d_tr >= cutoff).to_numpy(); model_mask = ~calib_mask
    X_tr_model, y_tr_model = X_tr[model_mask], y_tr[model_mask]
    X_tr_calib, y_tr_calib = X_tr[calib_mask], y_tr[calib_mask]
    fwd_tr_calib = fwd_tr[calib_mask]

    # Final classifier on full train; calibrated probs on forecast slice
    clf_final = build_model(model_name, best_params, cfg)
    clf_final.fit(X_tr, y_tr)
    p_raw = clf_final.predict_proba(X_fc)[:, 1]
    p_cal = apply_calibrator(iso, p_raw)

    # Magnitude head (two regressors on MODEL-SLICE)
    y_abs_model = np.abs(fwd_tr[model_mask])
    mr_mask_model = y_tr_model == 1; mo_mask_model = y_tr_model == 0

    def fit_reg(Xm, ym):
        if Xm.shape[0] < 20 or np.sum(np.isfinite(ym)) < 20:
            return None, np.nan
        reg = make_hgbr(cfg); reg.fit(Xm, ym); return reg, float(np.nanmedian(ym))

    reg_mr, med_mr = fit_reg(X_tr_model[mr_mask_model], y_abs_model[mr_mask_model])
    reg_mo, med_mo = fit_reg(X_tr_model[mo_mask_model], y_abs_model[mo_mask_model])

    def predict_or_const(model, X, const):
        if model is None:
            return np.full(X.shape[0], const if np.isfinite(const) else 0.0)
        try:
            return model.predict(X)
        except Exception:
            return np.full(X.shape[0], const if np.isfinite(const) else 0.0)

    pred_mr_abs = predict_or_const(reg_mr, X_fc, med_mr)
    pred_mo_abs = predict_or_const(reg_mo, X_fc, med_mo)

    # Expected signed move (mixture)
    exp_move = p_cal * (-mom_fc_sign) * pred_mr_abs + (1.0 - p_cal) * (mom_fc_sign) * pred_mo_abs

    # Conformal bands from CALIB residuals
    res_mr = np.abs(np.abs(fwd_tr_calib[y_tr_calib==1]) - predict_or_const(reg_mr, X_tr_calib[y_tr_calib==1], med_mr))
    res_mo = np.abs(np.abs(fwd_tr_calib[y_tr_calib==0]) - predict_or_const(reg_mo, X_tr_calib[y_tr_calib==0], med_mo))
    q_mr, q_mo = conformal_q(res_mr, cfg.conformal_alpha), conformal_q(res_mo, cfg.conformal_alpha)
    low_move  = p_cal * (-mom_fc_sign) * (pred_mr_abs - q_mr) + (1.0 - p_cal) * (mom_fc_sign) * (pred_mo_abs - q_mo)
    high_move = p_cal * (-mom_fc_sign) * (pred_mr_abs + q_mr) + (1.0 - p_cal) * (mom_fc_sign) * (pred_mo_abs + q_mo)

    # Optional: decisions in forecast (if user provides coverage/threshold)
    threshold_used = np.nan
    y_pred = np.full(p_cal.shape[0], ABSTAIN, dtype=int)
    abstained = np.ones(p_cal.shape[0], dtype=bool)

    # Define probability-based filter
    hi_prob = lo_prob = np.zeros_like(p_cal, dtype=bool)
    if cfg.threshold_fixed is not None:
        threshold_used = float(cfg.threshold_fixed)
        hi_prob = p_cal >= threshold_used
        lo_prob = p_cal <= (1.0 - threshold_used)
    elif cfg.min_coverage is not None and cfg.threshold_rule == "coverage":
        threshold_used, _ = choose_threshold_by_coverage(p_cal, cfg.threshold_grid, cfg.min_coverage)
        hi_prob = p_cal >= threshold_used
        lo_prob = p_cal <= (1.0 - threshold_used)

    # Define magnitude-based filter
    mag_filter = np.ones_like(p_cal, dtype=bool)
    if cfg.min_exp_move_abs is not None:
        mag_filter = np.abs(exp_move) >= cfg.min_exp_move_abs

    # Combine filters to make final decision
    hi = hi_prob & mag_filter
    lo = lo_prob & mag_filter

    if hi.any() or lo.any():
        y_pred[hi] = 1
        y_pred[lo] = 0
        abstained = ~(hi | lo)

    # Create the dictionary for the standard prediction output with consistent column names
    output_data = {
        "date": forecast_panel["Date"].values,
        "ticker": tickers_fc,
        "horizon": horizon_name,  # Use "horizon" instead of "Forecast Horizon"
        "raw_probability": p_raw,
        "calibrated_probability": p_cal,
        "confidence": np.maximum(p_cal, 1.0 - p_cal),
        "expected_move": exp_move,
        "prediction_interval_low": low_move,
        "prediction_interval_high": high_move,
        "momentum_sign": mom_fc_sign,
        "decision_threshold": threshold_used,
        "prediction": y_pred,
        "abstained": abstained,
    }
    
    # Extract the feature values from the forecast panel and add them to the output
    feature_values = forecast_panel[feat_cols].reset_index(drop=True)
    final_output_df = pd.concat([pd.DataFrame(output_data), feature_values], axis=1)

    # Create descriptions for the output columns
    descriptions = {
        "date": "The date for which the forecast is made.",
        "ticker": "The stock ticker symbol.",
        "horizon": "The time period for the forecast (e.g., d1 for 1 day).",
        "raw_probability": "The raw probability output from the model before calibration.",
        "calibrated_probability": "The model's probability output after calibration, representing the likelihood of an upward movement.",
        "confidence": "The model's confidence in its prediction, calculated as max(p, 1-p).",
        "expected_move": "The expected percentage change in price.",
        "prediction_interval_low": "The lower bound of the prediction interval for the price move.",
        "prediction_interval_high": "The upper bound of the prediction interval for the price move.",
        "momentum_sign": "The sign of the momentum feature, indicating recent price trends.",
        "decision_threshold": "The probability threshold used to make a buy (1) or sell (0) decision.",
        "prediction": "The final forecast: 1 for an expected upward move, 0 for a downward move.",
        "abstained": "Indicates if the model abstained from making a prediction (True/False).",
    }
    
    # Add descriptions for feature columns
    for col in feature_values.columns:
        descriptions[col] = f"Value of the '{col}' feature used in the model."

    return final_output_df, descriptions


def run_forecast(cfg: Config, model_name: str) -> None:
    """Train on most-recent window and emit next-day forecasts for all horizons."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir)

    np.random.seed(cfg.random_state)
    logger.info("--- Running in FORECAST mode ---")

    logger.info("Resolving input Excel + sheet…")
    excel_path, sheet_name = autodiscover_excel_and_sheet(cfg.excel_path, cfg.sheet_name, logger)
    cfg.excel_path, cfg.sheet_name = excel_path, sheet_name

    logger.info("Loading returns (wide)...")
    returns_wide = load_returns_wide(cfg.excel_path, cfg.sheet_name)
    returns_wide = returns_wide.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

    # Optional market context
    market_feats = None
    if cfg.with_market:
        logger.info("Loading market context…")
        mkt_df = load_market_from_excel(cfg.excel_path, sheet_name=cfg.market_sheet, logger=logger)
        if mkt_df is None:
            logger.info(f"No usable '{cfg.market_sheet}' sheet. Fetching S&P 500 via {cfg.market_source}…")
            mkt_df = get_sp500_market(cfg.market_source, start=cfg.market_start, logger=logger)
        if mkt_df is not None and not mkt_df.empty:
            market_feats = compute_market_features(mkt_df, cfg)
        else:
            logger.warning("Proceeding without market features (fetch or sheet failed).")

    logger.info("Building feature panel…")
    panel = build_feature_panel(returns_wide, cfg, logger)
    if market_feats is not None:
        panel = panel.merge(market_feats, on="Date", how="left")

    # Most recent feature date & training window
    last_feature_date = panel["Date"].max()
    logger.info(f"Most recent feature date: {last_feature_date.date()}")
    train_start_date = last_feature_date - pd.DateOffset(years=cfg.train_years)
    logger.info(f"Training window: {train_start_date.date()} to {last_feature_date.date()}")

    all_fc = []
    for name, h_days, _ in cfg.horizons:
        logger.info(f"[Forecast] Horizon {name}")
        # Avoid leakage: ensure targets would exist beyond forecast date by cutting train at (last - h_days)
        train_cutoff = last_feature_date - BDay(h_days)
        train_panel = panel[(panel["Date"] >= train_start_date) & (panel["Date"] <= train_cutoff)].copy()
        forecast_panel = panel[panel["Date"] == last_feature_date].copy()
        if train_panel.empty or forecast_panel.empty:
            logger.warning(f"[Forecast] Skipping {name}: insufficient rows for train/forecast.")
            continue
            
        # Properly unpack the tuple return
        df_fc, descriptions = forecast_one_horizon(train_panel, forecast_panel, name, cfg, model_name, logger)
        
        # Only append if we got a valid DataFrame
        if isinstance(df_fc, pd.DataFrame) and not df_fc.empty:
            # Ensure required columns exist
            if "horizon" not in df_fc.columns:
                df_fc["horizon"] = name
            if "ticker" not in df_fc.columns and "Ticker" in df_fc.columns:
                df_fc = df_fc.rename(columns={"Ticker": "ticker"})
            all_fc.append(df_fc)

    if not all_fc:
        logger.error("No forecasts produced for any horizon.")
        return

    # Verify all DataFrames have required columns before concatenation
    required_cols = {"horizon", "ticker"}
    bad_frames = []
    for i, df in enumerate(all_fc):
        if not isinstance(df, pd.DataFrame):
            bad_frames.append(f"Index {i}: not a DataFrame")
        elif not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            bad_frames.append(f"Index {i}: missing columns {missing}")
    
    if bad_frames:
        logger.error(f"Invalid forecast frames: {bad_frames}")
        return

    final_fc = pd.concat(all_fc, ignore_index=True).sort_values(["horizon", "ticker"])

    base_csv_path = out_dir / f"forecast_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
    out_csv = get_unique_filepath(base_csv_path)
    
    final_fc.to_csv(out_csv, index=False, float_format="%.6f")
    logger.info(f"Saved forecast file: {out_csv.name} ({len(final_fc):,} rows)")

    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "excel": str(cfg.excel_path),
        "sheet": cfg.sheet_name,
        "model": model_name,
        "mode": "forecast",
        "last_feature_date": str(last_feature_date.date()),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
        "artifacts": {"forecast_csv": out_csv.name},
    }
    meta_path = get_unique_filepath(out_dir / "run_meta_forecast.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info("Forecast complete. Artifacts: forecast_YYYYMMDD.csv, run_meta_forecast.json")


# =========================
# Entry
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mean-Reversion production: per-stock probabilities & magnitudes (no GUI)")
    # Mode
    p.add_argument("--mode", default=None, choices=["backtest","forecast"], help="Run mode: backtest or forecast (default: backtest)")
    # Excel can be omitted; we'll autodiscover a workbook and a plausible sheet
    p.add_argument("--excel", dest="excel", default=None, help="Path to Excel file (default: autodiscover MXUS_Data_*.xlsx/MXUS_Data.xlsx or any .xlsx)")
    p.add_argument("--sheet", dest="sheet", default=None, help="Sheet name for returns (default: 'Returns' or first sheet containing 'return')")
    p.add_argument("--out", dest="out", default="results", help="Output directory (default: results)")
    p.add_argument("--model", dest="model", default="logit", choices=["logit","xgb","lgb"], help="Classifier (default: logit)")
    p.add_argument("--train_years", type=int, default=3, help="Outer train years (default: 3)")
    p.add_argument("--val_years", type=int, default=1, help="Outer validation years (default: 1)")
    p.add_argument("--embargo", type=int, default=5, help="Embargo days after validation start (default: 5)")
    p.add_argument("--random_state", type=int, default=42, help="Random seed (default:  42)")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for param scoring (default: -1)")
    p.add_argument("--min_valid_frac", type=float, default=0.70, help="Minimum non-NaN fraction per ticker (default: 0.70)")
    # Market options
    p.add_argument("--no_market", dest="with_market", action="store_false", help="Disable market feature merge (default: on)")
    p.add_argument("--market_sheet", default="Market", help="Excel sheet name for market data if present (default: Market)")
    p.add_argument("--market_source", default="yahoo_price", choices=["yahoo_price","yahoo_tr","stooq","fred"], help="Fallback fetch source for S&P 500 (default: yahoo_price)")
    p.add_argument("--market_start", default="1990-01-01", help="Start date for fetched market series (default: 1990-01-01)")
    # Coverage / threshold options (apply to backtest decision reporting and optional forecast decisions)
    p.add_argument("--min_coverage", type=float, default=None, help="Minimum decision coverage (0-1). Example: 0.4 ⇒ ≥40% labeled; others abstain. Default: None")
    p.add_argument("--threshold_fixed", type=float, default=None, help="Use a fixed τ (e.g., 0.75). Overrides coverage rule if set.")
    p.add_argument("--min_exp_move", dest="min_exp_move_abs", type=float, default=None, help="Minimum absolute expected move to consider a signal (e.g., 0.01 for 1%). Default: None")
    p.add_argument("--threshold_rule", default="coverage", choices=["coverage","none"], help="Threshold selection rule when --min_coverage is set. Default: coverage")
    p.set_defaults(with_market=True)
    return p.parse_args()


def main():
    args = parse_args()
    # Create a dictionary of arguments to override Config defaults
    # This ensures that only arguments explicitly provided on the command line
    # will overwrite the defaults set in the Config class.
    override_args = {
        "mode": args.mode,
        "excel_path": args.excel,
        "sheet_name": args.sheet,
        "output_dir": args.out,
        "with_market": args.with_market,
        "market_sheet": args.market_sheet,
        "market_source": args.market_source,
        "market_start": args.market_start,
        "train_years": args.train_years,
        "validation_years": args.val_years,
        "embargo_days": args.embargo,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "min_valid_frac": args.min_valid_frac,
        "min_coverage": args.min_coverage,
        "threshold_fixed": args.threshold_fixed,
        "threshold_rule": args.threshold_rule,
        "min_exp_move_abs": args.min_exp_move_abs,
    }
    # Filter out any arguments that were not provided (i.e., are None)
    # so they don't overwrite the Config defaults.
    final_args = {k: v for k, v in override_args.items() if v is not None}
    
    cfg = Config(**final_args)
    if cfg.mode == "forecast":
        run_forecast(cfg, model_name=args.model)
    else:
        run_pipeline(cfg, model_name=args.model)


if __name__ == "__main__":
    main()
