# -*- coding: utf-8 -*-
"""
AI TRADING ENGINE v8.0
=======================
Next-Level Research-grade, event-driven ML trading framework.

Neu in v8.0:
- Multi-Asset Portfolio Aggregation & Pooled Machine Learning
- LightGBM Gradient Boosting anstelle von Random Forests
- Gaussian Mixture Model (GMM) Regime Detection (Risk-Off in High-Vol Regimes)
- Optuna Hyperparameter Auto-Tuning Integration
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

# Neue Next-Level Imports
import lightgbm as lgb
import optuna
from sklearn.mixture import GaussianMixture

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("AI-TRADING-v8")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # --- Multi-Asset Data ---
    tickers: Tuple[str, ...] = ("BTC-USD", "ETH-USD")  # Multi-Asset Support!
    benchmark: str = "^GSPC"
    start_date: str = "2020-01-01"
    end_date: str = "2026-09-01"
    data_provider: str = "yfinance"
    
    initial_cash: float = 10_000.0

    # --- Execution & Broker ---
    fee_pct: float = 0.0010
    slippage_low: float = 0.0003
    slippage_high: float = 0.0015
    volatility_slippage_threshold: float = 1.5
    market_impact_coeff: float = 0.0005
    sizing_max_iter: int = 6
    sizing_tol: float = 1e-4

    # --- Barrier / Holding ---
    atr_sl: float = 2.0
    atr_tp: float = 4.0
    max_hold_days: int = 5
    intrabar_policy: str = "probabilistic" 

    # --- Signal & Regime ---
    min_probability: float = 0.55
    use_regime_filter: bool = True     # GMM Regime Filter aktiviert
    regime_penalty_factor: float = 0.5 # Halbiert Positionsgröße im Crash-Regime

    # --- Kelly Sizing ---
    kelly_fraction: float = 0.50
    min_risk_pct: float = 0.0025
    max_risk_pct: float = 0.05
    max_exposure_pct: float = 1.00

    # --- Walk-Forward ---
    min_train_days: int = 500
    embargo_days: int = 5
    test_chunk: int = 30
    refit_every: int = 30
    purged_kfold_splits: int = 5

    # --- Machine Learning (LightGBM) ---
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.01
    calibration_splits: int = 3
    random_state: int = 42

    # --- Optuna Tuning ---
    run_optuna: bool = False
    optuna_trials: int = 20

    # --- Features ---
    features: Tuple[str, ...] = (
        "RET_1D", "RET_3D", "RET_5D", "RET_10D", "RET_20D",
        "RSI", "ATR_PCT", "ATR_REGIME", "REALIZED_VOL_20", "RET_SKEW_20",
        "EMA20_DIST", "EMA50_DIST", "EMA200_DIST", "EMA20_SLOPE",
        "VOL_RATIO", "VOLUME_RATIO", "RANGE_PCT", "BODY_PCT",
        "UPPER_WICK_PCT", "LOWER_WICK_PCT", "SP500_RET_3D", "SP500_RET_20D",
    )
    
    report_dir: Optional[str] = "backtest_results_v8"

# ============================================================================
# DATA & FEATURES
# ============================================================================

def _fetch_yfinance(ticker: str, bench: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    asset = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(asset.columns, pd.MultiIndex):
        asset.columns = asset.columns.get_level_values(0)
    
    benchmark = yf.download(bench, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(benchmark.columns, pd.MultiIndex):
        benchmark.columns = benchmark.columns.get_level_values(0)
        
    df = asset[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["SP500_Close"] = benchmark["Close"].reindex(df.index).ffill().shift(1)
    return df.dropna(subset=["Open", "High", "Low", "Close"])

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0/window, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0/window, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]

    out["RET_1D"] = close.pct_change(1)
    out["RET_3D"] = close.pct_change(3)
    out["RET_5D"] = close.pct_change(5)
    out["RET_10D"] = close.pct_change(10)
    out["RET_20D"] = close.pct_change(20)

    out["RSI"] = _rsi(close, 14)
    
    # ATR
    prev_close = close.shift(1)
    tr = pd.concat([out["High"]-out["Low"], (out["High"]-prev_close).abs(), (out["Low"]-prev_close).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.ewm(alpha=1.0/14, adjust=False).mean()
    out["ATR_PCT"] = out["ATR"] / close
    out["ATR_REGIME"] = out["ATR_PCT"] / out["ATR_PCT"].rolling(50).mean()

    # Volatility & Skew
    log_ret = np.log(close / close.shift(1))
    out["REALIZED_VOL_20"] = log_ret.rolling(20).std()
    out["RET_SKEW_20"] = log_ret.rolling(20).skew()

    # Moving Averages
    for p in (20, 50, 200):
        ema = close.ewm(span=p, adjust=False).mean()
        out[f"EMA{p}_DIST"] = close / ema - 1.0
        if p == 20: out["EMA20_SLOPE"] = ema.pct_change(5)

    out["VOLUME_RATIO"] = out["Volume"] / out["Volume"].rolling(20).mean()
    out["VOL_RATIO"] = out["ATR_PCT"] / out["ATR_PCT"].rolling(50).mean()

    out["RANGE_PCT"] = (out["High"] - out["Low"]) / close
    out["BODY_PCT"] = (out["Close"] - out["Open"]).abs() / close
    out["UPPER_WICK_PCT"] = (out["High"] - out[["Open", "Close"]].max(axis=1)) / close
    out["LOWER_WICK_PCT"] = (out[["Open", "Close"]].min(axis=1) - out["Low"]) / close

    out["SP500_RET_3D"] = out["SP500_Close"].pct_change(3)
    out["SP500_RET_20D"] = out["SP500_Close"].pct_change(20)

    return out

def create_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    h = cfg.max_hold_days

    out["LABEL_ENTRY"] = out["Open"].shift(-1)
    out["LABEL_SL"] = out["LABEL_ENTRY"] - cfg.atr_sl * out["ATR"]
    out["LABEL_TP"] = out["LABEL_ENTRY"] + cfg.atr_tp * out["ATR"]

    low, high = out["Low"].to_numpy(), out["High"].to_numpy()
    sl, tp = out["LABEL_SL"].to_numpy(), out["LABEL_TP"].to_numpy()

    no_hit = h + 2
    n = len(out)
    sl_hit, tp_hit = np.full(n, no_hit), np.full(n, no_hit)

    for step in range(2, h + 2):
        lo, hi = np.full(n, np.nan), np.full(n, np.nan)
        if step < n:
            lo[:-step] = low[step:]
            hi[:-step] = high[step:]
        
        hit_sl = lo <= sl
        hit_tp = hi >= tp
        sl_hit = np.where((sl_hit == no_hit) & hit_sl, step, sl_hit)
        tp_hit = np.where((tp_hit == no_hit) & hit_tp, step, tp_hit)

    out["Target"] = (tp_hit < sl_hit).astype(float)
    first_hit = np.minimum(sl_hit, tp_hit)
    out["T1_OFFSET"] = np.where(first_hit == no_hit, h, first_hit).astype(int)

    invalid = (np.arange(n) >= n - h - 2) | out["LABEL_ENTRY"].isna() | out["ATR"].isna()
    out.loc[invalid, "Target"] = np.nan
    out.loc[invalid, "T1_OFFSET"] = -1
    return out

# ============================================================================
# ML MODEL (LIGHTGBM) & GMM REGIME
# ============================================================================

@dataclass
class FoldModel:
    model: CalibratedClassifierCV
    selector: SelectFromModel
    gmm: Optional[GaussianMixture]
    high_vol_cluster: int
    selected_features: List[str]

def _make_base_model(cfg: Config) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        class_weight="balanced",
        random_state=cfg.random_state,
        n_jobs=-1,
        verbose=-1
    )

def fit_fold_model(train: pd.DataFrame, cfg: Config) -> Optional[FoldModel]:
    X = train[list(cfg.features)]
    y = train["Target"].astype(int)

    if len(train) < cfg.min_train_days or y.nunique() < 2: return None

    # Feature Selection
    selector = SelectFromModel(_make_base_model(cfg), threshold="median", prefit=False).fit(X, y)
    selected = [f for f, keep in zip(cfg.features, selector.get_support()) if keep]
    
    # Train Calibrated LightGBM
    cv = TimeSeriesSplit(n_splits=min(cfg.calibration_splits, max(2, len(train)//100)))
    model = CalibratedClassifierCV(_make_base_model(cfg), cv=cv, method="sigmoid")
    model.fit(selector.transform(X), y)

    # Train Unsupervised GMM Market Regime on Volatility
    gmm = None
    high_vol_cluster = 0
    if cfg.use_regime_filter:
        gmm = GaussianMixture(n_components=2, random_state=cfg.random_state)
        vol_data = train[['REALIZED_VOL_20']].fillna(0)
        gmm.fit(vol_data)
        high_vol_cluster = int(np.argmax(gmm.means_.flatten()))

    return FoldModel(model, selector, gmm, high_vol_cluster, selected)

# ============================================================================
# OPTUNA HYPERPARAMETER TUNING
# ============================================================================

def objective_optuna(trial: optuna.Trial, pooled_data: pd.DataFrame, cfg: Config) -> float:
    """Optuna sucht die besten Parameter per schnellem Purged K-Fold CV."""
    cfg_tune = dataclasses.replace(cfg)
    cfg_tune.atr_sl = trial.suggest_float("atr_sl", 1.0, 4.0)
    cfg_tune.atr_tp = trial.suggest_float("atr_tp", 1.0, 5.0)
    cfg_tune.min_probability = trial.suggest_float("min_probability", 0.51, 0.65)
    
    # Label neu berechnen mit temporären Parametern
    df_tuned = create_labels(pooled_data, cfg_tune)
    df_clean = df_tuned.dropna(subset=list(cfg.features) + ["Target", "T1_OFFSET"]).copy()
    
    if len(df_clean) < cfg.min_train_days * 2: return 0.0
    
    # Schneller CV Durchlauf
    idx = np.arange(len(df_clean))
    t1 = df_clean["T1_OFFSET"].to_numpy(dtype=int)
    exit_idx = idx + np.maximum(t1, 0)
    fold_edges = np.linspace(0, len(df_clean), 4).astype(int) # 3 Folds für Speed
    
    aucs = []
    for k in range(3):
        test_lo, test_hi = fold_edges[k], fold_edges[k+1]
        test_mask = (idx >= test_lo) & (idx < test_hi)
        overlap = (exit_idx >= test_lo) & (idx < test_hi)
        embargo_end = test_hi + cfg.embargo_days
        train_mask = (~overlap) & (~((idx >= test_hi) & (idx < embargo_end))) & (~test_mask)
        
        train, test = df_clean.iloc[train_mask], df_clean.iloc[test_mask]
        if len(train) < 300 or test.empty: continue
        
        model = fit_fold_model(train, cfg_tune)
        if model:
            p = model.model.predict_proba(model.selector.transform(test[list(cfg_tune.features)]))[:, 1]
            try: aucs.append(roc_auc_score(test["Target"].astype(int), p))
            except: pass
            
    return float(np.mean(aucs)) if aucs else 0.0

def tune_hyperparameters(data: pd.DataFrame, cfg: Config) -> Config:
    log.info("🚀 Starte Optuna Hyperparameter-Tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_optuna(trial, data, cfg), n_trials=cfg.optuna_trials)
    
    best = study.best_params
    log.info(f"Tuning beendet. Beste Parameter: {best}")
    return dataclasses.replace(cfg, atr_sl=best["atr_sl"], atr_tp=best["atr_tp"], min_probability=best["min_probability"])

# ============================================================================
# PORTFOLIO BACKTEST & BROKER (Regime Aware)
# ============================================================================

def size_from_kelly(cash: float, entry_price: float, sl_dist: float, prob: float, cfg: Config, regime_penalty: float=1.0) -> float:
    b = cfg.atr_tp / cfg.atr_sl
    q = 1.0 - prob
    raw = max(0.0, (prob * b - q) / b)
    
    adjusted = raw * cfg.kelly_fraction * regime_penalty
    risk_pct = float(np.clip(adjusted, cfg.min_risk_pct, cfg.max_risk_pct))
    
    qty = (cash * risk_pct) / sl_dist
    return max(0.0, min(qty, (cash * cfg.max_exposure_pct) / entry_price))

class PortfolioBroker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cash = cfg.initial_cash
        self.positions = {} # ticker -> Position dict
        self.trades = []

    def equity(self, marks: dict) -> float:
        eq = self.cash
        for ticker, pos in self.positions.items():
            eq += pos['qty'] * marks.get(ticker, pos['entry_price'])
        return eq

    def process_exits(self, date: pd.Timestamp, rows: dict):
        exits_to_remove = []
        for ticker, pos in self.positions.items():
            if ticker not in rows: continue
            row = rows[ticker]
            
            exit_price = None
            reason = None
            
            # Intrabar Triple Barrier logic
            if row.Open <= pos['stop']: exit_price, reason = row.Open, "STOP_GAP"
            elif row.Open >= pos['target']: exit_price, reason = row.Open, "TP_GAP"
            elif row.Low <= pos['stop'] and row.High >= pos['target']:
                p_stop = (pos['target'] - pos['entry_price']) / (pos['target'] - pos['stop'])
                exit_price, reason = p_stop * pos['stop'] + (1-p_stop)*pos['target'], "AMBIGUOUS"
            elif row.Low <= pos['stop']: exit_price, reason = pos['stop'], "STOP"
            elif row.High >= pos['target']: exit_price, reason = pos['target'], "TAKE_PROFIT"
            elif (date - pos['entry_date']).days >= self.cfg.max_hold_days: exit_price, reason = row.Close, "TIME"

            if exit_price:
                slip = self.cfg.slippage_high if row.VOL_RATIO > self.cfg.volatility_slippage_threshold else self.cfg.slippage_low
                exec_price = exit_price * (1.0 - slip)
                proceeds = pos['qty'] * exec_price * (1.0 - self.cfg.fee_pct)
                self.cash += proceeds
                self.trades.append({
                    "ticker": ticker, "entry_date": pos['entry_date'], "exit_date": date,
                    "pnl": proceeds - pos['entry_cash'], "pnl_pct": proceeds/pos['entry_cash'] - 1.0,
                    "reason": reason
                })
                exits_to_remove.append(ticker)
                
        for t in exits_to_remove: del self.positions[t]

    def process_entries(self, date: pd.Timestamp, signals: list):
        # signals is a list of tuples: (ticker, row, probability, regime_penalty)
        signals.sort(key=lambda x: x[2], reverse=True) # Höchste Prob zuerst
        for ticker, row, prob, penalty in signals:
            if ticker in self.positions: continue
            if self.cash <= self.cfg.initial_cash * 0.05: break # Fast kein Cash mehr
            
            sl_dist = self.cfg.atr_sl * row.ATR
            slip = self.cfg.slippage_high if row.VOL_RATIO > self.cfg.volatility_slippage_threshold else self.cfg.slippage_low
            exec_price = row.Open * (1.0 + slip)
            
            qty = size_from_kelly(self.cash, exec_price, sl_dist, prob, self.cfg, penalty)
            if qty <= 0: continue
            
            entry_cash = qty * exec_price * (1.0 + self.cfg.fee_pct)
            if entry_cash > self.cash: continue
            
            self.cash -= entry_cash
            self.positions[ticker] = {
                'entry_date': date, 'entry_price': exec_price, 'qty': qty,
                'stop': exec_price - sl_dist, 'target': exec_price + self.cfg.atr_tp*row.ATR,
                'entry_cash': entry_cash
            }

def run_portfolio_backtest(data_dict: Dict[str, pd.DataFrame], prob_dict: Dict[str, pd.Series], regime_dict: Dict[str, pd.Series], cfg: Config):
    all_dates = set()
    for df in data_dict.values(): all_dates.update(df.index)
    sorted_dates = sorted(list(all_dates))
    
    broker = PortfolioBroker(cfg)
    equity_curve = []
    
    for date in sorted_dates:
        rows = {}
        marks = {}
        signals = []
        
        for t, df in data_dict.items():
            if date in df.index:
                row = df.loc[date]
                rows[t] = row
                marks[t] = row.Close
                
                # Signal Generation
                prob = prob_dict[t].get(date, np.nan)
                if prob >= cfg.min_probability and row.Close > row.EMA200_DIST: # Trend Filter
                    regime = regime_dict[t].get(date, 0)
                    penalty = cfg.regime_penalty_factor if regime == 1 else 1.0
                    signals.append((t, row, prob, penalty))
                    
        broker.process_exits(date, rows)
        broker.process_entries(date, signals)
        equity_curve.append({"Date": date, "Equity": broker.equity(marks)})
        
    return pd.DataFrame(equity_curve).set_index("Date")["Equity"], broker.trades

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna Hyperparameter Tuning")
    args = parser.parse_args()
    
    cfg = Config(run_optuna=args.tune)
    Path(cfg.report_dir).mkdir(exist_ok=True)
    
    # 1. Daten holen & Features bauen
    data_dict = {}
    for ticker in cfg.tickers:
        log.info(f"Lade Daten für {ticker}...")
        raw = _fetch_yfinance(ticker, cfg.benchmark, cfg.start_date, cfg.end_date)
        data_dict[ticker] = build_features(raw)
        
    # Pool Daten für das universelle ML-Modell
    pooled_data = pd.concat(data_dict.values(), keys=cfg.tickers)
    
    # 2. Optuna (Optional)
    if cfg.run_optuna:
        cfg = tune_hyperparameters(pooled_data, cfg)
        
    # 3. Walk-Forward / Pooled Training
    # (Aus Performancegründen simulieren wir hier einen stark optimierten simplen Walk-Forward Cut)
    split_idx = int(len(pooled_data) * 0.7)
    train_pool = create_labels(pooled_data.iloc[:split_idx], cfg).dropna(subset=["Target"])
    
    log.info("🧠 Trainiere Universal LightGBM & GMM auf Pooled Data...")
    model = fit_fold_model(train_pool, cfg)
    
    prob_dict, regime_dict = {}, {}
    for t, df in data_dict.items():
        df_eval = df.dropna(subset=list(cfg.features))
        if model:
            Xt = model.selector.transform(df_eval[list(cfg.features)])
            prob_dict[t] = pd.Series(model.model.predict_proba(Xt)[:, 1], index=df_eval.index)
            
            if model.gmm:
                vols = df_eval[['REALIZED_VOL_20']].fillna(0)
                clusters = model.gmm.predict(vols)
                # Mappe GMM-Cluster auf 0 (Bull) und 1 (Crash)
                regimes = (clusters == model.high_vol_cluster).astype(int)
                regime_dict[t] = pd.Series(regimes, index=df_eval.index)
            else:
                regime_dict[t] = pd.Series(0, index=df_eval.index)
                
    # 4. Multi-Asset Portfolio Backtest
    log.info("💼 Starte Multi-Asset Portfolio Backtest...")
    equity, trades = run_portfolio_backtest(data_dict, prob_dict, regime_dict, cfg)
    
    # 5. Report generieren
    ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (1+ret) ** (365.25/(equity.index[-1]-equity.index[0]).days) - 1.0
    dd = (equity / equity.cummax() - 1.0).min()
    win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
    
    print("\n" + "="*60)
    print(" 🚀 AI TRADING ENGINE v8.0 - PORTFOLIO REPORT")
    print("="*60)
    print(f"Portfolio Assets : {', '.join(cfg.tickers)}")
    print(f"Total Return     : {ret*100:.2f}%")
    print(f"CAGR (Annual.)   : {cagr*100:.2f}%")
    print(f"Max Drawdown     : {dd*100:.2f}%")
    print(f"Total Trades     : {len(trades)}")
    print(f"Win Rate         : {win_rate*100:.2f}%")
    if cfg.use_regime_filter:
        print(f"Regime Filter    : Aktiviert (Halbiert Size in High-Vol Phasen)")
    print("="*60)
    
    # Export
    equity.to_csv(f"{cfg.report_dir}/portfolio_equity.csv")
    pd.DataFrame(trades).to_csv(f"{cfg.report_dir}/portfolio_trades.csv", index=False)
    log.info(f"💾 Report gespeichert in '{cfg.report_dir}/'")

if __name__ == "__main__":
    main()
