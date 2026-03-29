"""
walk_forward_validation.py — Out-of-Sample Research Infrastructure
====================================================================
Implements rolling walk-forward validation for systematic strategies.

Walk-forward validation is the minimum standard for evaluating whether
a systematic strategy has genuine predictive power vs in-sample overfit.

The key insight: in-sample Sharpe is nearly meaningless. A strategy
can always be tuned to look good on historical data it was optimized on.
Out-of-sample Sharpe on data the strategy never "saw" is the number
that actually matters.

ProjectV uses this framework to produce honest OOS performance estimates
before any live deployment decision.

Architecture:
    ┌─────────────────────────────────────────────┐
    │  Full backtest period (e.g. 2005 - 2025)    │
    └─────────────────────────────────────────────┘
         │
         ▼
    ┌──────────────┬──────────────┐   Window 1
    │  Train       │   Test (OOS) │
    │  (8 years)   │   (4 years)  │
    └──────────────┴──────────────┘
              ┌──────────────┬──────────────┐   Window 2 (step 2yr)
              │  Train       │   Test (OOS) │
              │  (8 years)   │   (4 years)  │
              └──────────────┴──────────────┘
                        ... and so on
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class WindowResult:
    """Results from a single walk-forward window."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    cagr: float
    sharpe: float
    ann_vol: float
    max_dd: float
    total_return: float


def build_walk_forward_windows(
    start: str,
    end: str,
    train_years: int = 8,
    test_years: int = 4,
    step_years: int = 2,
) -> List[Tuple[str, str, str, str]]:
    """
    Generate rolling train/test window pairs for walk-forward validation.

    Each window has:
    - A training period (used for parameter fitting if applicable)
    - A test period (OOS evaluation — strategy runs without seeing future data)

    Windows advance by step_years to produce multiple non-overlapping
    test periods across the full history.

    Args:
        start: Backtest start date (YYYY-MM-DD)
        end: Backtest end date (YYYY-MM-DD)
        train_years: Length of training window in years
        test_years: Length of OOS test window in years
        step_years: How far to advance each window

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples

    Example with 2005-2025, train=8, test=4, step=2:
        Window 1: Train 2005-2013, Test 2013-2017
        Window 2: Train 2007-2015, Test 2015-2019
        Window 3: Train 2009-2017, Test 2017-2021
        Window 4: Train 2011-2019, Test 2019-2023
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)

    windows = []
    t0 = s

    while True:
        train_start = t0
        train_end = t0 + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.DateOffset(days=1)

        if test_end > e:
            break

        windows.append((
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))

        t0 = t0 + pd.DateOffset(years=step_years)

    return windows


def metrics_from_returns(returns: pd.Series) -> dict:
    """
    Compute performance metrics from a daily returns series.

    Args:
        returns: Daily return series (e.g. 0.01 = 1% daily return)

    Returns:
        Dictionary with CAGR, Sharpe, AnnVol, MaxDD, TotalReturn
    """
    returns = returns.dropna()
    if returns.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "AnnVol": 0.0,
                "MaxDD": 0.0, "TotalReturn": 0.0}

    ann = 252.0
    equity = (1.0 + returns).cumprod()

    years = (returns.index[-1] - returns.index[0]).days / 365.25
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0

    vol = float(returns.std(ddof=0) * np.sqrt(ann))
    sharpe = float((returns.mean() * ann) / vol) if vol > 1e-12 else 0.0

    peak = equity.cummax()
    max_dd = float(((equity / peak) - 1.0).min())

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "AnnVol": vol,
        "MaxDD": max_dd,
        "TotalReturn": float(equity.iloc[-1] - 1.0),
    }


def summarize_walk_forward_results(results: List[WindowResult]) -> dict:
    """
    Aggregate walk-forward OOS results across all windows.

    Args:
        results: List of WindowResult objects from each window

    Returns:
        Summary statistics across all OOS windows

    Key question this answers:
        Is the strategy consistently profitable across different
        time periods, or did it just get lucky in one era?

    A robust strategy shows:
        - Positive OOS Sharpe in most windows
        - Relatively stable Sharpe across windows (low variance)
        - OOS Sharpe reasonably close to in-sample Sharpe
          (large gap signals overfit)
    """
    sharpes = [r.sharpe for r in results]
    cagrs = [r.cagr for r in results]
    max_dds = [r.max_dd for r in results]

    return {
        "n_windows": len(results),
        "pct_positive_sharpe": float(np.mean([s > 0 for s in sharpes])),
        "mean_oos_sharpe": float(np.mean(sharpes)),
        "median_oos_sharpe": float(np.median(sharpes)),
        "std_oos_sharpe": float(np.std(sharpes)),
        "min_oos_sharpe": float(np.min(sharpes)),
        "max_oos_sharpe": float(np.max(sharpes)),
        "mean_oos_cagr": float(np.mean(cagrs)),
        "worst_oos_drawdown": float(np.min(max_dds)),
    }
