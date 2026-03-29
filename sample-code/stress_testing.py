"""
stress_testing.py — Parameter Sensitivity and Stress Analysis
==============================================================
Tests strategy robustness by systematically varying parameters
and evaluating performance under stressed conditions.

The core principle: a genuine edge should be robust to small
parameter perturbations. If Sharpe collapses when you shift a
threshold by 0.02, the strategy is overfit to that specific value,
not genuinely predictive.

ProjectV's stress testing framework runs three categories of tests:

1. Transaction Cost Sensitivity
   How does performance degrade as costs increase?
   If the strategy only works at exactly 2 bps, it's not deployable.

2. Parameter Sensitivity  
   Does performance hold across reasonable parameter ranges?
   Sharp cliffs in the Sharpe surface indicate overfit.

3. Crisis Period Analysis
   How did the strategy behave during:
   - GFC (2008-03 to 2009-03)
   - European Debt Crisis (2011)
   - COVID Crash (2020-02 to 2020-03)
   - Rate Shock (2022)
   - SVB Crisis (2023-03)

Key insight from building ProjectV:
   Running stress tests BEFORE declaring a strategy complete is
   essential. ProjectV's metrics looked strong in-sample but
   stress testing revealed the regime classifier was never
   properly wired into the allocation logic — the system was
   effectively running regime-blind the entire time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd


# Crisis periods for dedicated stress analysis
CRISIS_PERIODS = {
    "GFC": ("2008-03-01", "2009-03-31"),
    "Euro_Crisis": ("2011-07-01", "2011-12-31"),
    "COVID_Crash": ("2020-02-19", "2020-03-23"),
    "Rate_Shock": ("2022-01-01", "2022-12-31"),
    "SVB_Crisis": ("2023-03-01", "2023-03-31"),
}


@dataclass
class StressResult:
    """Result from a single stress test case."""
    case_name: str
    cagr: float
    sharpe: float
    ann_vol: float
    max_dd: float
    vs_baseline_sharpe: float  # Sharpe delta vs baseline


def analyze_parameter_sensitivity(
    sharpe_surface: Dict[str, float],
    baseline_key: str,
    cliff_threshold: float = 0.20,
) -> dict:
    """
    Analyze a Sharpe surface for parameter sensitivity.

    Args:
        sharpe_surface: Dict mapping parameter value -> Sharpe ratio
                        e.g. {'0.15': 0.82, '0.20': 0.86, '0.25': 0.71}
        baseline_key: Key of the baseline parameter value
        cliff_threshold: Sharpe drop that constitutes a "cliff" (overfit signal)

    Returns:
        Sensitivity analysis summary

    Example:
        A robust parameter shows gradual degradation:
            0.15 -> 0.79, 0.20 -> 0.86, 0.25 -> 0.81, 0.30 -> 0.76
        
        An overfit parameter shows cliffs:
            0.15 -> 0.31, 0.20 -> 0.86, 0.25 -> 0.28
        The second pattern means the strategy only works at exactly
        one value — it learned the data, not the signal.
    """
    baseline_sharpe = sharpe_surface[baseline_key]
    sharpes = list(sharpe_surface.values())

    max_drop = baseline_sharpe - min(sharpes)
    has_cliff = max_drop > cliff_threshold

    return {
        "baseline_sharpe": baseline_sharpe,
        "min_sharpe_across_range": min(sharpes),
        "max_sharpe_across_range": max(sharpes),
        "max_sharpe_drop": max_drop,
        "has_cliff": has_cliff,
        "robust": not has_cliff,
        "interpretation": (
            "ROBUST: Sharpe stable across parameter range"
            if not has_cliff else
            f"WARNING: Sharpe drops {max_drop:.2f} — possible overfit to specific value"
        ),
    }


def analyze_crisis_performance(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> Dict[str, dict]:
    """
    Analyze strategy performance during known crisis periods.

    Args:
        returns: Daily strategy returns
        benchmark_returns: Daily benchmark (e.g. SPY) returns for comparison

    Returns:
        Dict mapping crisis name -> performance metrics during that period

    Key questions:
        - Did the strategy go defensive before or after the peak?
        - How does max drawdown compare to benchmark?
        - How quickly did it recover vs benchmark?
    """
    results = {}

    for crisis_name, (start, end) in CRISIS_PERIODS.items():
        crisis_rets = returns.loc[start:end].dropna()

        if crisis_rets.empty:
            continue

        equity = (1.0 + crisis_rets).cumprod()
        total_return = float(equity.iloc[-1] - 1.0)
        peak = equity.cummax()
        max_dd = float(((equity / peak) - 1.0).min())
        vol = float(crisis_rets.std(ddof=0) * np.sqrt(252))

        result = {
            "total_return": total_return,
            "max_drawdown": max_dd,
            "annualized_vol": vol,
            "n_trading_days": len(crisis_rets),
        }

        if benchmark_returns is not None:
            bench_rets = benchmark_returns.loc[start:end].dropna()
            if not bench_rets.empty:
                bench_equity = (1.0 + bench_rets).cumprod()
                bench_total = float(bench_equity.iloc[-1] - 1.0)
                bench_peak = bench_equity.cummax()
                bench_dd = float(((bench_equity / bench_peak) - 1.0).min())

                result["benchmark_total_return"] = bench_total
                result["benchmark_max_drawdown"] = bench_dd
                result["outperformance"] = total_return - bench_total
                result["drawdown_reduction"] = bench_dd - max_dd

        results[crisis_name] = result

    return results


def cost_sensitivity_analysis(
    baseline_sharpe: float,
    baseline_cagr: float,
    avg_daily_turnover: float,
    cost_multiples: List[float] = [1.0, 2.0, 3.0, 5.0],
    base_cost_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Estimate strategy performance degradation as transaction costs increase.

    This is a simplified analytical model — actual results require
    running the full backtest at each cost level.

    Args:
        baseline_sharpe: In-sample Sharpe at base cost assumption
        baseline_cagr: In-sample CAGR at base cost assumption
        avg_daily_turnover: Average daily portfolio turnover fraction
        cost_multiples: Cost multipliers to test
        base_cost_bps: Baseline cost assumption in basis points per side

    Returns:
        DataFrame showing estimated metrics at each cost level

    Note:
        A strategy that only survives at exactly 2 bps is not
        deployable in practice. Real costs vary with market impact,
        spread, and execution quality. Aim for robustness up to 5-10x
        the base cost assumption.
    """
    rows = []
    annual_turnover = avg_daily_turnover * 252

    for mult in cost_multiples:
        cost_bps = base_cost_bps * mult
        # Annual drag = turnover * cost (2-way)
        annual_cost_drag = annual_turnover * (cost_bps / 10000) * 2
        estimated_cagr = baseline_cagr - annual_cost_drag
        # Rough Sharpe adjustment (assumes vol unchanged)
        estimated_sharpe = baseline_sharpe - (annual_cost_drag / (baseline_cagr / baseline_sharpe + 1e-8))

        rows.append({
            "cost_multiple": mult,
            "cost_bps_per_side": cost_bps,
            "annual_cost_drag": annual_cost_drag,
            "estimated_cagr": max(estimated_cagr, -1.0),
            "estimated_sharpe": estimated_sharpe,
            "still_profitable": estimated_cagr > 0,
        })

    return pd.DataFrame(rows)
