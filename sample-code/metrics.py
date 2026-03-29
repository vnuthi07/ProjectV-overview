"""
metrics.py — Core Performance Metrics
======================================
Standard quantitative finance performance metrics used to
evaluate systematic trading strategies.

These functions form the foundation of ProjectV's reporting
and research infrastructure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum peak-to-trough decline in equity curve.

    Args:
        equity: Cumulative equity series (e.g. starting at 1.0)

    Returns:
        Maximum drawdown as a negative float (e.g. -0.15 = -15%)
    """
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compound Annual Growth Rate.

    Args:
        equity: Cumulative equity series
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized return as a float (e.g. 0.084 = 8.4%)
    """
    if equity.empty:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    n_periods = len(equity) - 1
    if n_periods <= 0:
        return 0.0
    years = n_periods / periods_per_year
    return float(total_return ** (1 / years) - 1)


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized volatility of daily returns.

    Args:
        returns: Daily return series
        periods_per_year: Trading periods per year

    Returns:
        Annualized volatility as a float (e.g. 0.10 = 10%)
    """
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Annualized Sharpe ratio.

    Args:
        returns: Daily return series
        rf: Annual risk-free rate (default 0.0)
        periods_per_year: Trading periods per year

    Returns:
        Sharpe ratio (e.g. 0.86 is strong for a systematic strategy)

    Note:
        A Sharpe of 0.5+ is generally considered acceptable for
        systematic strategies. Above 1.0 is strong. Above 2.0 is
        exceptional and warrants scrutiny for overfitting.
    """
    if returns.empty:
        return 0.0
    excess = returns - rf / periods_per_year
    vol = returns.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Annualized Sortino ratio — like Sharpe but penalizes
    only downside volatility.

    Args:
        returns: Daily return series
        rf: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sortino ratio
    """
    if returns.empty:
        return 0.0
    excess = returns - rf / periods_per_year
    downside = returns[returns < 0]
    downside_vol = downside.std(ddof=0) * np.sqrt(periods_per_year)
    if downside_vol == 0 or np.isnan(downside_vol):
        return 0.0
    return float(excess.mean() * periods_per_year / downside_vol)


def calmar(equity: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calmar ratio — CAGR divided by absolute max drawdown.
    Measures return per unit of worst-case risk.

    Args:
        equity: Cumulative equity series
        periods_per_year: Trading periods per year

    Returns:
        Calmar ratio (higher is better)
    """
    returns = equity.pct_change().dropna()
    c = cagr(equity, periods_per_year)
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(c / abs(mdd))


def full_metrics(equity: pd.Series, periods_per_year: int = 252) -> dict:
    """
    Compute full performance metric suite from equity curve.

    Args:
        equity: Cumulative equity series starting at 1.0
        periods_per_year: Trading periods per year

    Returns:
        Dictionary of all key performance metrics
    """
    returns = equity.pct_change().dropna()

    return {
        "CAGR": cagr(equity, periods_per_year),
        "Sharpe": sharpe(returns, periods_per_year=periods_per_year),
        "Sortino": sortino(returns, periods_per_year=periods_per_year),
        "Calmar": calmar(equity, periods_per_year),
        "AnnVol": annualized_vol(returns, periods_per_year),
        "MaxDD": max_drawdown(equity),
        "TotalReturn": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "WinRate": float((returns > 0).mean()),
    }
