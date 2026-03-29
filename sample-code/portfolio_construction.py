"""
portfolio_construction.py — Cross-Sectional Rank Portfolio Construction
=========================================================================
Implements score-to-weight conversion for a long/short equity strategy.

This module handles the translation of asset scores (from momentum signals
and ML predictions) into portfolio weights with proper risk controls:
- Gross exposure targeting
- Per-position weight caps
- Long/short leg balancing
- Weight renormalization after filtering

Used in ProjectV's portfolio construction pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_long_short_weights(
    scores: pd.Series,
    n_long: int,
    n_short: int,
    gross_target: float = 1.0,
    long_frac: float = 0.5,
    max_position_weight: float = 0.25,
    min_trade_weight: float = 0.01,
    current_weights: pd.Series | None = None,
) -> pd.Series:
    """
    Convert cross-sectional scores into long/short portfolio weights.

    Selects top n_long assets for the long leg and bottom n_short for
    the short leg, then scales to gross_target exposure with per-position
    caps applied.

    Args:
        scores: Cross-sectional signal scores indexed by ticker.
                Higher score = stronger long candidate.
        n_long: Number of assets to hold long
        n_short: Number of assets to hold short (0 for long-only)
        gross_target: Total gross exposure (1.0 = 100% gross)
        long_frac: Fraction of gross exposure allocated to long leg.
                   (1 - long_frac) goes to short leg.
                   In risk-off regimes this shifts toward 1.0 (long-only)
        max_position_weight: Maximum absolute weight per position
        min_trade_weight: Minimum weight change required to execute trade.
                          Filters out noise trades that generate unnecessary
                          transaction costs. Larger in volatile regimes.
        current_weights: Existing weights for hysteresis filtering.
                         If provided, only update positions where change
                         exceeds min_trade_weight.

    Returns:
        pd.Series of portfolio weights indexed by ticker.
        Positive = long, negative = short.

    Example:
        >>> scores = pd.Series({'SPY': 0.8, 'QQQ': 0.6, 'TLT': -0.4,
        ...                     'GLD': 0.2, 'IEF': -0.3})
        >>> weights = rank_long_short_weights(scores, n_long=2, n_short=1)
        >>> print(weights)
        SPY    0.333...
        QQQ    0.333...
        TLT   -0.333...
        GLD    0.000...
        IEF    0.000...
    """
    s = scores.dropna()
    if s.empty:
        return pd.Series(dtype=float)

    # Clamp to available assets
    n_long = max(0, min(n_long, len(s)))
    n_short = max(0, min(n_short, len(s)))

    if n_long == 0 and n_short == 0:
        return pd.Series(0.0, index=s.index)

    # Select top/bottom assets
    long_tickers = s.nlargest(n_long).index if n_long > 0 else pd.Index([])
    short_tickers = s.nsmallest(n_short).index if n_short > 0 else pd.Index([])

    # Ensure no overlap between long and short legs
    overlap = long_tickers.intersection(short_tickers)
    if not overlap.empty:
        short_tickers = short_tickers.difference(overlap)

    w = pd.Series(0.0, index=s.index)

    # Allocate gross exposure to each leg
    long_gross = gross_target * long_frac
    short_gross = gross_target * (1.0 - long_frac)

    if len(long_tickers) > 0:
        w.loc[long_tickers] = long_gross / len(long_tickers)
    if len(short_tickers) > 0:
        w.loc[short_tickers] = -(short_gross / len(short_tickers))

    # Apply per-position cap
    if max_position_weight is not None:
        w = w.clip(
            lower=-abs(max_position_weight),
            upper=abs(max_position_weight)
        )
        # Renormalize after clipping to maintain gross target
        gross_after_clip = float(np.abs(w.values).sum())
        if gross_after_clip > 1e-8:
            w = w * (gross_target / gross_after_clip)

    # Apply hysteresis filter — only trade if change exceeds threshold
    if current_weights is not None and min_trade_weight > 0:
        current_aligned = current_weights.reindex(w.index).fillna(0.0)
        delta = (w - current_aligned).abs()
        # Keep existing weight where change is too small
        w = w.where(delta >= min_trade_weight, current_aligned)
        # Renormalize gross after hysteresis filtering
        gross_final = float(np.abs(w.values).sum())
        if gross_final > 1e-8:
            w = w * (gross_target / gross_final)

    return w


def compute_turnover(
    new_weights: pd.Series,
    old_weights: pd.Series,
) -> float:
    """
    Compute one-way portfolio turnover between rebalances.

    Turnover = sum of absolute weight changes / 2
    (divided by 2 because a buy and a sell are one "round trip")

    Args:
        new_weights: Target portfolio weights
        old_weights: Previous portfolio weights

    Returns:
        One-way turnover as a fraction (e.g. 0.15 = 15% turnover)

    Note:
        Lower turnover = lower transaction costs. ProjectV targets
        < 10% monthly turnover to keep costs manageable at 2 bps
        per side.
    """
    all_tickers = new_weights.index.union(old_weights.index)
    new = new_weights.reindex(all_tickers).fillna(0.0)
    old = old_weights.reindex(all_tickers).fillna(0.0)
    return float((new - old).abs().sum() / 2.0)
