# ProjectV — First Systematic Equity Strategy (Deprecated)



The original system that started everything. ProjectV was my

first attempt at building a institutional-grade systematic

trading strategy from scratch. It is deprecated and succeeded

by ProjectR, but preserved as documentation of the development

process that made ProjectR possible.



**Status:** Deprecated | Succeeded by [ProjectR](https://github.com/vnuthi07/ProjectR-overview)

**Built:** 2024-2025 (\~2-3 months, 7 major versions)

**Backtest Period:** 2005-2025



\---



## Performance Summary



| Metric | Peak Version | Final Version |

|---|---|---|

| CAGR | 11.0% | 8.2% |

| Sharpe Ratio | 0.85 | 0.65 |

| Annualized Vol | 13.3% | 10.7% |

| Max Drawdown | -21.4% | -20.9% |

| Total Return | 595% | 419% |

| Avg Turnover | 6.3% | 7.7% |



\*Peak metrics were achieved during development but could

not be consistently replicated after further experimentation —

the signal that led directly to ProjectR's disciplined

research infrastructure.\*



\---



## Equity Curves



### Peak Version (0.85 Sharpe)

!\[ProjectV Peak Equity Curve](assets/equity-curve-peak.png)



### Final Version (0.65 Sharpe)

!\[ProjectV Final Equity Curve](assets/equity-curve-final.png)



\*The degradation between peak and final versions is visible

in the pre-2020 period — changes made during experimentation

hurt early period performance while post-2020 behavior

remained similar.\*



\---



## What ProjectV Was



ProjectV was built around a core thesis: capture momentum

across a diversified ETF universe while managing downside

risk through regime detection and volatility targeting.



**Core Components:**

* Three-state regime classifier (risk\_on, neutral, risk\_off)

&#x20; using SPY 150-day return and 20-day realized volatility

* Multi-horizon momentum signals (21, 63, 126 day lookbacks)

&#x20; with fixed weights (0.5, 0.3, 0.2)

* Vol-normalized cross-sectional ranking
* Hard direction gate — negative 63-day trend excluded

&#x20; asset from long consideration entirely

* Volatility targeting with drawdown de-risking
* Correlation penalty (flat 0.40 multiplier above 0.75

&#x20; avg pairwise correlation threshold)

* Alpha sleeve (25 tickers) + hedge sleeve (7 tickers)
* ML layer planned but never implemented — stubs existed

&#x20; but were empty



\---



## Development History — 7 Versions



ProjectV went through 7 major versions before being deprecated.

The pattern was consistent across versions:



1. Add new feature or change universe
2. In-sample metrics improve temporarily
3. Metrics degrade after further changes
4. Peak performance impossible to replicate



**What I tried across versions:**

* Universe changes — different ticker combinations,

&#x20; adding and removing assets

* Tri-horizon momentum thinking adding more lookback periods
* Various ML implementations that added noise rather

&#x20; than signal

* Parameter adjustments chasing better Sharpe



The core problem: I was experimenting without hypotheses.

Every change was "let's try this and see if it helps"

rather than "here's why this should improve performance

and here's how I'll test it." The result was a system

that became progressively harder to understand and

less reliable.



\---



## Fundamental Flaws Discovered



### 1\. Regime Classifier Was Never Wired In

The regime classifier existed as a separate module but

was never properly connected to portfolio construction.

The engine computed regime labels that downstream

allocation logic was ignoring entirely.



### 2\. Universe Overlap

TLT, IEF, SHY, GLD, SLV, UUP appeared in both alpha

and hedge universes simultaneously. The system could

hold the same asset in both sleeves at once with weights

adding up uncontrollably. No overlap validation existed.



### 3\. Structural Asset Problem

VIXY was in the alpha universe despite massive negative

carry from VIX futures roll costs — structurally

unsuitable as a momentum signal source.



### 4\. Correlation Penalty Design Flaw

A flat 0.40 gross scale multiplier applied whenever

average pairwise correlation exceeded 0.75 — regardless

of regime. During crisis periods when everything

correlates by definition, this double-punished an

already-defensive portfolio.



### 5\. No Research Infrastructure

One in-sample backtest. Walk-forward and stress test

modules existed in code but outputs were never

integrated into reporting. No Monte Carlo. No parameter

sensitivity. No factor decomposition. Evaluating

entirely on in-sample metrics.



### 6\. Monolithic Architecture

\~1,000 line engine.py with backtest loop, regime

classifier, signal computation, weight construction,

and risk management all interleaved. Adding any

feature required understanding the entire file.

No unit tests.



\---



## What I Learned



**On overfitting:** Peak metrics that can't be replicated

aren't evidence of alpha — they're evidence of overfit.

A strategy you can't explain or reproduce isn't a strategy.



**On complexity:** Adding features without hypotheses

doesn't improve a system. It adds noise and makes

results harder to interpret. Every addition needs

a reason and a test.



**On architecture:** Modularity isn't just good software

engineering — it's essential for a system you want to

understand, debug, and trust. A 1,000 line file where

everything is interleaved isn't a trading system,

it's a script.



**On research:** An in-sample backtest tells you almost

nothing. Out-of-sample validation, Monte Carlo, and

parameter sensitivity are the evidence that matters.



\---



## Why It Was Worth Building



Every flaw in ProjectV became a design requirement in ProjectR:



| ProjectV Flaw | ProjectR Solution |

|---|---|

| 3 regime states, poorly connected | 6 regimes, fully wired into all decisions |

| Universe overlap unvalidated | Hard assertion at startup |

| ML never implemented | XGBoost + LightGBM ensemble |

| Correlation penalty regime-blind | Fully regime-conditioned |

| In-sample backtest only | Walk-forward, Monte Carlo, sensitivity |

| Monolithic engine.py | Fully modular package, unit tested |

| No hypothesis-driven development | Every change tested against specific thesis |



ProjectV taught me what to build.

ProjectR taught me how to build it correctly.



→ \*\*[See ProjectR](https://github.com/vnuthi07/ProjectR-overview)

for the rebuilt system.\*\*



\---



## Sample Outputs



See `sample-outputs/` for metrics JSON files from

peak and final versions.



\---



\*Strategy code is private. This repository contains

documentation, equity curves, and metrics only.\*

