"""Risk and performance metrics for trading strategy evaluation."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def annualized_return(returns: list[float], periods_per_year: int = 252) -> float:
    """Compute annualized geometric return.

    Formula:
    ``Annualized Return = (1 + total_return)^(periods_per_year / n_days) - 1``.

    Args:
        returns: Per-period simple returns.
        periods_per_year: Number of trading periods in one calendar year.

    Returns:
        Annualized return as a decimal.
    """

    if not returns:
        return 0.0
    returns_array = np.asarray(returns, dtype=np.float64)
    cumulative_growth = float(np.prod(1.0 + returns_array))
    if cumulative_growth <= 0.0:
        return -1.0
    return float(cumulative_growth ** (periods_per_year / len(returns_array)) - 1.0)


def sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualized Sharpe ratio.

    Formula:
    ``Sharpe Ratio = (mean(R) - Rf) / std(R) * sqrt(periods_per_year)``.
    Measures excess return per unit of risk. Risk-free rate assumed 0 for
    simplicity; US 3-month T-bill yields were near 5% in 2023, but 0 is standard
    for RL baselines to isolate agent skill.

    Args:
        returns: Per-period simple returns.
        risk_free_rate: Annual risk-free rate as a decimal.
        periods_per_year: Number of trading periods in one calendar year.

    Returns:
        Annualized Sharpe ratio. Returns 0 when variance is unavailable.
    """

    if len(returns) < 2:
        return 0.0
    returns_array = np.asarray(returns, dtype=np.float64)
    per_period_rf = risk_free_rate / periods_per_year
    excess_returns = returns_array - per_period_rf
    std = float(np.std(excess_returns, ddof=1))
    if math.isclose(std, 0.0):
        return 0.0
    return float(np.mean(excess_returns) / std * math.sqrt(periods_per_year))


def max_drawdown(equity_curve: list[float]) -> float:
    """Compute the maximum peak-to-trough drawdown.

    Formula:
    ``Max Drawdown = max_t((peak_before_t - trough_after_peak) / peak_before_t)``.
    Measures the worst peak-to-trough decline, a critical risk-management
    quantity for deployable trading strategies.

    Args:
        equity_curve: Sequence of portfolio values.

    Returns:
        Maximum drawdown as a positive decimal.
    """

    if not equity_curve:
        return 0.0
    equity_array = np.asarray(equity_curve, dtype=np.float64)
    running_peak = np.maximum.accumulate(equity_array)
    drawdowns = np.divide(
        running_peak - equity_array,
        running_peak,
        out=np.zeros_like(equity_array),
        where=running_peak != 0.0,
    )
    return float(np.max(drawdowns))


def win_rate(returns: list[float]) -> float:
    """Compute the fraction of non-zero return days that are profitable.

    Formula:
    ``Win Rate = count(r > 0) / count(r != 0)``.

    Args:
        returns: Per-period simple returns.

    Returns:
        Win rate as a decimal. Returns 0 when no non-zero returns exist.
    """

    non_zero_returns = [return_value for return_value in returns if not math.isclose(return_value, 0.0)]
    if not non_zero_returns:
        return 0.0
    wins = sum(return_value > 0.0 for return_value in non_zero_returns)
    return float(wins / len(non_zero_returns))


def calmar_ratio(returns: list[float], equity_curve: list[float]) -> float:
    """Compute the Calmar ratio.

    Formula:
    ``Calmar Ratio = Annualized Return / |Max Drawdown|``.
    Preferred by many hedge funds over Sharpe for strategy comparison because it
    emphasizes realized capital impairment.

    Args:
        returns: Per-period simple returns.
        equity_curve: Sequence of portfolio values.

    Returns:
        Calmar ratio. Returns 0 when max drawdown is zero.
    """

    drawdown = max_drawdown(equity_curve)
    if math.isclose(drawdown, 0.0):
        return 0.0
    return float(annualized_return(returns) / abs(drawdown))


def total_return(equity_curve: list[float]) -> float:
    """Compute total strategy return from the first to last equity value.

    Args:
        equity_curve: Sequence of portfolio values.

    Returns:
        Total return as a decimal.
    """

    if len(equity_curve) < 2 or math.isclose(equity_curve[0], 0.0):
        return 0.0
    return float(equity_curve[-1] / equity_curve[0] - 1.0)


def summarize_performance(
    returns: Sequence[float],
    equity_curve: Sequence[float],
    actions: Sequence[int] | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float | list[float] | list[int]]:
    """Create the standard metric dictionary used by all agents.

    Args:
        returns: Per-period simple returns.
        equity_curve: Sequence of portfolio values.
        actions: Optional executed action history.
        risk_free_rate: Annual risk-free rate as a decimal.
        periods_per_year: Number of trading periods in one calendar year.

    Returns:
        Dictionary containing scalar metrics and plotting series.
    """

    returns_list = [float(return_value) for return_value in returns]
    equity_list = [float(value) for value in equity_curve]
    action_list = [int(action) for action in actions] if actions is not None else []
    return {
        "return": total_return(equity_list),
        "annual_return": annualized_return(returns_list, periods_per_year),
        "sharpe": sharpe_ratio(returns_list, risk_free_rate, periods_per_year),
        "max_dd": max_drawdown(equity_list),
        "win_rate": win_rate(returns_list),
        "calmar": calmar_ratio(returns_list, equity_list),
        "equity_curve": equity_list,
        "returns": returns_list,
        "actions": action_list,
    }

