"""Publication-quality plotting utilities for RL trading experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = PROJECT_ROOT / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent_dir(save_path: str) -> None:
    """Create the parent directory for a plot path if needed.

    Args:
        save_path: Destination path for a figure.
    """

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def _drawdown_series(equity_curve: Sequence[float]) -> np.ndarray:
    """Compute a drawdown series for plotting.

    Args:
        equity_curve: Sequence of portfolio values.

    Returns:
        Drawdown values as negative decimals.
    """

    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.size == 0:
        return np.asarray([], dtype=np.float64)
    peaks = np.maximum.accumulate(equity)
    return np.divide(equity - peaks, peaks, out=np.zeros_like(equity), where=peaks != 0.0)


def plot_equity_curves(results: dict[str, list[float]], save_path: str) -> None:
    """Plot equity curves and drawdowns for all agents on shared axes.

    Args:
        results: Mapping such as ``{"PPO": [values...], "BuyAndHold": [values...]}``.
        save_path: Output PNG path.
    """

    _ensure_parent_dir(save_path)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.0]},
    )

    for agent_name, equity_curve in results.items():
        x_axis = np.arange(len(equity_curve))
        axes[0].plot(x_axis, equity_curve, linewidth=2.0, label=agent_name)
        axes[1].plot(x_axis, _drawdown_series(equity_curve), linewidth=1.6, label=agent_name)

    axes[0].set_title("Strategy Equity Curves")
    axes[0].set_ylabel("Portfolio Value")
    axes[0].legend(frameon=False)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Trading Step")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(rewards: list[float], val_sharpes: list[float], save_path: str) -> None:
    """Plot training reward and validation Sharpe checkpoints.

    Args:
        rewards: Per-step or per-rollout rewards collected during training.
        val_sharpes: Validation Sharpe ratios at each evaluation checkpoint.
        save_path: Output PNG path.
    """

    _ensure_parent_dir(save_path)
    rolling_window = min(100, max(1, len(rewards)))
    rewards_array = np.asarray(rewards, dtype=np.float64)
    if rewards_array.size >= rolling_window:
        kernel = np.ones(rolling_window, dtype=np.float64) / rolling_window
        rolling_rewards = np.convolve(rewards_array, kernel, mode="valid")
        reward_x = np.arange(rolling_rewards.size) + rolling_window - 1
    else:
        rolling_rewards = rewards_array
        reward_x = np.arange(rewards_array.size)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11.0, 7.0))
    axes[0].plot(reward_x, rolling_rewards, color="#1f77b4", linewidth=2.0)
    axes[0].set_title("Rolling Mean Training Reward")
    axes[0].set_ylabel("Reward")
    axes[1].plot(np.arange(len(val_sharpes)), val_sharpes, marker="o", color="#2ca02c", linewidth=2.0)
    axes[1].set_title("Validation Sharpe at Evaluation Checkpoints")
    axes[1].set_xlabel("Evaluation Checkpoint")
    axes[1].set_ylabel("Sharpe")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_action_distribution(actions: list[int], save_path: str) -> None:
    """Plot how often the agent chose hold, buy, and sell.

    Args:
        actions: Executed action sequence.
        save_path: Output PNG path.
    """

    _ensure_parent_dir(save_path)
    labels = ["Hold", "Buy", "Sell"]
    counts = [actions.count(action_id) for action_id in range(len(labels))]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.bar(labels, counts, color=["#4c78a8", "#59a14f", "#e15759"], width=0.6)
    axis.set_title("PPO Action Distribution")
    axis.set_ylabel("Count")
    axis.set_xlabel("Action")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
