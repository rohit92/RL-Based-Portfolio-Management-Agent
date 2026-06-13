"""Evaluation entry point comparing PPO against deterministic and random baselines."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = PROJECT_ROOT / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3.common.utils import set_random_seed

from agent.baselines import BuyAndHoldAgent, RandomAgent
from agent.ppo_agent import PPOTrader
from env.trading_env import TradingEnv
from utils.plot import plot_action_distribution, plot_equity_curves


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration from disk.

    Args:
        config_path: Path to a YAML config file.

    Returns:
        Parsed configuration dictionary.
    """

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_global_seeds(seed: int) -> None:
    """Set random seeds for reproducible evaluation.

    Args:
        seed: Reproducibility seed from configuration.
    """

    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(ticker: str, split_name: str) -> pd.DataFrame:
    """Load one processed ticker split from ``data/raw``.

    Args:
        ticker: Ticker symbol.
        split_name: Split name.

    Returns:
        Processed split DataFrame.
    """

    path = PROJECT_ROOT / "data" / "raw" / f"{ticker}_{split_name}.csv"
    return pd.read_csv(path)


def main() -> None:
    """Evaluate PPO, buy-and-hold, and random agents on the test split."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ticker", default="AAPL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config = load_config(str(PROJECT_ROOT / args.config))
    config["ticker"] = args.ticker
    set_global_seeds(int(config["seed"]))

    test_data = load_split(args.ticker, "test")
    train_env = TradingEnv(load_split(args.ticker, "train"), config)
    ppo_env = TradingEnv(test_data, config)
    buy_hold_env = TradingEnv(test_data, config)
    random_env = TradingEnv(test_data, config)

    ppo = PPOTrader(config, train_env)
    checkpoint_path = PROJECT_ROOT / str(config["checkpoint_dir"]) / str(config["best_model_name"])
    if not checkpoint_path.exists():
        checkpoint_path = PROJECT_ROOT / str(config["checkpoint_dir"]) / str(config["final_model_name"])
    ppo.load(str(checkpoint_path))

    agents = {
        "PPO": (ppo, ppo_env),
        "Buy-and-Hold": (BuyAndHoldAgent(), buy_hold_env),
        "Random": (RandomAgent(seed=int(config["seed"])), random_env),
    }
    results: dict[str, dict[str, Any]] = {}
    for agent_name, (agent, env) in agents.items():
        results[agent_name] = agent.evaluate(env)

    print("=" * 65)
    print(f"{'Agent':<20} {'Return':>8} {'Sharpe':>8} " f"{'MaxDD':>8} {'WinRate':>8} {'Calmar':>8}")
    print("=" * 65)
    for agent_name, metrics in results.items():
        print(
            f"{agent_name:<20} {metrics['return']:>7.2%} "
            f"{metrics['sharpe']:>8.3f} {metrics['max_dd']:>7.2%} "
            f"{metrics['win_rate']:>7.2%} {metrics['calmar']:>8.3f}"
        )
    print("=" * 65)

    results_dir = PROJECT_ROOT / str(config["results_dir"])
    plot_equity_curves(
        {agent_name: metrics["equity_curve"] for agent_name, metrics in results.items()},
        str(results_dir / str(config["equity_curve_name"])),
    )
    plot_action_distribution(
        [int(action) for action in results["PPO"]["actions"]],
        str(results_dir / str(config["action_distribution_name"])),
    )


if __name__ == "__main__":
    main()
