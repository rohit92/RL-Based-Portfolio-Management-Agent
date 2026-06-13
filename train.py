"""Training entry point for the RL trading agent."""

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

from agent.ppo_agent import PPOTrader
from env.trading_env import TradingEnv
from utils.plot import plot_training_curve


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
    """Set Python-library random seeds used by the training stack.

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
    """Train PPO on the configured training split and evaluate on validation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ticker", default="AAPL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config = load_config(str(PROJECT_ROOT / args.config))
    config["ticker"] = args.ticker
    set_global_seeds(int(config["seed"]))

    train_data = load_split(args.ticker, "train")
    val_data = load_split(args.ticker, "val")
    train_env = TradingEnv(train_data, config)
    val_env = TradingEnv(val_data, config)

    trader = PPOTrader(config, train_env)
    trader.train(total_timesteps=int(config["total_timesteps"]), val_env=val_env)

    results_dir = PROJECT_ROOT / str(config["results_dir"])
    plot_training_curve(
        trader.training_rewards,
        trader.val_sharpes,
        str(results_dir / str(config["training_curve_name"])),
    )
    print(f"Training complete. Best val Sharpe: {trader.best_val_sharpe:.3f} at step {trader.best_step}")


if __name__ == "__main__":
    main()
