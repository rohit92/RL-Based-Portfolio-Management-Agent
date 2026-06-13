"""PPO training and evaluation wrapper for the trading environment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from utils.metrics import sharpe_ratio, summarize_performance


LOGGER = logging.getLogger(__name__)


class ValidationSharpeCallback(BaseCallback):
    """Stable-Baselines callback that evaluates validation Sharpe periodically."""

    def __init__(self, trader: "PPOTrader", val_env: gym.Env[Any, int], eval_freq: int) -> None:
        """Initialize validation evaluator state.

        Args:
            trader: Parent ``PPOTrader`` wrapper.
            val_env: Validation trading environment.
            eval_freq: Number of training calls between validation episodes.
        """

        super().__init__(verbose=0)
        self.trader = trader
        self.val_env = val_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        """Run validation when the configured frequency is reached.

        Returns:
            ``True`` to continue training.
        """

        reward_value = float(np.mean(self.locals.get("rewards", [0.0])))
        self.trader.training_rewards.append(reward_value)
        if self.n_calls % self.eval_freq != 0:
            return True

        returns = self.trader.run_deterministic_episode(self.val_env)["returns"]
        val_sharpe = sharpe_ratio(
            [float(return_value) for return_value in returns],
            risk_free_rate=float(self.trader.config["risk_free_rate"]),
            periods_per_year=int(self.trader.config["periods_per_year"]),
        )
        self.trader.val_sharpes.append(float(val_sharpe))
        LOGGER.info("Step %s | Val Sharpe: %.3f", self.num_timesteps, val_sharpe)

        if val_sharpe > self.trader.best_val_sharpe:
            self.trader.best_val_sharpe = float(val_sharpe)
            self.trader.best_step = int(self.num_timesteps)
            checkpoint_path = self.trader.checkpoint_dir / str(self.trader.config["best_model_name"])
            self.model.save(checkpoint_path)
            LOGGER.info("Saved new best PPO checkpoint to %s", checkpoint_path)
        return True


class PPOTrader:
    """Production-oriented PPO wrapper for training and evaluating a trading policy."""

    def __init__(self, config: dict[str, Any], env: gym.Env[Any, int]) -> None:
        """Initialize Stable-Baselines3 PPO with project hyperparameters.

        Args:
            config: Full project configuration.
            env: Training environment.
        """

        self.config = config
        self.seed = int(config["seed"])
        set_random_seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.checkpoint_dir = Path(str(config["checkpoint_dir"]))
        self.runs_dir = Path(str(config["runs_dir"]))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.env = Monitor(env)
        self.best_val_sharpe = float("-inf")
        self.best_step = 0
        self.training_rewards: list[float] = []
        self.val_sharpes: list[float] = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy_kwargs = {"net_arch": list(config["policy_hidden_layers"])}

        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=float(config["learning_rate"]),
            n_steps=int(config["n_steps"]),
            batch_size=int(config["batch_size"]),
            n_epochs=int(config["n_epochs"]),
            gamma=float(config["gamma"]),
            gae_lambda=float(config["gae_lambda"]),
            clip_range=float(config["clip_range"]),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(self.runs_dir),
            seed=self.seed,
            device=device,
            verbose=0,
        )
        LOGGER.info("Initialized PPO on device=%s with policy net_arch=%s", device, policy_kwargs["net_arch"])

    def train(self, total_timesteps: int, val_env: gym.Env[Any, int]) -> None:
        """Train PPO and save best-validation and final checkpoints.

        Args:
            total_timesteps: Total PPO environment steps.
            val_env: Validation environment for periodic Sharpe evaluation.
        """

        callback = ValidationSharpeCallback(self, val_env, int(self.config["eval_freq"]))
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="ppo_trading",
            progress_bar=False,
        )
        final_path = self.checkpoint_dir / str(self.config["final_model_name"])
        self.model.save(final_path)
        LOGGER.info("Saved final PPO model to %s", final_path)

    def evaluate(self, env: gym.Env[Any, int]) -> dict[str, float | list[float] | list[int]]:
        """Run one deterministic episode and return standard performance metrics.

        Args:
            env: Evaluation environment.

        Returns:
            Metrics dictionary from ``utils.metrics.summarize_performance``.
        """

        episode = self.run_deterministic_episode(env)
        return summarize_performance(
            episode["returns"],
            episode["equity_curve"],
            episode["actions"],
            risk_free_rate=float(self.config["risk_free_rate"]),
            periods_per_year=int(self.config["periods_per_year"]),
        )

    def load(self, path: str) -> None:
        """Load PPO weights from a checkpoint path.

        Args:
            path: Stable-Baselines3 ``.zip`` model path.
        """

        self.model = PPO.load(path, env=self.env, device="cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loaded PPO checkpoint from %s", path)

    def run_deterministic_episode(self, env: gym.Env[Any, int]) -> dict[str, list[float] | list[int]]:
        """Run a full deterministic PPO episode and collect trajectories.

        Args:
            env: Trading environment.

        Returns:
            Dictionary with returns, equity curve, and action history.
        """

        obs, _ = env.reset(seed=self.seed)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
        return {
            "returns": list(getattr(env, "return_history")),
            "equity_curve": list(getattr(env, "portfolio_history")),
            "actions": list(getattr(env, "action_history")),
        }
