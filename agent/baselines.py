"""Baseline trading agents used for PPO comparison."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from utils.metrics import summarize_performance


class BuyAndHoldAgent:
    """Deterministic baseline that buys once, holds, and exits at the end."""

    def __init__(self) -> None:
        """Initialize internal action counter state."""

        self.step_index = 0
        self.action_hold: int | None = None
        self.action_buy: int | None = None
        self.action_sell: int | None = None

    def predict(self, obs: np.ndarray, is_last_step: bool = False) -> int:
        """Return buy on first step, sell on last step, otherwise hold.

        Args:
            obs: Current observation. It is unused because the policy is fixed.
            is_last_step: Whether the current decision is the final tradable step.

        Returns:
            Discrete action id.
        """

        _ = obs
        if self.action_hold is None or self.action_buy is None or self.action_sell is None:
            raise RuntimeError("BuyAndHoldAgent action ids must be synchronized with an environment first.")
        if self.step_index == 0:
            action = self.action_buy
        elif is_last_step:
            action = self.action_sell
        else:
            action = self.action_hold
        self.step_index += 1
        return action

    def evaluate(self, env: gym.Env[Any, int]) -> dict[str, float | list[float] | list[int]]:
        """Run one full episode and return standard performance metrics.

        Args:
            env: Trading environment.

        Returns:
            Metrics dictionary matching ``PPOTrader.evaluate``.
        """

        self.step_index = 0
        self._sync_action_ids(env)
        obs, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            current_index = int(getattr(env, "current_index"))
            data_length = len(getattr(env, "data"))
            is_last_step = current_index >= data_length - 1
            action = self.predict(obs, is_last_step=is_last_step)
            obs, _, terminated, truncated, _ = env.step(action)
        return summarize_performance(
            getattr(env, "return_history"),
            getattr(env, "portfolio_history"),
            getattr(env, "action_history"),
        )

    def _sync_action_ids(self, env: gym.Env[Any, int]) -> None:
        """Read action identifiers from the environment configuration.

        Args:
            env: Trading environment exposing action id attributes.
        """

        self.action_hold = int(getattr(env, "action_hold"))
        self.action_buy = int(getattr(env, "action_buy"))
        self.action_sell = int(getattr(env, "action_sell"))


class RandomAgent:
    """Seeded random-action baseline for measuring environment difficulty."""

    def __init__(self, seed: int) -> None:
        """Initialize a reproducible random number generator.

        Args:
            seed: Seed controlling random action selection.
        """

        self.rng = np.random.default_rng(seed)
        self.n_actions: int | None = None

    def predict(self, obs: np.ndarray) -> int:
        """Sample a uniformly random action from hold, buy, and sell.

        Args:
            obs: Current observation. It is unused by the random policy.

        Returns:
            Discrete action id.
        """

        _ = obs
        if self.n_actions is None:
            raise RuntimeError("RandomAgent action count must be synchronized with an environment first.")
        return int(self.rng.integers(0, self.n_actions))

    def evaluate(self, env: gym.Env[Any, int]) -> dict[str, float | list[float] | list[int]]:
        """Run one full episode and return standard performance metrics.

        Args:
            env: Trading environment.

        Returns:
            Metrics dictionary matching ``PPOTrader.evaluate``.
        """

        self.n_actions = int(env.action_space.n)
        obs, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = self.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
        return summarize_performance(
            getattr(env, "return_history"),
            getattr(env, "portfolio_history"),
            getattr(env, "action_history"),
        )
