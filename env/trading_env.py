"""Custom Gymnasium environment for single-asset RL trading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from gymnasium import spaces


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
FEATURE_COLUMNS = [
    "norm_open",
    "norm_high",
    "norm_low",
    "norm_close",
    "norm_volume",
    "daily_return",
    "rolling_vol_5d",
    "ma_ratio_10",
    "ma_ratio_20",
]


def load_default_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load project defaults used when callers pass a partial environment config.

    Args:
        config_path: YAML file containing project defaults.

    Returns:
        Default configuration dictionary.
    """

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class TradingEnv(gym.Env[np.ndarray, int]):
    """Single-asset long/flat trading environment.

    The state is a flattened lookback window of market features plus two
    portfolio-state features. Actions are discrete: hold, buy, or sell. The
    reward uses risk-aware shaping:

    - ``log_return`` captures compounding correctly, unlike arithmetic return.
    - ``transaction_cost`` penalizes overtrading and models real market friction.
    - ``volatility_penalty`` moves the objective toward risk-adjusted return,
      analogous to optimizing a Sharpe-like criterion instead of raw return.
    - Connection to thesis: this reward shaping is conceptually identical to the
      guidance reward in Raw2Drive v3; both encode domain knowledge into the
      reward signal to guide policies toward safe or profitable behavior.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        config: dict[str, Any] | None,
        max_steps: int | None = None,
    ) -> None:
        """Initialize the trading environment.

        Args:
            data: Processed market DataFrame containing close prices and model features.
            config: Environment configuration. Missing keys are filled from ``default.yaml``.
            max_steps: Optional episode step cap. If ``None``, the episode ends at data exhaustion.

        Raises:
            ValueError: If data is too short or required columns are missing.
        """

        super().__init__()
        default_config = load_default_config()
        self.config = {**default_config, **(config or {})}
        self.data = data.reset_index(drop=True).copy()
        self.feature_columns = list(FEATURE_COLUMNS)
        self.lookback_window = int(self.config["lookback_window"])
        self.initial_portfolio_value = float(self.config["initial_portfolio_value"])
        self.transaction_cost = float(self.config["transaction_cost"])
        self.invalid_action_penalty = float(self.config["invalid_action_penalty"])
        self.volatility_penalty_weight = float(self.config["volatility_penalty_weight"])
        self.action_hold = int(self.config["action_hold"])
        self.action_buy = int(self.config["action_buy"])
        self.action_sell = int(self.config["action_sell"])
        self.position_flat = int(self.config["position_flat"])
        self.position_long = int(self.config["position_long"])
        self.max_steps = max_steps

        required_columns = {"close", "daily_return", *self.feature_columns}
        missing_columns = sorted(required_columns.difference(self.data.columns))
        if missing_columns:
            raise ValueError(f"TradingEnv data missing required columns: {missing_columns}")
        if len(self.data) <= self.lookback_window:
            raise ValueError(
                f"TradingEnv requires more rows than lookback_window={self.lookback_window}; "
                f"received {len(self.data)} rows."
            )

        observation_size = self.lookback_window * len(self.feature_columns) + int(self.config["portfolio_state_size"])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(int(self.config["n_actions"]))

        self.current_index = self.lookback_window
        self.step_count = 0
        self.position = self.position_flat
        self.entry_price = 0.0
        self.portfolio_value = self.initial_portfolio_value
        self.last_reward = 0.0
        self.last_action_taken = self.action_hold
        self.portfolio_history: list[float] = []
        self.return_history: list[float] = []
        self.action_history: list[int] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the first tradable index.

        Args:
            seed: Optional random seed passed to Gymnasium for reproducibility.
            options: Reserved for Gymnasium compatibility.

        Returns:
            Tuple of initial observation and empty info dictionary.
        """

        super().reset(seed=seed)
        self.current_index = self.lookback_window
        self.step_count = 0
        self.position = self.position_flat
        self.entry_price = 0.0
        self.portfolio_value = self.initial_portfolio_value
        self.last_reward = 0.0
        self.last_action_taken = self.action_hold
        self.portfolio_history = [self.portfolio_value]
        self.return_history = []
        self.action_history = []
        return self._get_observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one trading action and advance the market by one day.

        Args:
            action: Discrete action where 0=hold, 1=buy, and 2=sell.

        Returns:
            Observation, reward, terminated flag, truncated flag, and info dictionary.
        """

        action_int = int(action)
        previous_position = self.position
        previous_price = self._price_at(self.current_index - 1)
        current_price = self._price_at(self.current_index)
        action_taken = self._resolve_action(action_int, current_price)
        invalid_penalty = self.invalid_action_penalty if action_taken != action_int else 0.0
        changed_position = previous_position != self.position

        if previous_position == self.position_long:
            log_return = float(np.log(current_price / previous_price))
            self.portfolio_value *= float(np.exp(log_return))
        else:
            log_return = 0.0

        # Transaction costs discourage churn and approximate commissions/slippage.
        transaction_cost = self.transaction_cost if changed_position else 0.0
        if changed_position:
            self.portfolio_value *= 1.0 - transaction_cost
        # Volatility penalty rewards smoother return streams, not just high raw PnL.
        volatility_penalty = self.volatility_penalty_weight * self._recent_return_volatility()
        # Log return models multiplicative wealth dynamics under compounding.
        reward = log_return - transaction_cost - volatility_penalty - invalid_penalty

        self.last_reward = float(reward)
        self.last_action_taken = action_taken
        self.step_count += 1
        self.current_index += 1
        self.portfolio_history.append(self.portfolio_value)
        self.return_history.append(float((self.portfolio_history[-1] / self.portfolio_history[-2]) - 1.0))
        self.action_history.append(action_taken)

        terminated = self.current_index >= len(self.data)
        truncated = self.max_steps is not None and self.step_count >= self.max_steps
        observation = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "daily_return": float(self.data.loc[self.current_index - 1, "daily_return"]),
            "action_taken": action_taken,
            "price": current_price,
        }
        return observation, float(reward), terminated, bool(truncated), info

    def render(self, mode: str = "human") -> None:
        """Render the latest environment state as one human-readable log line.

        Args:
            mode: Render mode. Only ``human`` is supported.

        Raises:
            ValueError: If a non-human render mode is requested.
        """

        if mode != "human":
            raise ValueError("TradingEnv only supports render(mode='human').")
        LOGGER.info(
            "Step %s | Price: %.2f | Position: %s | Portfolio: %.2f | Reward: %.6f",
            self.step_count,
            self._price_at(min(self.current_index, len(self.data) - 1)),
            self.position,
            self.portfolio_value,
            self.last_reward,
        )

    def _get_observation(self) -> np.ndarray:
        """Build the flattened market-window plus portfolio-state observation.

        Returns:
            Float32 observation vector with shape ``observation_space.shape``.
        """

        safe_index = min(self.current_index, len(self.data) - 1)
        start_index = safe_index - self.lookback_window
        market_window = self.data.loc[start_index : safe_index - 1, self.feature_columns].to_numpy(dtype=np.float32)
        current_price = self._price_at(safe_index)
        unrealized_pnl_pct = (
            (current_price - self.entry_price) / self.entry_price
            if self.position == self.position_long and self.entry_price > 0.0
            else 0.0
        )
        portfolio_state = np.array([float(self.position), float(unrealized_pnl_pct)], dtype=np.float32)
        return np.concatenate([market_window.flatten(), portfolio_state]).astype(np.float32)

    def _resolve_action(self, action: int, current_price: float) -> int:
        """Apply validity rules and mutate position state.

        Args:
            action: Requested action.
            current_price: Current close price used as trade execution price.

        Returns:
            Executed action after invalid actions are converted to hold.
        """

        if action == self.action_buy:
            if self.position == self.position_long:
                return self.action_hold
            self.position = self.position_long
            self.entry_price = current_price
            return self.action_buy

        if action == self.action_sell:
            if self.position == self.position_flat:
                return self.action_hold
            self.position = self.position_flat
            self.entry_price = 0.0
            return self.action_sell

        return self.action_hold

    def _price_at(self, index: int) -> float:
        """Read a close price at a bounded integer index.

        Args:
            index: Row index into the environment DataFrame.

        Returns:
            Close price as a float.
        """

        safe_index = min(max(index, 0), len(self.data) - 1)
        return float(self.data.loc[safe_index, "close"])

    def _recent_return_volatility(self) -> float:
        """Compute recent realized volatility used in reward shaping.

        Returns:
            Standard deviation of the last configured return window.
        """

        window = int(self.config["rolling_vol_window"])
        start_index = max(0, self.current_index - window + 1)
        recent_returns = self.data.loc[start_index : self.current_index, "daily_return"].to_numpy(dtype=np.float64)
        return float(np.std(recent_returns, ddof=0)) if len(recent_returns) > 0 else 0.0
