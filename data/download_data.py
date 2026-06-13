"""Download, feature-engineer, split, and normalize daily OHLCV data.

The pipeline intentionally fits normalization statistics on the training split
only. In financial ML, using validation or test data to estimate scalers leaks
future distributional information into the agent and invalidates out-of-sample
evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
import yaml


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SCALER_STATS_PATH = PROJECT_ROOT / "data" / "scaler_stats.json"
YFINANCE_CACHE_DIR = PROJECT_ROOT / "data" / ".yfinance_cache"
BASE_FEATURES = ["open", "high", "low", "close", "volume"]
ENGINEERED_FEATURES = ["daily_return", "rolling_vol_5d", "ma_ratio_10", "ma_ratio_20"]
NORMALIZED_BASE_FEATURES = [f"norm_{feature}" for feature in BASE_FEATURES]
MODEL_FEATURES = NORMALIZED_BASE_FEATURES + ENGINEERED_FEATURES


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load YAML configuration for data dates, tickers, and feature windows.

    Args:
        config_path: Path to the project YAML configuration.

    Returns:
        Parsed configuration dictionary.
    """

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV bars for one ticker from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol.
        start: Inclusive start date in ``YYYY-MM-DD`` format.
        end: Inclusive end date in ``YYYY-MM-DD`` format.

    Returns:
        DataFrame with normalized column names and one row per trading day.

    Raises:
        RuntimeError: If Yahoo Finance returns no rows.
    """

    YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))

    # yfinance treats the end date as exclusive, so add one day to preserve the
    # user-specified inclusive project boundary.
    inclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = yf.download(
        ticker,
        start=start,
        end=inclusive_end,
        auto_adjust=False,
        progress=False,
        actions=False,
    )
    if data.empty:
        raise RuntimeError(f"No data downloaded for ticker {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data.columns = [str(column).lower().replace(" ", "_") for column in data.columns]
    data = data.rename(columns={"date": "date"})
    required_columns = ["date", "open", "high", "low", "close", "volume"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise RuntimeError(f"Missing required columns for {ticker}: {missing_columns}")

    data = data[required_columns].copy()
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    return data


def compute_features(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Compute return, volatility, and moving-average momentum features.

    Args:
        data: Raw OHLCV DataFrame with lowercase columns.
        config: Project configuration containing rolling feature windows.

    Returns:
        Feature-enriched DataFrame with rows containing complete history only.
    """

    rolling_vol_window = int(config["rolling_vol_window"])
    ma_windows = [int(window) for window in config["ma_windows"]]

    features = data.sort_values("date").reset_index(drop=True).copy()
    features["daily_return"] = features["close"].pct_change()
    features["rolling_vol_5d"] = features["daily_return"].rolling(
        window=rolling_vol_window,
        min_periods=rolling_vol_window,
    ).std()

    for window in ma_windows:
        moving_average = features["close"].rolling(window=window, min_periods=window).mean()
        features[f"ma_ratio_{window}"] = features["close"] / moving_average

    return features.dropna().reset_index(drop=True)


def split_by_date(data: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Split data into chronological train, validation, and test partitions.

    Args:
        data: Feature-enriched DataFrame with a ``date`` column.
        config: Project configuration containing split boundaries.

    Returns:
        Mapping from split name to split DataFrame.

    Raises:
        AssertionError: If split boundaries overlap or are not chronological.
    """

    date_series = pd.to_datetime(data["date"])
    train_start = pd.Timestamp(config["train_start"])
    train_end = pd.Timestamp(config["train_end"])
    val_start = pd.Timestamp(config["val_start"])
    val_end = pd.Timestamp(config["val_end"])
    test_start = pd.Timestamp(config["test_start"])
    test_end = pd.Timestamp(config["test_end"])

    # This assertion is a guardrail against data leakage through overlapping
    # date windows; model selection must never see test-period observations.
    assert train_start <= train_end < val_start <= val_end < test_start <= test_end

    splits = {
        "train": data.loc[(date_series >= train_start) & (date_series <= train_end)].copy(),
        "val": data.loc[(date_series >= val_start) & (date_series <= val_end)].copy(),
        "test": data.loc[(date_series >= test_start) & (date_series <= test_end)].copy(),
    }

    for split_name, split_data in splits.items():
        if split_data.empty:
            raise RuntimeError(f"{split_name} split is empty after date filtering.")
        splits[split_name] = split_data.reset_index(drop=True)

    max_train_date = pd.to_datetime(splits["train"]["date"]).max()
    min_val_date = pd.to_datetime(splits["val"]["date"]).min()
    max_val_date = pd.to_datetime(splits["val"]["date"]).max()
    min_test_date = pd.to_datetime(splits["test"]["date"]).min()
    assert max_train_date < min_val_date and max_val_date < min_test_date

    return splits


def fit_scaler(train_data: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float]]:
    """Fit z-score scaler statistics on training data only.

    Args:
        train_data: Training split used to estimate normalization statistics.
        feature_columns: Raw base columns to normalize.

    Returns:
        Mapping from feature name to mean and standard deviation.
    """

    scaler_stats: dict[str, dict[str, float]] = {}
    for column in feature_columns:
        mean = float(train_data[column].mean())
        std = float(train_data[column].std(ddof=0))
        if np.isclose(std, 0.0):
            LOGGER.warning("Feature %s has near-zero std; using 1.0 to avoid division by zero.", column)
            std = 1.0
        scaler_stats[column] = {"mean": mean, "std": std}
    return scaler_stats


def apply_scaler(
    data: pd.DataFrame,
    scaler_stats: dict[str, dict[str, float]],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Apply training-set scaler statistics to a split.

    Args:
        data: Split DataFrame to transform.
        scaler_stats: Training-only mean and standard deviation per feature.
        feature_columns: Raw base columns to normalize.

    Returns:
        DataFrame with normalized base feature columns added.
    """

    scaled = data.copy()
    for column in feature_columns:
        stats = scaler_stats[column]
        scaled[f"norm_{column}"] = (scaled[column] - stats["mean"]) / stats["std"]
    return scaled


def save_split(data: pd.DataFrame, ticker: str, split_name: str) -> Path:
    """Persist one processed split to ``data/raw``.

    Args:
        data: Processed split DataFrame.
        ticker: Ticker symbol used in the output filename.
        split_name: Split name such as ``train``, ``val``, or ``test``.

    Returns:
        Path of the saved CSV file.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / f"{ticker}_{split_name}.csv"
    output_columns = ["date"] + BASE_FEATURES + MODEL_FEATURES
    data.loc[:, output_columns].to_csv(output_path, index=False)
    return output_path


def process_ticker(ticker: str, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Run the complete download and preprocessing pipeline for one ticker.

    Args:
        ticker: Ticker symbol to download and process.
        config: Project configuration.

    Returns:
        Mapping from split name to processed split DataFrame.
    """

    raw_data = download_ohlcv(ticker, config["download_start"], config["download_end"])
    featured_data = compute_features(raw_data, config)
    splits = split_by_date(featured_data, config)

    # WARNING: fitting the scaler on validation or test data would leak future
    # distributional information into training and overstate strategy quality.
    scaler_stats = fit_scaler(splits["train"], BASE_FEATURES)
    transformed_splits = {
        split_name: apply_scaler(split_data, scaler_stats, BASE_FEATURES)
        for split_name, split_data in splits.items()
    }

    for split_name, split_data in transformed_splits.items():
        save_split(split_data, ticker, split_name)

    if ticker == config["ticker"]:
        SCALER_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SCALER_STATS_PATH.open("w", encoding="utf-8") as file:
            json.dump(scaler_stats, file, indent=2)

    return transformed_splits


def main() -> None:
    """Download all configured tickers and print a concise pipeline summary."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config = load_config()
    tickers = list(config["tickers"])
    summary: dict[str, dict[str, pd.DataFrame]] = {}

    for ticker in tickers:
        LOGGER.info("Processing ticker %s", ticker)
        summary[ticker] = process_ticker(ticker, config)

    default_ticker = str(config["ticker"])
    default_splits = summary[default_ticker]
    print(
        f"Train: {len(default_splits['train'])} rows | "
        f"Val: {len(default_splits['val'])} rows | "
        f"Test: {len(default_splits['test'])} rows"
    )
    print(f"Features: {MODEL_FEATURES}")
    print("Scaler stats saved to data/scaler_stats.json")


if __name__ == "__main__":
    main()
