from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import json
import os
import random

import numpy as np
import pandas as pd


@dataclass
class MegaConfig:
    SEASONAL_PERIOD: int = 1
    MIN_LENGTH: int = 7
    COUNTRY: str = "RU"
    MIN_DATE: pd.Timestamp = pd.to_datetime("1900-01-01")
    MAX_DATE: pd.Timestamp = pd.to_datetime(datetime.today())
    FH_SIZE: int = 1
    SEED: int | None = 42
    FIG_WIDTH: int = 900
    FIG_HEIGHT: int = 500
    COLOR: str = "#1f77b4"
    MARGIN: dict[str, int] = field(default_factory=lambda: dict(l=30, r=30, t=50, b=30))

    def __post_init__(self):
        self.MIN_DATE = pd.to_datetime(self.MIN_DATE)
        self.MAX_DATE = pd.to_datetime(self.MAX_DATE)

        if self.SEASONAL_PERIOD < 1:
            raise ValueError("SEASONAL_PERIOD must be >= 1")
        if self.MIN_LENGTH < 1:
            raise ValueError("MIN_LENGTH must be >= 1")
        if self.FH_SIZE < 1:
            raise ValueError("FH_SIZE must be >= 1")
        if self.MIN_DATE > self.MAX_DATE:
            raise ValueError("MIN_DATE must be <= MAX_DATE")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["MIN_DATE"] = self.MIN_DATE.isoformat()
        data["MAX_DATE"] = self.MAX_DATE.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MegaConfig":
        parsed = dict(data)
        if "MIN_DATE" in parsed:
            parsed["MIN_DATE"] = pd.to_datetime(parsed["MIN_DATE"])
        if "MAX_DATE" in parsed:
            parsed["MAX_DATE"] = pd.to_datetime(parsed["MAX_DATE"])
        return cls(**parsed)


CONFIG = MegaConfig()


def _apply_globals(cfg: MegaConfig) -> None:
    globals().update(cfg.__dict__)


def set_seed(seed: int | None = None) -> int | None:
    value = CONFIG.SEED if seed is None else seed
    CONFIG.SEED = value

    if value is None:
        return None

    random.seed(value)
    np.random.seed(value)
    os.environ["PYTHONHASHSEED"] = str(value)
    return value


def _infer_date_bounds_from_data(y: pd.DataFrame | pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not isinstance(y.index, (pd.MultiIndex, pd.DatetimeIndex)):
        raise ValueError("Expected DateTimeIndex or MultiIndex with date on last level")

    if isinstance(y.index, pd.MultiIndex):
        date_index = pd.to_datetime(y.index.get_level_values(-1))
    else:
        date_index = pd.to_datetime(y.index)

    return date_index.min(), date_index.max()


def initialize_config(
    y: pd.DataFrame | pd.Series | None = None,
    save_path: str | Path | None = None,
    load_path: str | Path | None = None,
    **overrides,
) -> MegaConfig:
    cfg = CONFIG

    if load_path is not None:
        cfg = load_config(load_path)

    if y is not None:
        min_date, max_date = _infer_date_bounds_from_data(y)
        cfg = MegaConfig.from_dict(
            {**cfg.to_dict(), "MIN_DATE": min_date, "MAX_DATE": max_date}
        )

    if overrides:
        cfg = MegaConfig.from_dict({**cfg.to_dict(), **overrides})

    set_config_obj(cfg)

    if save_path is not None:
        save_config(save_path)

    return CONFIG


def save_config(path: str | Path) -> Path:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(CONFIG.to_dict(), indent=2), encoding="utf-8")
    return file


def load_config(path: str | Path) -> MegaConfig:
    file = Path(path)
    data = json.loads(file.read_text(encoding="utf-8"))
    return MegaConfig.from_dict(data)


def set_config_obj(cfg: MegaConfig) -> MegaConfig:
    global CONFIG
    CONFIG = cfg
    _apply_globals(CONFIG)
    set_seed(CONFIG.SEED)
    return CONFIG


def get_config() -> MegaConfig:
    return CONFIG


def set_config(**kwargs):
    cfg = MegaConfig.from_dict({**CONFIG.to_dict(), **kwargs})
    return set_config_obj(cfg)


_apply_globals(CONFIG)
set_seed(CONFIG.SEED)
