import numpy as np
import pandas as pd

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
import os, random


@dataclass
class Config:
    SEASONAL_PERIOD: int = 1
    FH_SIZE: int = 1
    MAX_LAG_W_SIZE: int = 7
    MIN_LENGTH: int = 7
    COUNTRY: str = "RU"
    MIN_DATE: pd.Timestamp = pd.to_datetime("1900-01-01")
    MAX_DATE: pd.Timestamp = pd.to_datetime(datetime.today())
    SEED: int = 42
    FIG_WIDTH: int = 900
    FIG_HEIGHT: int = 500
    COLOR: str = "#1f77b4"
    MARGIN: dict[str, int] = field(default_factory=lambda: dict(l=30, r=30, t=50, b=30))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data["MIN_LENGTH"] = data["MAX_LAG_W_SIZE"] + data["FH_SIZE"] + 1
        return cls(**data)


CONFIG = Config()


def set_seed(seed: int | None):
    if seed is not None:
        CONFIG.SEED = seed

    random.seed(CONFIG.SEED)
    np.random.seed(CONFIG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CONFIG.SEED)


def _apply_globals(cfg: Config):
    globals().update(cfg.to_dict())


def set_config(**kwargs):
    global CONFIG

    CONFIG = Config.from_dict({**CONFIG.to_dict(), **kwargs})
    _apply_globals(CONFIG)
    set_seed(CONFIG.SEED)


_apply_globals(CONFIG)
set_seed(CONFIG.SEED)
