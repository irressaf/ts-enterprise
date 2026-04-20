import pandas as pd
from pathlib import Path
import megatron.config as config
import pytest

pytestmark = pytest.mark.unit


# Set global config for unit tests placing it inside autocall fixture
@pytest.fixture(autouse=True)
def test_config():
    config.set_config(
        SEASONAL_PERIOD=7,
        MAX_LAG_W_SIZE=30,
        MIN_DATE=pd.to_datetime("2025-01-01"),
        MAX_DATE=pd.to_datetime("2025-12-31"),
        COUNTRY="EC",
    )


from megatron.transformers import (
    DemandClassifier,
    PlateauDetector,
    ChangePointDetector,
)


# set up intstance which is a reusable synthetic data initialized inside itself
@pytest.fixture
def synthetic_data():
    data_path = (
        Path(__file__).resolve().parents[1] / "data" / "test" / "unit_test_synthetic.pq"
    )
    return pd.read_parquet(data_path)


def test_demand_classifier(synthetic_data):
    temp = DemandClassifier().fit_transform(synthetic_data)
    actual, detected = temp.index.get_level_values("class"), temp["class"].values  # type: ignore
    assert all(actual == detected)


def test_plateau_detector(synthetic_data):
    data = synthetic_data.loc["erratic"].droplevel([0, 1])
    temp = PlateauDetector(w=2 * config.SEASONAL_PERIOD, value=0, n_jobs=1).fit_transform(data)  # type: ignore
    assert temp.size > 0  # type: ignore

    temp = PlateauDetector(w=5 * config.SEASONAL_PERIOD, value=0, n_jobs=1).fit_transform(data)  # type: ignore
    assert temp.size == 0  # type: ignore

    temp = PlateauDetector(w=2 * config.SEASONAL_PERIOD, n_jobs=1).fit_transform(data)  # type: ignore
    assert temp.size == 0  # type: ignore

    data = synthetic_data.loc["smooth"].droplevel([0, 1])
    temp = PlateauDetector(w=2 * config.SEASONAL_PERIOD, value=0, n_jobs=1).fit_transform(data)  # type: ignore
    assert temp.size == 0  # type: ignore


def test_change_point_detector(synthetic_data):
    data = synthetic_data.loc["smooth"].droplevel([0, 1])
    temp = ChangePointDetector(w=config.MIN_LENGTH, n_jobs=1).fit_transform(data)  # type: ignore
    assert temp > pd.to_datetime("2025-10-29")  # type: ignore

    data = synthetic_data.loc["erratic"].droplevel([0, 1]).tail(2 * config.MIN_LENGTH - 1)  # type: ignore
    temp = ChangePointDetector(w=config.MIN_LENGTH, n_jobs=1).fit_transform(data)  # type: ignore
    assert temp == data.index.min()[-1]
