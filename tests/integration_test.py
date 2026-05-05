from pathlib import Path

import pandas as pd
import pytest

import megatron.config as config

pytestmark = pytest.mark.integration


# Set global config for integration tests placing it inside autocall fixture
@pytest.fixture(autouse=True)
def test_config():
    config.set_config(
        SEASONAL_PERIOD=7,
        FH_SIZE=16,
        MAX_LAG_W_SIZE=30,
        MIN_DATE=pd.to_datetime("2015-01-01"),
        MAX_DATE=pd.to_datetime("2017-12-31"),
        COUNTRY="EC",
    )


from megatron.clusterers import IntermittentLumpyClusterer, SmoothErraticClusterer  # noqa: E402, I001
from megatron.transformers import (  # noqa: E402
    ChangePointDetector,
    ExogenousDataTransformer,
    Mapper,
    OutlierDetector,
    PlateauDetector,
)


@pytest.fixture
def sliced_train_data():
    data_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "test"
        / "integration_test_sliced_train.pq"
    )
    return pd.read_parquet(data_path)


def test_mapper(sliced_train_data):
    mapper = Mapper()
    mapped = mapper.fit_transform(sliced_train_data)
    inv_mapped = mapper.inverse_transform(mapped)

    assert mapped.index.nlevels == 2  # type: ignore
    pd.testing.assert_frame_equal(sliced_train_data, inv_mapped)  # type: ignore


def test_transformers(sliced_train_data):
    data = sliced_train_data.loc["smooth"]
    model = PlateauDetector(
        w=2 * config.SEASONAL_PERIOD,
        value=0,
        truncate=True,
        n_jobs=1,  # type: ignore
    )  # type: ignore
    temp = model.fit_transform(data)
    assert not temp.empty  # type: ignore
    assert model is not None

    model = ChangePointDetector(w=2 * config.MIN_LENGTH, truncate=True, n_jobs=1)  # type: ignore
    temp = model.fit_transform(data)
    assert not temp.empty  # type: ignore
    assert model is not None

    model = OutlierDetector(demand="smooth", truncate=True, n_jobs=1)  # type: ignore
    temp = model.fit_transform(data)
    assert not temp.empty  # type: ignore
    assert model is not None

    data = sliced_train_data.loc["intermittent"]
    model = OutlierDetector(demand="intermittent", truncate=True, n_jobs=1)  # type: ignore
    temp = model.fit_transform(data)
    assert not temp.empty  # type: ignore
    assert model is not None

    model = ExogenousDataTransformer()
    temp = model.fit_transform(sliced_train_data[[]])
    assert temp.index.equals(sliced_train_data.index)  # type: ignore
    assert {
        "month_of_year",
        "day_of_month",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "is_wage",
    } == set(
        temp.columns  # type: ignore
    )


def test_clusterers(sliced_train_data):
    data = sliced_train_data.loc["erratic"].bfill()
    model = SmoothErraticClusterer(n_jobs=1)  # type: ignore
    model.fit(data)
    assert model.labels is not None

    data = sliced_train_data.loc["lumpy"].bfill()
    model = IntermittentLumpyClusterer(n_jobs=1)  # type: ignore
    model.fit(data)
    assert model.labels is not None
