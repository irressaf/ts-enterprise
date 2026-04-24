import pandas as pd
from pathlib import Path
import pytest

pytestmark = pytest.mark.data_quality


# Set instance once to run through whole file
@pytest.fixture(scope="module")
def processed_data():
    root = Path(__file__).resolve().parents[1] / "data" / "processed"
    return {
        "sales": pd.read_parquet(root / "sales.pq"),
        "sales_exog_train": pd.read_parquet(root / "sales_exog_train.pq"),
        "sales_exog_test": pd.read_parquet(root / "sales_exog_test.pq"),
        "stores": pd.read_parquet(root / "stores.pq"),
        "transactions": pd.read_parquet(root / "transactions.pq"),
        "oil": pd.read_parquet(root / "oil.pq"),
    }


def test_index(processed_data):
    for name, data in processed_data.items():
        index = data.index.names
        if name != "stores":
            assert index[-1] == "date"

            if data.index.nlevels > 1:
                temp = data.groupby(index[:-1]).apply(
                    lambda x: x.droplevel(index[:-1]).index.inferred_freq
                )
                assert all(temp == "D")
            else:
                assert data.index.inferred_freq == "D"

        assert data.index.has_duplicates is False
    else:
        index = processed_data["sales"].sort_index().droplevel(-1).index.unique()
        assert index.equals(
            processed_data["sales_exog_train"].sort_index().droplevel(-1).index.unique()
        )
        assert index.equals(
            processed_data["sales_exog_test"].sort_index().droplevel(-1).index.unique()
        )

        index = processed_data["stores"].sort_index().index.unique()
        assert index.equals(
            processed_data["sales"].sort_index().index.get_level_values(0).unique()
        )
        assert index.equals(
            processed_data["transactions"]
            .sort_index()
            .index.get_level_values(0)
            .unique()
        )

        assert (
            processed_data["sales_exog_train"]
            .index.intersection(processed_data["sales_exog_test"].index)
            .empty
        )


def test_values(processed_data):
    for name, data in processed_data.items():
        if name in ["sales", "transactions"]:
            data = processed_data[name]
            column, index = data.columns[0], data.index.names
            temp = data.groupby(index[:-1])[column].apply(lambda x: x.isna().mean())
            assert temp.lt(0.10).all()

            temp = (
                data.dropna()
                .groupby(index[:-1])[column]
                .apply(lambda x: x.lt(0).mean())
            )
            assert temp.eq(0).all()
