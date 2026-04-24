import pandas as pd
from pathlib import Path
from time import perf_counter
import megatron.config as config


def formatter(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def logs(name, start):
    text = f"{name} passed in {formatter(perf_counter() - start)}"
    print(f"\033[92m{text}\033[0m")

start = perf_counter()  # timer

root = Path(__file__).resolve().parents[1] / "data" / "raw"
test = pd.read_csv(
    filepath_or_buffer=root / "test.csv",
    parse_dates=["date"],
    index_col=["id"],
    engine="pyarrow",
)

max_date = test["date"].max()
min_date = max_date - pd.DateOffset(years=2)

# Set global config variables for running session
config.set_config(
    SEASONAL_PERIOD=7,
    MAX_LAG_W_SIZE=30,
    COUNTRY="EC",
    MIN_DATE=min_date,
    MAX_DATE=max_date,
    FH_SIZE=(max_date - test["date"].min()).days + 1,
)

from megatron.transformers import Mapper, InitialPreprocessing, PlateauDetector

# Oil data
oil = (
    pd.read_csv(
        filepath_or_buffer=root / "oil.csv",
        index_col=["date"],
        parse_dates=["date"],
    )
    .asfreq("D")
    .loc[min_date:]
    .interpolate(method="linear")
    .rename(columns={"dcoilwtico": "oil"})
)

# Stores data
stores = (
    pd.read_csv(root / "stores.csv", index_col=["store_nbr"])
    .drop(columns=["city", "state"])
    .astype({"cluster": float})
    .rename(columns={"cluster": "store_group"})
)
stores["type"] = stores["type"].factorize()[0].astype(float)

logs("Raw oil and stores data loading -", start)
stage = perf_counter()  # timer

# Transactions data
transactions = (
    pd.read_csv(
        filepath_or_buffer=root / "transactions.csv",
        index_col=["store_nbr", "date"],
        parse_dates=["date"],
        engine="pyarrow",
    )
    .groupby("store_nbr")
    .apply(lambda x: x.droplevel(["store_nbr"]).asfreq("D").loc[min_date:])
    .sort_index()
)

processing = InitialPreprocessing(w=2 * config.SEASONAL_PERIOD)  # type: ignore
transactions = processing.drop_zero_series(transactions)
transactions = processing.trim_leading_zeros(transactions)
transactions = processing.drop_trailing_zero_window_series(transactions)
transactions = PlateauDetector(
    w=2 * config.SEASONAL_PERIOD, truncate=True  # type: ignore
).fit_transform(transactions)

logs("Raw transactions data loading and processing -", stage)
stage = perf_counter()  # timer

# Sales data
sales = (
    pd.read_csv(
        filepath_or_buffer=root / "train.csv",
        index_col=["store_nbr", "family", "date"],
        parse_dates=["date"],
        engine="pyarrow",
    )
    .groupby(["store_nbr", "family"])
    .apply(lambda x: x.droplevel(["store_nbr", "family"]).asfreq("D").loc[min_date:])
    .sort_index()
)
X_exog_train = (
    sales[["onpromotion"]]
    .fillna(0)
    .astype(float)
    .rename(columns={"onpromotion": "on_promotion"})
)
X_exog_test = (
    test.set_index(["store_nbr", "family", "date"])
    .astype(float)
    .rename(columns={"onpromotion": "on_promotion"})
    .sort_index()
)
sales = sales[["sales"]]

mapper = Mapper()
sales = mapper.fit_transform(sales)
processing = InitialPreprocessing(w=2 * config.SEASONAL_PERIOD)  # type: ignore
sales = processing.drop_zero_series(sales)  # type: ignore
sales = processing.trim_leading_zeros(sales)
sales = processing.drop_trailing_zero_window_series(sales)
sales = mapper.inverse_transform(sales)

sales = sales.join(transactions[[]], how="inner")  # type: ignore
X_exog_train = X_exog_train.join(sales[[]], how="inner")
X_exog_test = X_exog_test.join(
    sales[~sales.droplevel(-1).index.duplicated(keep="first")][[]].droplevel(-1),
    how="inner",
)

logs("Raw sales data loading and processing -", stage)
stage = perf_counter()  # timer

# Save processed data
root = Path(__file__).resolve().parents[1] / "data" / "processed"
root.mkdir(exist_ok=True)

oil.to_parquet(root / "oil.pq")
stores.to_parquet(root / "stores.pq")
transactions.to_parquet(root / "transactions.pq")  # type: ignore
sales.to_parquet(root / "sales.pq")
X_exog_train.to_parquet(root / "sales_exog_train.pq")
X_exog_test.to_parquet(root / "sales_exog_test.pq")

logs("Total work -", start)
