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

# Load processed data
root = Path(__file__).resolve().parents[1] / "data" / "processed"
oil, stores, transactions, sales, X_exog_train, X_exog_test = (
    pd.read_parquet(root / "oil.pq"),
    pd.read_parquet(root / "stores.pq"),
    pd.read_parquet(root / "transactions.pq"),
    pd.read_parquet(root / "sales.pq"),
    pd.read_parquet(root / "sales_exog_train.pq"),
    pd.read_parquet(root / "sales_exog_test.pq"),
)

# Set global config variables for running session
config.set_config(
    SEASONAL_PERIOD=7,
    MAX_LAG_W_SIZE=30,
    COUNTRY="EC",
    MIN_DATE=oil.index.min(),
    MAX_DATE=oil.index.max(),
    FH_SIZE=X_exog_test.index.get_level_values(-1).nunique(),
)

from megatron.pipelines import E2EForecaster

# Transactions forecast
X_temp_train = X_exog_train.groupby(["store_nbr", "date"]).sum().join(oil).join(stores)
X_temp_test = X_exog_test.groupby(["store_nbr", "date"]).sum().join(oil).join(stores)

logs("Processed data loading -", start)
stage = perf_counter()  # timer

root = Path(__file__).resolve().parents[1] / "models"
model = E2EForecaster(dir_path=root)
model.fit(y=transactions, X=X_temp_train, fh=config.FH_SIZE)  # type: ignore

logs("Transactions data training -", stage)
stage = perf_counter()  # timer

forecasts = model.predict(X=X_temp_test)
transactions = pd.concat(
    [
        transactions.interpolate(method="linear").bfill().ffill(),
        forecasts,  # type: ignore
    ]
)

logs("Transactions data forecasting -", stage)
stage = perf_counter()  # timer

# Sales forecast
X_exog_train = X_exog_train.join(oil).join(stores).join(transactions)
X_exog_test = X_exog_test.join(oil).join(stores).join(transactions)

root = Path(__file__).resolve().parents[1] / "models"
model = E2EForecaster(dir_path=root)
model.fit(y=sales, X=X_exog_train, fh=config.FH_SIZE)  # type: ignore

logs("Sales data training -", stage)
stage = perf_counter()  # timer

forecasts = model.predict(X=X_exog_test)

logs("Sales data forecasting -", stage)
stage = perf_counter()  # timer

# Prepare submission
root = Path(__file__).resolve().parents[1] / "data"
test = pd.read_csv(
    filepath_or_buffer=root / "raw" / "test.csv",
    parse_dates=["date"],
    index_col=["id", "store_nbr", "family", "date"],
    engine="pyarrow",
)[[]]
submission = pd.read_csv(
    filepath_or_buffer=root / "raw" / "sample_submission.csv",
    index_col="id",
    engine="pyarrow",
)
column = forecasts.columns[0]  # type: ignore
submission = (
    submission[[]]
    .join(test.join(forecasts[column]).fillna(0))  # type: ignore
    .reset_index()[["id", column]]
)
root = Path(__file__).resolve().parents[1] / "data" / "submissions"
root.mkdir(exist_ok=True)
submission.to_csv(root / "latest.csv", index=False)

logs("Total work -", start)
