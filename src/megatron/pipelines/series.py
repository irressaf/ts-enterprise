import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

from megatron.transformers.series import (
    PlateauDetector,
    ChangePointDetector,
    OutlierDetector,
    ExogenousDataTransformer,
)
from megatron.transformers.additional import Mapper, DemandClassifier
from megatron.clusterers.series import (
    SmoothErraticClusterer,
    IntermittentLumpyClusterer,
)
from megatron.forecasters.series import CommonForecaster

from pathlib import Path
from joblib import dump, load
import megatron.config as config


class CommonPipeline(BaseForecaster):
    _tags = {
        "y_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "scitype:transform-input": "Dataframe",
        "scitype:transform-output": "Dataframe",
        "capability:missing_values": True,
    }

    def __init__(self, dir_path: Path, value: str, demand: str):
        self.dir_path = dir_path
        self.value = value
        self.demand = demand

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        self.index = y.index.names

        if self.demand in ("smooth", "erratic"):
            # plateau detection
            pld = PlateauDetector(w=2 * config.SEASONAL_PERIOD, value=0, truncate=True)
            y = pld.fit_transform(y)

            # change point detection
            cpd = ChangePointDetector(w=config.MIN_LENGTH, truncate=True)
            y = cpd.fit_transform(y)

        # outliers detection
        if X is not None:
            od = OutlierDetector(
                demand=self.demand, exog_column="on_promotion", truncate=True
            )
            y = od.fit_transform(y.join(X))  # type: ignore
        else:
            od = OutlierDetector(demand=self.demand, truncate=True)
            y = od.fit_transform(y)

        # fill missing values
        if self.demand in ("smooth", "erratic"):
            y = y.groupby(self.index[0]).transform(  # type: ignore
                lambda x: x.interpolate(method="linear").bfill().ffill()
            )
        else:
            y = y.groupby(y.droplevel(-1).index.names).transform(  # type: ignore
                lambda x: x.fillna(
                    x.rolling(window=config.SEASONAL_PERIOD, min_periods=1).median()
                ).fillna(0)
            )

        # exogenous data transformation
        if X is not None:
            self.edt = ExogenousDataTransformer()
            X = self.edt.fit_transform(X)

        print(
            f"{self.value.capitalize()} {self.demand} series successfully transformed!"
        )

        # clustering
        if self.demand in ("smooth", "erratic"):
            self.clusterer = SmoothErraticClusterer(w=90)
        else:
            self.clusterer = IntermittentLumpyClusterer()

        path = self.dir_path / (
            "_".join([self.value, self.demand, self.clusterer.get_tag("object_type")])  # type: ignore
            + ".joblib"
        )

        if not Path.is_file(path):
            self.clusterer.fit(y)
            dump(self.clusterer, path)
        else:
            self.clusterer = load(path)

        y = (
            y.join(
                pd.Series(self.clusterer.labels, name="cluster").rename_axis(
                    self.index[0]
                )
            )
            .set_index("cluster", append=True)
            .reorder_levels(["cluster"] + self.index)
            .sort_index()
        )
        if X is not None:
            X = y[[]].join(X)

        print(f"{self.value.capitalize()} {self.demand} clusterer successfully fitted!")

        self.forecaster = CommonForecaster(
            dir_path=self.dir_path, value=self.value, demand=self.demand
        )
        self.forecaster.fit(y=y, X=X, fh=fh)

        print(
            f"{self.value.capitalize()} {self.demand} forecaster successfully fitted!"
        )

    def _predict(self, fh, X=None):
        if X is not None:
            X = self.edt.transform(X)

            X = (
                X.join(  # type: ignore
                    pd.Series(self.clusterer.labels, name="cluster").rename_axis(
                        self.index[0]
                    ),
                )
                .set_index("cluster", append=True)
                .reorder_levels(["cluster"] + self.index)
                .sort_index()
            )

        return self.forecaster.predict(X=X, fh=fh).droplevel(0)  # type: ignore


class E2EForecaster(BaseForecaster):
    _tags = {
        "y_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "scitype:transform-input": "Dataframe",
        "scitype:transform-output": "Dataframe",
        "capability:missing_values": True,
    }

    def __init__(self, dir_path=Path.cwd().parent / "models"):
        self.dir_path = dir_path
        Path.mkdir(self.dir_path, exist_ok=True)
        self.models = {}

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        self.value, self.index = y.columns[0], y.droplevel(-1).index.names
        fh = ForecastingHorizon(
            values=[*range(1, config.FH_SIZE + 1)], is_relative=True, freq="D"
        )

        self.int_items = (
            y.groupby(self.index)[self.value]
            .apply(lambda x: (x.dropna() % 1).eq(0).all())
            .to_dict()
        )

        y = y.sort_index()
        if X is not None:
            X = X.sort_index()

        self.mapper = Mapper()
        y = self.mapper.fit_transform(y)
        if X is not None:
            X = self.mapper.transform(X)

        self.demand_class = DemandClassifier().fit_transform(y)["class"]  # type: ignore
        y = (
            y.join(self.demand_class)  # type: ignore
            .set_index("class", append=True)
            .reorder_levels(["class"] + y.index.names)  # type: ignore
        )
        if X is not None:
            X = y[[]].join(X)

        for demand in y.index.get_level_values(0).unique():
            self.models[demand] = CommonPipeline(
                dir_path=self.dir_path, value=self.value, demand=demand
            )

        for demand in self.models:
            self.models[demand].fit(
                y=y.loc[demand], X=X if X is None else X.loc[demand], fh=fh
            )
        else:
            return self

    def _predict(self, fh, X=None):
        fh = ForecastingHorizon(
            values=[*range(1, config.FH_SIZE + 1)], is_relative=True, freq="D"
        )

        if X is not None:
            X = self.mapper.transform(X.sort_index())
            X = (
                X.join(self.demand_class)  # type: ignore
                .set_index("class", append=True)
                .reorder_levels(["class"] + X.index.names)  # type: ignore
            )

        temp = self.mapper.inverse_transform(
            pd.concat(
                [
                    self.models[demand].predict(
                        X=X.loc[demand] if X is not None else X, fh=fh
                    )
                    for demand in self.models
                ]
            ).sort_index()
        )
        temp[self.value] = [
            round(x) if self.int_items[y] else x
            for x, y in zip(temp[self.value], temp.droplevel(-1).index)
        ]
        return temp
