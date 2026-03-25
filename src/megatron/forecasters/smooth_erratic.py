import numpy as np
import pandas as pd
from itertools import product

from optuna.samplers import TPESampler
from optuna.distributions import (
    IntDistribution,
    FloatDistribution,
    CategoricalDistribution,
)

from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.statsforecast import StatsForecastAutoTheta

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_log_error
from sklearn.preprocessing import TargetEncoder

from sktime.performance_metrics.forecasting import make_forecasting_scorer
from sktime.split import ExpandingGreedySplitter
from sktime.forecasting.model_selection import ForecastingOptunaSearchCV
from sktime.forecasting.compose import (
    make_reduction,
    MultiplexForecaster,
    FallbackForecaster,
)
from sktime.transformations.series.summarize import WindowSummarizer
from sktime.forecasting.base import BaseForecaster

from pathlib import Path
from joblib import Parallel, delayed, dump, load
from tqdm_joblib import ParallelPbar
import warnings
import megatron.config as config

warnings.filterwarnings("ignore")

rmsle = make_forecasting_scorer(root_mean_squared_log_error, name="RMSLE")
cv = ExpandingGreedySplitter(test_size=config.FH_SIZE, folds=1)


class GlobalModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, enable_target_encoding=False, enable_weights=False):
        self.estimator = estimator
        self.enable_target_encoding = enable_target_encoding
        self.enable_weights = enable_weights

        super().__init__()

    def _hyperbolic(self, x):
        return (1 / x.shape[0] ** 0.5) * (1 / (1 - np.linspace(0.1, 0.99, x.shape[0])))

    def fit(self, X, y, **kwargs):
        self.estimator_, weights = clone(self.estimator), None

        X = X.assign(target=y).dropna()
        if self.enable_weights:
            weights = np.concatenate(
                X.groupby(X.index.names[:-1])
                .apply(lambda x: self._hyperbolic(x[[]]))
                .tolist()
            )

        s = X.apply(lambda x: x.value_counts(normalize=True).max())
        X = X[s[s.lt(0.99)].index]

        if self.enable_target_encoding:
            s = X.drop(columns=["target"]).nunique()
            self.c_features = list(s[s.between(3, 31)].index)

            self.encoder = TargetEncoder(smooth=1, random_state=config.SEED)
            X[self.c_features] = self.encoder.fit_transform(
                X[self.c_features], X["target"]
            )

            self.s_features = list(s[s.gt(31)].index) + self.c_features
            self.scaler = StandardScaler()
            X[self.s_features] = self.scaler.fit_transform(X[self.s_features])

        y = X["target"]
        X = X.drop(columns=["target"])
        self.features_ = list(X.columns)
        return self.estimator_.fit(X=X, y=y, sample_weight=weights, **kwargs)

    def predict(self, X):
        X = X[self.features_]

        if self.enable_target_encoding:
            X[self.c_features] = self.encoder.transform(X[self.c_features])
            X[self.s_features] = self.scaler.transform(X[self.s_features])
        return self.estimator_.predict(X).clip(min=0)


complex_global = ForecastingOptunaSearchCV(
    forecaster=make_reduction(
        GlobalModelWrapper(LGBMRegressor(subsample_freq=1, n_jobs=1, verbose=-1)),
        transformers=[
            WindowSummarizer(
                lag_feature={
                    "lag": [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        config.SEASONAL_PERIOD,
                        2 * config.SEASONAL_PERIOD,
                        3 * config.SEASONAL_PERIOD,
                        4 * config.SEASONAL_PERIOD,
                    ],
                    "mean": [
                        [1, config.SEASONAL_PERIOD],
                        [1, 2 * config.SEASONAL_PERIOD],
                        [1, 4 * config.SEASONAL_PERIOD],
                    ],
                    "std": [
                        [1, config.SEASONAL_PERIOD],
                        [1, 2 * config.SEASONAL_PERIOD],
                        [1, 4 * config.SEASONAL_PERIOD],
                    ],
                    "sum": [
                        [1, config.SEASONAL_PERIOD],
                        [1, 2 * config.SEASONAL_PERIOD],
                    ],
                    "max": [
                        [1, config.SEASONAL_PERIOD],
                        [1, 2 * config.SEASONAL_PERIOD],
                    ],
                    "min": [
                        [1, config.SEASONAL_PERIOD],
                        [1, 2 * config.SEASONAL_PERIOD],
                    ],
                },
                n_jobs=1,
            )
        ],
        window_length=None,  # type: ignore
        pooling="global",
    ),
    cv=cv,
    param_grid={
        "estimator__enable_target_encoding": CategoricalDistribution([True, False]),
        "estimator__enable_weights": CategoricalDistribution([True, False]),
        "estimator__estimator__boosting_type": CategoricalDistribution(
            ["gbdt", "dart", "rf"]
        ),
        "estimator__estimator__n_estimators": IntDistribution(100, 1000),
        "estimator__estimator__learning_rate": FloatDistribution(0.005, 0.5),
        "estimator__estimator__max_depth": IntDistribution(2, 7),
        "estimator__estimator__subsample": FloatDistribution(0.6, 1),
        "estimator__estimator__colsample_bytree": FloatDistribution(0.5, 1),
        "estimator__estimator__reg_alpha": FloatDistribution(0, 10),
        "estimator__estimator__reg_lambda": FloatDistribution(0, 10),
    },
    scoring=rmsle,
    error_score="raise",  # type: ignore
    sampler=TPESampler(seed=config.SEED),
    verbose=-1,
)

simplex_global = ForecastingOptunaSearchCV(
    forecaster=make_reduction(
        GlobalModelWrapper(ElasticNet(), enable_target_encoding=True),
        transformers=[
            WindowSummarizer(
                lag_feature={
                    "lag": [
                        1,
                        config.SEASONAL_PERIOD,
                        2 * config.SEASONAL_PERIOD,
                        4 * config.SEASONAL_PERIOD,
                    ],
                    "mean": [[1, 4 * config.SEASONAL_PERIOD]],
                    "std": [[1, 4 * config.SEASONAL_PERIOD]],
                    "sum": [[1, config.SEASONAL_PERIOD]],
                },
                n_jobs=1,
            )
        ],
        window_length=None,  # type: ignore
        pooling="global",
    ),
    cv=cv,
    param_grid={
        "estimator__estimator__alpha": FloatDistribution(0.01, 15),
        "estimator__estimator__l1_ratio": FloatDistribution(0.01, 1),
    },
    scoring=rmsle,
    error_score="raise",  # type: ignore
    sampler=TPESampler(seed=config.SEED),
    verbose=-1,
)


class LocalModelWrapper(BaseForecaster):
    def __init__(self, estimator, whether_to_use_X=False, drop_holiday_flag=False):
        self.estimator = estimator
        self.whether_to_use_X = whether_to_use_X
        self.drop_holiday_flag = drop_holiday_flag

        super().__init__()

    def _fit(self, y, X=None, fh=None, **kwargs):
        self.estimator_ = clone(self.estimator)

        if self.whether_to_use_X and X is not None:
            X = X.assign(target=y)
            if self.drop_holiday_flag:
                X = X.drop(columns=["is_holiday"])

            s = X.apply(lambda x: x.value_counts(normalize=True).max())
            X = X[s[s.lt(0.99)].index]

            s = X.drop(columns=["target"]).nunique()
            self.c_features = list(s[s.between(3, 31)].index)

            self.encoder = TargetEncoder(smooth=1, random_state=config.SEED)
            X[self.c_features] = self.encoder.fit_transform(
                X[self.c_features], X["target"]
            )

            self.s_features = list(s[s.gt(31)].index) + self.c_features
            self.scaler = StandardScaler()
            X[self.s_features] = self.scaler.fit_transform(X[self.s_features])

            X = X.drop(columns=["target"]).asfreq("D")
            self.features_ = list(X.columns)
        return self.estimator_.fit(
            X=X if self.whether_to_use_X else None,
            y=y.asfreq("D"),
            fh=fh,
            **kwargs,
        )

    def _predict(self, fh, X=None):
        if self.whether_to_use_X and X is not None:
            X = X[self.features_]
            X[self.c_features] = self.encoder.transform(X[self.c_features])
            X[self.s_features] = self.scaler.transform(X[self.s_features])
        return self.estimator_.predict(
            fh=fh, X=X if self.whether_to_use_X else None
        ).clip(lower=0)


complex_local = ForecastingOptunaSearchCV(
    forecaster=MultiplexForecaster(
        forecasters=[
            (
                "fallback1",
                FallbackForecaster(
                    [
                        (
                            "prophet",
                            LocalModelWrapper(
                                estimator=Prophet(
                                    add_country_holidays={
                                        "country_name": config.COUNTRY
                                    },
                                    weekly_seasonality=True,  # type: ignore
                                    verbose=-1,
                                ),
                                drop_holiday_flag=True,
                            ),
                        ),
                        (
                            "theta",
                            LocalModelWrapper(
                                estimator=StatsForecastAutoTheta(
                                    season_length=config.SEASONAL_PERIOD
                                ),
                                whether_to_use_X=False,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "fallback2",
                FallbackForecaster(
                    [
                        ("sarimax", LocalModelWrapper(estimator=SARIMAX(disp=False))),
                        (
                            "theta",
                            LocalModelWrapper(
                                estimator=StatsForecastAutoTheta(
                                    season_length=config.SEASONAL_PERIOD
                                ),
                                whether_to_use_X=False,
                            ),
                        ),
                    ]
                ),
            ),
        ]
    ),
    cv=cv,
    param_grid={
        "selected_forecaster": CategoricalDistribution(["fallback1", "fallback2"]),
        "fallback1__prophet__whether_to_use_X": CategoricalDistribution([True, False]),
        "fallback2__sarimax__whether_to_use_X": CategoricalDistribution([True, False]),
        "fallback1__prophet__estimator__n_changepoints": IntDistribution(1, 25),
        "fallback1__prophet__estimator__seasonality_prior_scale": FloatDistribution(
            0.001, 10, log=True
        ),
        "fallback1__prophet__estimator__holidays_prior_scale": FloatDistribution(
            0.001, 10, log=True
        ),
        "fallback1__prophet__estimator__changepoint_prior_scale": FloatDistribution(
            0.001, 10, log=True
        ),
        "fallback1__prophet__estimator__seasonality_mode": CategoricalDistribution(
            ["additive", "multiplicative"]
        ),
        "fallback2__sarimax__estimator__order": CategoricalDistribution(
            [*product([*range(3)], [*range(2)], [*range(3)])]  # type: ignore
        ),
        "fallback2__sarimax__estimator__seasonal_order": CategoricalDistribution(
            [*product([*range(3)], [*range(2)], [*range(3)], [config.SEASONAL_PERIOD])]  # type: ignore
        ),
        "fallback2__sarimax__estimator__trend": CategoricalDistribution(["c", "ct"]),
        "fallback1__theta__estimator__decomposition_type": CategoricalDistribution(
            ["additive", "multiplicative"]
        ),
        "fallback2__theta__estimator__decomposition_type": CategoricalDistribution(
            ["additive", "multiplicative"]
        ),
        "fallback1__theta__estimator__model": CategoricalDistribution(
            ["STM", "OTM", "DSTM", "DOTM"]
        ),
        "fallback2__theta__estimator__model": CategoricalDistribution(
            ["STM", "OTM", "DSTM", "DOTM"]
        ),
    },
    scoring=rmsle,
    error_score="raise",  # type: ignore
    sampler=TPESampler(seed=config.SEED),
    verbose=-1,
)


class SmoothErraticForecaster(BaseForecaster):
    _tags = {
        "y_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "scitype:transform-input": "Dataframe",
        "scitype:transform-output": "Dataframe",
    }

    def __init__(self, dir_path: Path, value: str, demand: str, n_jobs=-1):
        self.dir_path = dir_path
        self.value = value
        self.demand = demand
        self.n_jobs = n_jobs
        self.pipelines = {
            "complex_global": complex_global,
            "simplex_global": simplex_global,
            "complex_local": complex_local,
        }
        self.models = {}

        super().__init__()

    def _fit_per_instance(self, index, fh):
        import warnings, cmdstanpy, logging
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        logging.getLogger("statsmodels").setLevel(logging.ERROR)
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.ERROR)
        warnings.simplefilter("ignore", ConvergenceWarning)
        cmdstanpy.disable_logging()

        model = clone(self.pipelines[self.models[index]])  # type: ignore
        path = self.dir_path / (
            "_".join(
                [
                    self.value,
                    self.demand,
                    model.get_tag("object_type"),
                    "cluster",
                    "_".join([str(x) for x in index]),
                ]
            )
            + ".joblib"
        )

        if not Path.is_file(path):
            if len(index) == 1:
                y_temp = self.y_.loc[index]
                X_temp = None if self.X_ is None else self.X_.loc[index]
            else:
                y_temp = self.y_.loc[index[0]].loc[[index[-1]]]
                X_temp = (
                    None if self.X_ is None else self.X_.loc[index[0]].loc[[index[-1]]]
                )

            model.fit(y=y_temp, X=X_temp, fh=fh)
            dump({"model": model.best_forecaster_, "score": model.best_score_}, path)
            print(
                f"{path.relative_to(self.dir_path)} best {rmsle.name} score: {round(model.best_score_, 3)}"
            )

        return index, path

    def _fit(self, y, X=None, fh=None):
        self.y_, self.X_ = y, X

        for cluster in self.y_.index.get_level_values(0).unique():
            if self.y_.loc[cluster].shape[0] >= 1e4:
                self.models[tuple([cluster])] = "complex_global"
            elif self.y_.loc[cluster].shape[0] >= 1e3:
                self.models[tuple([cluster])] = "simplex_global"
            else:
                for item in self.y_.loc[cluster].index.get_level_values(0).unique():
                    self.models[tuple([cluster, item])] = "complex_local"

        temp = ParallelPbar(
            desc=f"Successfully fitted {self.value} {self.demand} models",
        )(n_jobs=self.n_jobs)(
            delayed(self._fit_per_instance)(index, fh) for index in self.models
        )

        for index, path in temp:  # type: ignore
            self.models[index] = path

        self.mapper = self.y_[~self.y_.droplevel(-1).index.duplicated(keep="first")][
            []
        ].droplevel(-1)
        del self.y_, self.X_
        return self

    def _predict_per_instance(self, index, fh):
        obj = load(self.models[index])
        if len(index) == 1:
            X_temp = None if self.X_ is None else self.X_.loc[index]
        else:
            X_temp = None if self.X_ is None else self.X_.loc[index[0]].loc[[index[-1]]]
        return obj["model"].predict(X=X_temp, fh=fh)

    def _predict(self, fh, X=None):
        self.X_ = X

        temp = pd.concat(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._predict_per_instance)(index, fh) for index in self.models
            )
        )
        del self.X_

        return self.mapper.join(temp)
