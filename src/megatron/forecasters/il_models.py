import numpy as np
from itertools import product

from optuna.samplers import TPESampler
from optuna.distributions import (
    IntDistribution,
    FloatDistribution,
    CategoricalDistribution,
)

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import ElasticNet


from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.metrics import root_mean_squared_log_error

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

import megatron.config as config


rmsle = make_forecasting_scorer(root_mean_squared_log_error, name="RMSLE")
cv = ExpandingGreedySplitter(test_size=config.FH_SIZE, folds=1)


class GlobalModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        classifier,
        regressor,
        enable_target_encoding=False,
        enable_weights=False,
    ):
        self.classifier = classifier
        self.regressor = regressor
        self.enable_target_encoding = enable_target_encoding
        self.enable_weights = enable_weights

        super().__init__()

    def _hyperbolic(self, x):
        return (1 / x.shape[0] ** 0.5) * (1 / (1 - np.linspace(0.1, 0.99, x.shape[0])))

    def fit(self, X, y, **kwargs):
        self.classifier_ = clone(self.classifier)
        self.regressor_ = clone(self.regressor)
        weights = None

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

            self.encoder = TargetEncoder(
                smooth=1, random_state=config.SEED, target_type="continuous"
            )
            X[self.c_features] = self.encoder.fit_transform(
                X[self.c_features], X["target"]
            )

            self.s_features = list(s[s.gt(31)].index) + self.c_features
            self.scaler = StandardScaler()
            X[self.s_features] = self.scaler.fit_transform(X[self.s_features])

        y = X["target"]
        X = X.drop(columns=["target"])
        self.features_ = list(X.columns)

        self.X_train, self.y_train = X, y

        self.classifier_.fit(
            X=X, y=y.gt(0).astype(int), sample_weight=weights, **kwargs
        )
        self.regressor_.fit(X=X[y.gt(0)], y=y[y.gt(0)], sample_weight=weights, **kwargs)

        return self

    def predict(self, X):
        X = X[self.features_]

        if self.enable_target_encoding:
            X[self.c_features] = self.encoder.transform(X[self.c_features])
            X[self.s_features] = self.scaler.transform(X[self.s_features])

        self.X_test = X

        p = self.classifier_.predict_proba(X)[:, 1]
        v = self.regressor_.predict(X).clip(min=0)
        return p * v


il_complex_global = ForecastingOptunaSearchCV(
    forecaster=make_reduction(
        estimator=GlobalModelWrapper(
            classifier=LGBMClassifier(
                objective="binary",
                n_jobs=1,
                verbose=-1,
                random_state=config.SEED,
            ),
            regressor=LGBMRegressor(
                objective="tweedie",
                n_jobs=1,
                verbose=-1,
                random_state=config.SEED,
            ),
        ),
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
        "estimator__min_positive_samples": IntDistribution(10, 50),
        "estimator__classifier__boosting_type": CategoricalDistribution(
            ["gbdt", "dart", "rf"]
        ),
        "estimator__classifier__n_estimators": IntDistribution(100, 700),
        "estimator__classifier__learning_rate": FloatDistribution(0.01, 0.15, log=True),
        "estimator__classifier__num_leaves": IntDistribution(15, 63),
        "estimator__classifier__max_depth": IntDistribution(3, 8),
        "estimator__classifier__min_child_samples": IntDistribution(10, 100),
        "estimator__classifier__subsample": FloatDistribution(0.6, 1.0),
        "estimator__classifier__colsample_bytree": FloatDistribution(0.5, 1.0),
        "estimator__classifier__reg_alpha": FloatDistribution(0.0, 10.0),
        "estimator__classifier__reg_lambda": FloatDistribution(0.0, 10.0),
        "estimator__regressor__objective": CategoricalDistribution(
            ["tweedie", "regression_l1"]
        ),
        "estimator__regressor__tweedie_variance_power": FloatDistribution(1.1, 1.7),
        "estimator__regressor__boosting_type": CategoricalDistribution(
            ["gbdt", "dart", "rf"]
        ),
        "estimator__regressor__n_estimators": IntDistribution(100, 900),
        "estimator__regressor__learning_rate": FloatDistribution(0.005, 0.15, log=True),
        "estimator__regressor__num_leaves": IntDistribution(15, 63),
        "estimator__regressor__max_depth": IntDistribution(3, 8),
        "estimator__regressor__min_child_samples": IntDistribution(5, 50),
        "estimator__regressor__subsample": FloatDistribution(0.6, 1.0),
        "estimator__regressor__colsample_bytree": FloatDistribution(0.5, 1.0),
        "estimator__regressor__reg_alpha": FloatDistribution(0.0, 10.0),
        "estimator__regressor__reg_lambda": FloatDistribution(0.0, 10.0),
    },
    scoring=rmsle,
    error_score="raise",  # type: ignore
    sampler=TPESampler(seed=config.SEED),
    verbose=-1,
)

il_simplex_global = None

il_complex_local = None
