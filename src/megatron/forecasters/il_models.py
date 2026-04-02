import numpy as np
import pandas as pd
from feature_engine.selection import SmartCorrelatedSelection

from optuna.samplers import TPESampler
from optuna.distributions import (
    IntDistribution,
    FloatDistribution,
    CategoricalDistribution,
)

from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import TargetEncoder

from sktime.forecasting.model_selection import ForecastingOptunaSearchCV
from sktime.forecasting.compose import MultiplexForecaster
from sktime.transformations.series.summarize import WindowSummarizer
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.croston import Croston
from sktime.forecasting.statsforecast import StatsForecastADIDA
from sktime.forecasting.tsb import TSB

from megatron.transformers.additional import (
    b_mad,
    b_median,
    c_occ_last,
    c_occ_rate,
    c_occ_count,
    c_non_occ_head,
    c_non_occ_tail,
    r_pos_last,
    r_pos_mean,
    r_pos_std,
    r_pos_sum,
)
from megatron.forecasters.se_models import scoring, cv
import megatron.config as config


class HurdleModel(BaseEstimator, RegressorMixin):
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

        super().__init__()

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.classifier_ = clone(self.classifier)
        self.regressor_ = clone(self.regressor)
        y = y[y.columns[0]]

        cols = [
            x
            for x in X.columns
            if not any(y in x for y in ["_lag_", "_b_", "_c_", "_r_"])
        ]
        s = X[cols].nunique()
        self.cat_features = list(s[s.between(3, 31)].index)

        if self.cat_features:
            self.encoder = TargetEncoder(
                smooth=1, random_state=config.SEED, target_type="continuous"
            )
            X[self.cat_features] = self.encoder.fit_transform(X[self.cat_features], y)

        temp = X[[*filter(lambda x: not x.count("_r_"), X.columns)]]
        fs = SmartCorrelatedSelection(
            threshold=0.5,
            selection_method="model_performance",
            estimator=LogisticRegression(C=np.inf),
        )
        self.c_features = list(
            fs.fit_transform(X=temp, y=y.gt(0).astype(int)).columns  # type: ignore
        )
        self.classifier_.fit(
            X=temp[self.c_features],
            y=y.gt(0).astype(int),
            sample_weight=sample_weight,
            **kwargs,
        )

        temp = X[[*filter(lambda x: not x.count("_c_"), X.columns)]][y.gt(0).values]
        fs = SmartCorrelatedSelection(
            threshold=0.5,
            selection_method="model_performance",
            estimator=ElasticNet(alpha=0),
            scoring="r2",
        )
        self.r_features = list(
            fs.fit_transform(X=temp, y=y[y.gt(0)]).columns  # type: ignore
        )
        self.regressor_.fit(X=temp[self.r_features], y=y[y.gt(0)], **kwargs)
        return self

    def predict(self, X):
        if self.cat_features:
            X[self.cat_features] = self.encoder.transform(X[self.cat_features])

        p = self.classifier_.predict_proba(X[self.c_features])[:, 1]
        v = self.regressor_.predict(X[self.r_features]).clip(min=0)
        return p * v


class DirectGlobalForecaster(BaseForecaster):
    _tags = {
        "y_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "scitype:transform-input": "Dataframe",
        "scitype:transform-output": "Dataframe",
        "requires-fh-in-fit": True,
    }

    def __init__(self, estimator, enable_weights=False):
        self.estimator = estimator
        self.enable_weights = enable_weights
        self.summarizer = WindowSummarizer(
            lag_feature={
                "lag": [1, 2, 3, config.SEASONAL_PERIOD],
                b_median: [[1, config.SEASONAL_PERIOD]],
                b_mad: [[1, config.SEASONAL_PERIOD]],
                c_occ_last: [[1, 1]],
                c_occ_rate: [[1, config.SEASONAL_PERIOD]],
                c_occ_count: [[1, config.SEASONAL_PERIOD]],
                c_non_occ_head: [[1, config.SEASONAL_PERIOD]],
                c_non_occ_tail: [[1, config.SEASONAL_PERIOD]],
                r_pos_last: [[1, config.SEASONAL_PERIOD]],
                r_pos_mean: [[1, config.SEASONAL_PERIOD]],
                r_pos_std: [[1, config.SEASONAL_PERIOD]],
                r_pos_sum: [[1, config.SEASONAL_PERIOD]],
            },
            n_jobs=1,
        )

        super().__init__()

    def _linear(self, x):
        return (1 / x.shape[0] ** 0.5) * np.linspace(0.1, 0.99, x.shape[0])

    def _fit(self, y, X=None, fh=None):
        self.estimators_, self.index, self.value = [], y.index.names, y.columns[0]
        X_lag = self.summarizer.fit_transform(y).dropna()  # type: ignore
        s = X_lag.apply(lambda x: x.value_counts(normalize=True).max())
        X_lag = X_lag.drop(columns=s[s.ge(0.99)].index)

        y = X_lag[[]].join(y)
        if X is not None:
            X = X_lag[[]].join(X)

        if fh is not None:
            for x in fh.to_numpy():
                y_temp = y.groupby(self.index[0])[y.columns].shift(-x).dropna()
                if X is not None:
                    X_temp = (
                        X.groupby(self.index[0])[X.columns]
                        .shift(-x)
                        .dropna()
                        .join(X_lag)
                    )
                else:
                    X_temp = y_temp[[]].join(X_lag)

                if self.enable_weights:
                    weights = np.concatenate(
                        y_temp.groupby(y_temp.droplevel(-1).index.names)
                        .apply(lambda x: self._linear(x[[]]))
                        .tolist()
                    )
                else:
                    weights = None

                estimator = clone(self.estimator)
                estimator.fit(y=y_temp, X=X_temp, sample_weight=weights)
                self.estimators_ += [estimator]
            else:
                self.X_last_lag = (
                    X_lag.groupby(self.index[0])[X_lag.columns].tail(1).droplevel(-1)
                )
                self.fh_test_index = pd.date_range(
                    start=y.index.get_level_values(-1).max(),
                    periods=config.FH_SIZE + 1,
                    freq="D",
                    inclusive="right",
                )

    def _predict(self, fh, X=None):
        forecasts = []
        if X is not None:
            X = X.join(self.X_last_lag).reorder_levels(self.index[::-1])

        for i, date in enumerate(self.fh_test_index):
            y_temp = self.X_last_lag[[]]
            y_temp[self.index[-1]] = date
            y_temp = y_temp.set_index(self.index[-1], append=True)
            X_temp = X.loc[date] if X is not None else self.X_last_lag
            y_temp[self.value] = self.estimators_[i].predict(X_temp).clip(min=0)
            forecasts += [y_temp.copy()]
        else:
            return pd.concat(forecasts).sort_index()


il_complex_global = ForecastingOptunaSearchCV(
    forecaster=DirectGlobalForecaster(
        estimator=LGBMRegressor(subsample_freq=1, n_jobs=1, verbose=-1)
    ),
    cv=cv,
    param_grid={
        "enable_weights": CategoricalDistribution([False, True]),
        "estimator__objective": CategoricalDistribution(
            ["tweedie", "regression", "regression_l1", "huber", "fair"]
        ),
        "estimator__tweedie_variance_power": FloatDistribution(1, 2),
        "estimator__boosting_type": CategoricalDistribution(["gbdt", "dart", "rf"]),
        "estimator__n_estimators": IntDistribution(100, 900),
        "estimator__learning_rate": FloatDistribution(0.005, 0.3),
        "estimator__max_depth": IntDistribution(2, 7),
        "estimator__subsample": FloatDistribution(0.6, 1.0),
        "estimator__colsample_bytree": FloatDistribution(0.5, 1.0),
        "estimator__reg_alpha": FloatDistribution(0.0, 10.0),
        "estimator__reg_lambda": FloatDistribution(0.0, 10.0),
    },
    scoring=scoring,
    sampler=TPESampler(seed=config.SEED),
    verbose=-1,
)

il_simplex_global = ForecastingOptunaSearchCV(
    forecaster=DirectGlobalForecaster(
        estimator=HurdleModel(classifier=LogisticRegression(), regressor=ElasticNet())
    ),
    cv=cv,
    param_grid={
        "enable_weights": CategoricalDistribution([False, True]),
        "estimator__classifier__C": FloatDistribution(0.01, 10),
        "estimator__classifier__l1_ratio": FloatDistribution(0, 1),
        "estimator__regressor__alpha": FloatDistribution(0.01, 10),
        "estimator__regressor__l1_ratio": FloatDistribution(0, 1),
    },
    scoring=scoring,
    sampler=TPESampler(seed=config.SEED),
    verbose=-1,
)

il_simplex_local = ForecastingOptunaSearchCV(
    forecaster=MultiplexForecaster(
        forecasters=[
            ("croston", Croston()),
            ("adida", StatsForecastADIDA()),
            ("tsb", TSB()),
        ]
    ),
    cv=cv,
    param_grid={
        "selected_forecaster": CategoricalDistribution(["croston", "adida", "tsb"]),
        "croston__smoothing": FloatDistribution(0.01, 1),
        "tsb__alpha": FloatDistribution(0.01, 1),
        "tsb__beta": FloatDistribution(0.01, 1),
    },
    scoring=scoring,
    sampler=TPESampler(seed=config.SEED),
    verbose=-1,
)
