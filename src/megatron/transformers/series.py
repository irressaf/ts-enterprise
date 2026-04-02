import numpy as np
import pandas as pd
import scipy.stats as sp
import ruptures as rpt
from holidays import country_holidays
from pyod.models.iforest import IForest

from sktime.transformations.base import BaseTransformer
from sktime.transformations.series.holiday import HolidayFeatures
from sktime.transformations.series.func_transform import FunctionTransformer
from sktime.transformations.series.date import DateTimeFeatures

from joblib import Parallel, delayed
import megatron.config as config


class PlateauDetector(BaseTransformer):
    _tags = {
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "transform-returns-same-time-index": False,
        "scitype:transform-output": ["Primitives", "Dataframe"],
    }

    def __init__(self, w: int, value=np.nan, truncate=False, n_jobs=-1):
        self.w = w
        self.value = value
        self.truncate = truncate
        self.n_jobs = n_jobs

        super().__init__()

    def _pld(self, data: pd.DataFrame):
        column = data.columns[0]
        data["mask"] = (
            data[column].isna() if pd.isna(self.value) else data[column].eq(self.value)
        )
        data["shift_mask"] = data["mask"].shift()
        data["gap"] = data["mask"].ne(data["shift_mask"]).cumsum()
        temp = data[data["mask"]].join(
            data.groupby("gap").size().rename("len"), on="gap"
        )

        temp = temp[temp["len"].ge(self.w)]
        if self.truncate:
            return (
                data.loc[:, [column]]
                if temp.empty
                else data.loc[temp.index.max() :, [column]].iloc[1:]
            )
        else:
            return (
                temp.groupby("gap")["mask"]
                .apply(lambda x: (x.index.min(), x.index.max()))
                .values
            )

    def _transform(self, X, y=None):
        if self.truncate:
            index = X.droplevel(-1).index.names
            groups = [x for _, x in X.groupby(index)]
            temp = Parallel(n_jobs=self.n_jobs)(delayed(self._pld)(g) for g in groups)
            return pd.concat(temp)
        else:
            return self._pld(X.droplevel(0))


class ChangePointDetector(BaseTransformer):
    _tags = {
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "transform-returns-same-time-index": False,
        "scitype:transform-output": ["Primitives", "Dataframe"],
    }

    def __init__(self, w: int, truncate=False, n_jobs=-1):
        self.w = w
        self.truncate = truncate
        self.n_jobs = n_jobs

        super().__init__()

    def _cpd(self, data: pd.DataFrame):
        if 2 * self.w < data.shape[0]:
            models = {
                "kernelcpd": rpt.KernelCPD(kernel="linear", min_size=self.w),
                "binreg": rpt.Binseg(model="ar", min_size=self.w),
            }

            temp, cps = data.dropna(), []
            for name in models:
                try:
                    model = models[name].fit(temp.values)
                    i = model.predict(n_bkps=1)[0]
                    cps += [temp.index[i]]
                except rpt.exceptions.BadSegmentationParameters:
                    pass
            else:
                return data.loc[max(cps) :] if self.truncate else max(cps)
        else:
            return data if self.truncate else data.index.min()

    def _transform(self, X, y=None):
        if self.truncate:
            index = X.droplevel(-1).index.names
            groups = [x for _, x in X.groupby(index)]
            temp = Parallel(n_jobs=self.n_jobs)(delayed(self._cpd)(g) for g in groups)
            return pd.concat(temp)
        else:
            return self._cpd(X.droplevel(0))


class OutlierDetector(BaseTransformer):
    _tags = {
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "scitype:transform-output": "Dataframe",
    }

    def __init__(self, demand: str, exog_column=None, truncate=False, n_jobs=-1):
        self.demand = demand
        self.exog_column = exog_column
        self.truncate = truncate
        self.n_jobs = n_jobs

        super().__init__()

    def _od(self, data: pd.DataFrame):
        temp, columns = data.dropna(), data.columns
  
        if self.demand in ("smooth", "erratic"):
            mask = temp[columns[-1]].values  # type: ignore
            model = IForest(contamination=0.05, behaviour="new", random_state=config.SEED)
            model.fit(temp[[columns[0]]].values)
            outlier_mask = model.predict(temp[[columns[0]]].values).astype(bool)  # type: ignore
            mask &= outlier_mask
        else:
            mask = temp[columns[-1]].values
            values = np.log(temp.loc[temp[columns[0]].gt(0), columns[0]])
            mad = sp.median_abs_deviation(values, scale=1.4826)
            mad = sp.iqr(values) / 1.349 if mad == 0 else mad

            if mad > 0:
                threshold = 3 if self.demand == "intermittent" else 4
                values = (values - np.median(values)) / mad
                outlier_mask = temp.index.isin(values.index[values.gt(threshold)])  # type: ignore
                mask &= outlier_mask  # type: ignore
            else:
                mask = np.zeros(len(mask), dtype=bool)

        if self.truncate:
            data.loc[temp[mask].index, columns[0]] = np.nan
            return data[[columns[0]]]
        else:
            return temp[mask][columns[0]]

    def _transform(self, X, y=None):
        hf = HolidayFeatures(
            calendar=country_holidays(
                country=config.COUNTRY,
                years=[*range(config.MIN_DATE.year, config.MAX_DATE.year + 1)],
            ),
            include_bridge_days=True,
            return_dummies=False,
            return_indicator=True,
        )
        temp = pd.DataFrame(
            index=pd.Index(
                pd.date_range(config.MIN_DATE, config.MAX_DATE), name=X.index.names[-1]
            )
        )
        temp = ~hf.fit_transform(temp).astype(bool)  # type: ignore
        temp = temp[temp.columns[0]]

        if self.exog_column is not None:
            temp = temp & ~X[self.exog_column].astype(bool)
            X = X[X.columns[0]].to_frame()
        X = X.join(temp.rename("mask"))

        if self.truncate:
            index = X.droplevel(-1).index.names
            groups = [x for _, x in X.groupby(index)]
            temp = Parallel(n_jobs=self.n_jobs)(delayed(self._od)(g) for g in groups)
            return pd.concat(temp)
        else:
            return self._od(X.droplevel(0))


class ExogenousDataTransformer(BaseTransformer):
    _tags = {
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "scitype:transform-output": "Dataframe",
        "fit_is_empty": False,
    }

    def __init__(self):

        super().__init__()

    def _wageDateFlag(self, data: pd.DataFrame):
        wage = pd.DataFrame(
            {
                "date": pd.date_range(config.MIN_DATE, config.MAX_DATE, freq="SME"),
                "is_wage": 1,
            }
        ).set_index("date")
        return data.join(wage).fillna(0).astype(int)

    def _fit(self, X, y=None):
        import warnings

        warnings.filterwarnings("ignore")

        self.date_time_transformer = DateTimeFeatures(
            manual_selection=[
                "month_of_year",
                "day_of_month",
                "day_of_week",
                "is_weekend",
            ],
            keep_original_columns=True,
        )
        self.holidays_transformer = HolidayFeatures(
            calendar=country_holidays(
                country=config.COUNTRY,
                years=[*range(config.MIN_DATE.year, config.MAX_DATE.year + 1)],
            ),
            include_bridge_days=True,
            return_dummies=False,
            return_indicator=True,
            keep_original_columns=True,
        )

        self.wage_transformer = FunctionTransformer(self._wageDateFlag)

        return self

    def _transform(self, X, y=None):
        import warnings

        warnings.filterwarnings("ignore")

        temp = pd.DataFrame(
            index=pd.Index(pd.date_range(config.MIN_DATE, config.MAX_DATE), name="date")
        )
        temp = self.date_time_transformer.fit_transform(temp)
        temp = self.holidays_transformer.fit_transform(temp)
        temp = self.wage_transformer.fit_transform(temp)
        return X.join(temp.astype(float))  # type: ignore
