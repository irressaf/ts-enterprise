import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

from megatron import config


class Mapper(BaseTransformer):
    _tags = {
        "scitype:transform-input": "Dataframe",
        "scitype:transform-output": "Dataframe",
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "capability:inverse_transform": True,
        "fit_is_empty": False,
    }

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        self.index = X.index.names
        self._mapper = {x: i for i, x in enumerate(X.droplevel(-1).index.unique())}
        return self

    def _transform(self, X, y=None):
        return (
            X.assign(index=[self._mapper[x] for x in X.droplevel(-1).index])
            .droplevel(self.index[:-1])
            .set_index(["index"], append=True)
            .reorder_levels(["index"] + self.index[-1:])
        )

    def _inverse_transform(self, X, y=None):
        temp = pd.DataFrame(
            data=self._mapper.keys(),
            index=pd.Index(self._mapper.values(), name="index"),
            columns=self.index[:-1],
        )
        return X.join(temp).reset_index(self.index[-1:]).set_index(self.index)


class InitialPreprocessing:
    def __init__(self, w=config.SEASONAL_PERIOD) -> None:
        self.w = w

    def drop_zero_series(self, X: pd.DataFrame):
        temp = X.groupby(level=0)[X.columns[0]].nunique()
        return X.loc[temp[temp.gt(1)].index]

    def trim_leading_zeros(self, X: pd.DataFrame):
        return (
            X.groupby(level=0)[X.columns[0]]
            .apply(lambda x: x.loc[x.gt(0).idxmax() :].droplevel(0))
            .to_frame()
        )

    def drop_trailing_zero_window_series(self, X: pd.DataFrame):
        temp = X.groupby(level=0)[X.columns[0]].apply(
            lambda x: x.loc[x[::-1].gt(0).idxmax() :].size - 1
        )
        return X.loc[temp[temp.lt(self.w)].index]


class DemandClassifier(BaseTransformer):
    _tags = {
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "scitype:transform-output": "Dataframe",
    }

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        index, X = X.index.names, X[X.columns[0]]

        g = X.groupby(index[:-1])
        T, N = g.size(), g.apply(lambda s: s.gt(0).sum())
        g_nonzero = X[X.gt(0)].groupby(index[:-1])

        adi, cv2 = T / N, (g_nonzero.std() / g_nonzero.mean()) ** 2
        temp = pd.DataFrame({"adi": adi, "cv2": cv2})

        conditions = [
            (temp["adi"].lt(1.32) & temp["cv2"].lt(0.49)).values.flatten(),  # type: ignore
            (temp["adi"].lt(1.32) & temp["cv2"].ge(0.49)).values.flatten(),  # type: ignore
            (temp["adi"].ge(1.32) & temp["cv2"].lt(0.49)).values.flatten(),  # type: ignore
            (temp["adi"].ge(1.32) & temp["cv2"].ge(0.49)).values.flatten(),  # type: ignore
        ]
        classes = [
            np.array([x] * temp.shape[0])
            for x in ["smooth", "erratic", "intermittent", "lumpy"]
        ]
        temp["class"] = np.select(conditions, classes, default="unknown")

        return temp
