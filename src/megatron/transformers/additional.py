import numpy as np
import pandas as pd
import scipy.stats as sp

from sktime.transformations.base import BaseTransformer


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
    def __init__(self, w: int):
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


def catch22_custom(x, w: int) -> list[float]:
    x, n = np.asarray(x, dtype=float), len(x)

    if n < w:
        return [0] * 22

    is_demand = x > 0
    nz_count = is_demand.sum()

    if nz_count == 0:
        return [0] * 22

    non_zero = x[is_demand]
    indices = np.flatnonzero(is_demand)
    intervals = np.diff(indices)
    n_intervals = intervals.size

    total_sum = x.sum()
    mean_x = x.mean()
    mean_nz = non_zero.mean()
    median_nz = np.median(non_zero)
    max_nz = non_zero.max()

    f_adi = n / nz_count

    if n_intervals > 0:
        mean_tau = intervals.mean()
        std_tau = intervals.std()
        f_max_drought = intervals.max() / n
    else:
        mean_tau = std_tau = 0
        f_max_drought = 1 - nz_count / n

    if n_intervals > 1 and mean_tau > 0:
        denom = std_tau + mean_tau
        f_burstiness = (std_tau - mean_tau) / denom if denom > 0 else 0
        f_interval_cv = std_tau / mean_tau
    else:
        f_burstiness = f_interval_cv = 0

    f_sparsity = 1 - nz_count / n
    f_onset = indices[0] / n
    f_obsolescence = (n - 1 - indices[-1]) / n

    prev, nxt = is_demand[:-1], is_demand[1:]
    total_z = (~prev).sum()
    total_o = prev.sum()
    f_p_wake = (~prev & nxt).sum() / total_z if total_z > 0 else 0
    f_p_clump = (prev & nxt).sum() / total_o if total_o > 0 else 0

    padded = np.zeros(n + 2, dtype=np.int8)
    padded[1:-1] = (~is_demand).view(np.uint8)  # zero-copy bool → uint8
    changes = np.diff(padded)
    zero_runs = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    f_mean_zero_run = zero_runs.mean() if zero_runs.size else 0

    if nz_count > 1 and mean_nz > 0:
        std_nz = non_zero.std()
        f_cv2 = (std_nz / mean_nz) ** 2

        sorted_nz = np.sort(non_zero)
        ranks = np.arange(1, nz_count + 1)
        f_size_gini = (2 * (ranks @ sorted_nz) - (nz_count + 1) * sorted_nz.sum()) / (
            nz_count * sorted_nz.sum()
        )
    else:
        f_cv2 = f_size_gini = 0

    hist = np.histogram(non_zero, bins="auto")[0]
    f_size_entropy = sp.entropy(hist) if hist.sum() > 0 else 0
    f_size_skew = np.nan_to_num(sp.skew(non_zero), nan=0) if nz_count > 2 else 0

    f_spike_ratio = max_nz / mean_nz if mean_nz > 0 else 0
    f_max_prop = max_nz / total_sum if total_sum > 0 else 0

    q90 = np.quantile(non_zero, 0.9)
    f_q90_q50_ratio = q90 / median_nz if median_nz > 0 else 0

    if n_intervals > 1:
        corr = np.corrcoef(intervals, non_zero[1:])[0, 1]
        f_wait_vol_corr = np.nan_to_num(corr, nan=0)
    else:
        f_wait_vol_corr = 0

    return [
        float(x)
        for x in [
            f_burstiness,
            f_sparsity,
            f_onset,
            f_obsolescence,
            f_max_drought,
            f_p_wake,
            f_p_clump,
            f_interval_cv,
            f_adi,
            f_size_gini,
            f_size_entropy,
            f_size_skew,
            f_cv2,
            f_spike_ratio,
            f_max_prop,
            f_wait_vol_corr,
            np.log1p(n),
            f_mean_zero_run,
            mean_nz,
            median_nz,
            f_q90_q50_ratio,
            mean_x,
        ]
    ]


def b_mad(x):
    return sp.median_abs_deviation(x)


def b_median(x):
    return np.median(x)


def c_occ_last(x):
    return float(x[-1] > 0)


def c_occ_rate(x):
    return np.mean(x > 0)


def c_occ_count(x):
    return np.sum(x > 0)


def c_non_occ_tail(x):
    return np.argmin((x == 0)[::-1]) if (x == 0).any() else x.size


def c_non_occ_head(x):
    return np.argmin(x == 0) if (x == 0).any() else x.size


def r_pos_last(x):
    return x[x > 0][-1] if x.any() else 0


def r_pos_mean(x):
    return x[x > 0].mean() if x.any() else 0


def r_pos_std(x):
    return x[x > 0].std() if x.any() else 0


def r_pos_sum(x):
    return x[x > 0].sum() if x.any() else 0
