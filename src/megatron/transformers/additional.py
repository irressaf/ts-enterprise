import numpy as np
import pandas as pd
import scipy.stats as sp

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


def catch22_custom(x, w=3):
    x, n = np.asarray(x, dtype=float), len(x)

    if n < w:
        return [0.0] * 22

    is_demand = x > 0
    nz_count = int(is_demand.sum())
    if nz_count == 0:
        return [0.0] * 22

    non_zero = x[is_demand]
    indices = np.flatnonzero(is_demand)
    intervals = np.diff(indices)

    # Core reusable stats
    total_sum = float(x.sum())
    mean_x = float(x.mean())
    mean_non_zero = float(non_zero.mean())
    median_non_zero = float(np.median(non_zero))
    max_non_zero = float(non_zero.max())

    # Inter-event features
    if intervals.size > 0:
        mean_tau = float(intervals.mean())
        std_tau = float(intervals.std())
        f_max_drought = float(intervals.max() / n)
        f_adi = float(n / nz_count)
    else:
        mean_tau = 0.0
        std_tau = 0.0
        f_max_drought = float(1.0 - nz_count / n)
        f_adi = float(n / nz_count)

    denom = std_tau + mean_tau
    f_burstiness = (
        float((std_tau - mean_tau) / denom) if intervals.size > 1 and denom > 0 else 0.0
    )
    f_interval_cv = (
        float(std_tau / mean_tau) if intervals.size > 1 and mean_tau > 0 else 0.0
    )

    # Timing features
    f_sparsity = float(1.0 - nz_count / n)
    f_onset = float(indices[0] / n)
    f_obsolescence = float((n - 1 - indices[-1]) / n)

    # Transition dynamics
    prev = is_demand[:-1]
    nxt = is_demand[1:]

    total_z = int((~prev).sum())
    total_o = int(prev.sum())

    z_to_o = int((~prev & nxt).sum())
    o_to_o = int((prev & nxt).sum())

    f_p_wake = float(z_to_o / total_z) if total_z > 0 else 0.0
    f_p_clump = float(o_to_o / total_o) if total_o > 0 else 0.0

    # Zero-run lengths via run-boundaries
    zero_int = (~is_demand).astype(np.int8)
    padded = np.pad(zero_int, (1, 1), constant_values=0)
    changes = np.diff(padded)
    run_starts = np.flatnonzero(changes == 1)
    run_ends = np.flatnonzero(changes == -1)
    zero_runs = run_ends - run_starts
    f_mean_zero_run = float(zero_runs.mean()) if zero_runs.size else 0.0

    # Distribution of non-zero sizes
    if nz_count > 1 and mean_non_zero > 0:
        std_non_zero = float(non_zero.std())
        f_cv2 = float((std_non_zero / mean_non_zero) ** 2)
        f_size_gini = float(
            np.abs(np.subtract.outer(non_zero, non_zero)).mean() / (2.0 * mean_non_zero)
        )
    else:
        f_cv2 = 0.0
        f_size_gini = 0.0

    hist = np.histogram(non_zero, bins="auto")[0]
    f_size_entropy = float(sp.entropy(hist)) if hist.sum() > 0 else 0.0

    f_size_skew = float(np.nan_to_num(sp.skew(non_zero), nan=0.0)) if nz_count > 2 else 0.0

    f_spike_ratio = float(max_non_zero / mean_non_zero) if mean_non_zero > 0 else 0.0
    f_max_prop = float(max_non_zero / total_sum) if total_sum > 0 else 0.0

    q90 = float(np.quantile(non_zero, 0.90))
    f_q90_q50_ratio = float(q90 / median_non_zero) if median_non_zero > 0 else 0.0

    # Wait-volume correlation
    if intervals.size > 1 and non_zero[1:].size == intervals.size:
        corr = np.corrcoef(intervals, non_zero[1:])[0, 1]
        f_wait_vol_corr = float(np.nan_to_num(corr, nan=0.0))
    else:
        f_wait_vol_corr = 0.0

    # Added non-overlapping features
    f_log_length = float(np.log1p(n))
    f_mean_nonzero = mean_non_zero
    f_median_nonzero = median_non_zero
    f_mean_interval_demand = mean_x

    return [
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
        f_log_length,
        f_mean_zero_run,
        f_mean_nonzero,
        f_median_nonzero,
        f_q90_q50_ratio,
        f_mean_interval_demand,
    ]