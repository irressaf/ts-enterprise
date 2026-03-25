import numpy as np
import pandas as pd
from itertools import combinations

from pycatch22 import catch22_all

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from sktime.clustering.base import BaseClusterer
from sktime.distances import pairwise_distance

from joblib import Parallel, delayed

import megatron.config as config


class SmoothErraticClusterer(BaseClusterer):
    _tags = {
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "capability:unequal_length": True,
    }

    def __init__(self, w=config.MIN_LENGTH, n_jobs=-1):
        self.w = w
        self.n_jobs = n_jobs

        super().__init__()

    def _wdtw_matching(self, item):
        X = (
            self.X_valid_temp[["labels"]]
            .assign(
                score=pairwise_distance(
                    x=np.array(self.X_invalid_temp.loc[item, self.column]).reshape(
                        1, -1
                    ),
                    y=self.X_valid_temp_series_array,
                    metric="wdtw",
                    g=0.05,
                ).reshape(-1, 1)
            )
            .assign(index=item)
        )
        return (
            X.loc[X["score"].idxmin()]
            .drop(index="score")
            .to_frame()
            .T.set_index("index")
        )

    def _clustering(self, n: int, init: int):
        model = KMeans(n_clusters=n, n_init=init)
        labels = model.fit_predict(self.X_valid_temp_features_array)
        sil_score = silhouette_score(self.X_valid_temp_features_array, labels)
        return [labels, sil_score, model.inertia_]

    def _statistics_per_n_clusters(self, n: int):
        temp = Parallel(n_jobs=self.n_jobs)(
            delayed(self._clustering)(n, 25) for _ in range(100)
        )

        aris = np.array(
            [
                adjusted_rand_score(x, y)
                for x, y in combinations([x[0] for x in temp], 2)  # type: ignore
            ]
        )
        sil_scores = np.array([x[1] for x in temp])  # type: ignore 
        inertias = np.array([x[2] for x in temp]) # type: ignore

        return (
            pd.Series(
                {
                    "avg_sil_score": sil_scores.mean(),
                    "avg_ari": aris.mean(),
                    "std_ari": aris.std(ddof=1),
                    "std_inertia": inertias.std(ddof=1),
                    "score": (
                        aris.mean()
                        - aris.std(ddof=1)
                        + 0.5 * sil_scores.mean()
                        - 0.01 * inertias.std(ddof=1)
                    ),
                }
            )
            .rename(n)
            .to_frame()
            .T
        )

    def _fit(self, X, y=None):
        self.items, self.column = X.index.droplevel(-1).unique(), X.columns[0]
        lengths = X.groupby(self.items.names).size()
        self.valid_ids = lengths[lengths.ge(self.w)].index
        self.invalid_ids = lengths[lengths.lt(self.w)].index

        self.X_valid_temp = (
            X.loc[self.valid_ids]
            .groupby(self.items.names)
            .apply(lambda x: x.values[-self.w :].tolist())
            .rename(self.column)
            .to_frame()
        )
        self.X_valid_temp_series_array = np.array(
            self.X_valid_temp[self.column].tolist()
        ).reshape(-1, self.w)
        self.X_valid_temp_features_array = np.vstack(
            self.X_valid_temp[self.column].apply(
                lambda x: [
                    x**0.5 if i > 21 else x
                    for i, x in enumerate(
                        catch22_all(np.array(x).flatten(), catch24=True)["values"]
                    )
                ]
            )
        )

        self.X_invalid_temp = (
            X.loc[self.invalid_ids]
            .groupby(self.items.names)
            .apply(lambda x: x.values.tolist())
            .rename(self.column)
            .to_frame()
        )

        n = self.X_valid_temp.shape[0]
        self.metrics = pd.concat(
            [
                self._statistics_per_n_clusters(i) if i else pd.DataFrame()
                for i in range(int(n**0.5 / 2), int(n**0.5) + 1)
            ]
        )
        self.n_clusters = self.metrics["score"].idxmax()

        self.X_valid_temp["labels"] = self._clustering(self.n_clusters, 1000)[0]  # type: ignore

        temp = pd.concat(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._wdtw_matching)(item) for item in self.invalid_ids
            )
        )

        self.labels = pd.concat(
            [
                self.X_valid_temp["labels"],
                self.X_invalid_temp.join(temp)["labels"],
            ]
        ).to_dict()
        self.labels = {int(k): int(v) for k, v in self.labels.items()}

        del (
            self.X_valid_temp,
            self.X_valid_temp_series_array,
            self.X_valid_temp_features_array,
            self.X_invalid_temp,
        )

        return self.items

    def _predict(self, X, y=None):
        return np.array([self.labels.get(x) for x in self.items])
