import pandas as pd

from sklearn.base import clone

from sktime.forecasting.base import BaseForecaster

from pathlib import Path
from joblib import Parallel, delayed, dump, load
from tqdm_joblib import ParallelPbar
import warnings

from megatron.forecasters.se_models import (
    se_complex_global,
    se_simplex_global,
    se_complex_local,
)
from megatron.forecasters.il_models import (
    il_complex_global,
    il_simplex_global,
    il_simplex_local,
)

warnings.filterwarnings("ignore")


class CommonForecaster(BaseForecaster):
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

        if self.demand in ("smooth", "erratic"):
            self.pipelines = {
                "complex_global": se_complex_global,
                "simplex_global": se_simplex_global,
                "complex_local": se_complex_local,
            }
        else:
            self.pipelines = {
                "complex_global": il_complex_global,
                "simplex_global": il_simplex_global,
                "simplex_local": il_simplex_local,
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
                f"{path.relative_to(self.dir_path)} best {model.scoring.name} score: {round(model.best_score_, 3)}"
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
                    self.models[tuple([cluster, item])] = "simplex_local"

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
