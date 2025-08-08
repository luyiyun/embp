from typing import Literal, Callable
from copy import deepcopy
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import ndarray
from numpy.random import Generator
from scipy.special import ndtri

from .base import EM, BiomarkerPoolBase
from .continuous import ContinueEM
from .binary import LapBinaryEM, ISBinaryEM, VIBinaryEM


def bootstrap_estimator(
    estimator: EM | Callable,
    X: ndarray,
    Y: ndarray,
    W: ndarray,
    S: ndarray,
    Z: ndarray | None = None,
    Y_type: Literal["continue", "binary"] = False,
    n_repeat: int = 200,
    seed: int | None | Generator = None,
    pbar: bool = True,
    init_disturb: None | float = None,
) -> pd.DataFrame:
    assert hasattr(estimator, "params_"), "please run regular EM iteration firstly!"
    init_params = estimator.params_
    estimator._pbar = False
    seed = np.random.default_rng(seed)

    if init_disturb is not None:
        init_params = init_params + init_disturb * seed.normal(
            size=init_params.shape[0]
        )

    ind_bootstrap = seed.choice(Y.shape[0], (n_repeat, Y.shape[0]), replace=True)

    params_bs = []
    for i in tqdm(range(n_repeat), disable=not pbar, desc="Bootstrap: ", leave=False):
        ind_bs = ind_bootstrap[i]
        try:
            estimator.run(
                X[ind_bs],
                S[ind_bs],
                W[ind_bs],
                Y[ind_bs],
                None if Z is None else Z[ind_bs],  # nbs x N x nz
                init_params=init_params,
            )
        except np.linalg.LinAlgError:
            # warnings.warn("LinAlgError in bootstrap, skip this iteration")
            continue
        params_bs.append(estimator.parameters)

    return np.stack(params_bs, axis=0)


class EMBP(BiomarkerPoolBase):
    def __init__(
        self,
        outcome_type: Literal["continue", "binary"],
        max_iter: int | None = None,
        max_iter_inner: int = 100,
        delta1: float = 1e-3,
        delta1_inner: float = 1e-4,
        delta2: float | None = None,
        delta2_inner: float = 1e-7,
        delta1_var: float = 1e-1,
        delta2_var: float = 1e-3,
        ci: bool = False,
        ci_method: Literal["sem", "bootstrap"] = "bootstrap",
        ci_level: float = 0.95,
        n_bootstrap: int = 200,
        pbar: bool = True,
        seed: int | None = 0,
        device: str = "cpu",
        quasi_mc_K: int = 100,
        gem: bool = False,
        binary_solve: Literal["lap", "is", "vi"] = "is",
        importance_sampling_minK: int = 100,
        importance_sampling_maxK: int = 5000,
        bootstrap_init_disturb: float | None = None,
    ) -> None:
        """
        delta2: 1e-5 for continue, 1e-2 for binary
        max_iter: 500 for continue, 300 for binary
        """
        assert outcome_type in ["continue", "binary"]
        assert ci_method in ["bootstrap", "sem"]
        assert binary_solve in ["lap", "is", "vi"]
        if device != "cpu":
            try:
                import torch
            except ImportError:
                raise ImportError(
                    "torch is not installed, please install torch or BBP[torch]"
                )
        if (device != "cpu") and (
            outcome_type == "continue"
            or (outcome_type == "binary" and binary_solve == "lap")
        ):
            raise NotImplementedError
        # if outcome_type == "binary" and ci:
        #     raise NotImplementedError
        # if outcome_type == "binary" and variance_estimate_method == "sem":
        #     raise NotImplementedError(
        #         "use bootstrap for outcome_type = binary"
        #     )

        self.outcome_type_ = outcome_type
        self.max_iter_ = max_iter or {"continue": 500, "binary": 300}[outcome_type]
        self.max_iter_inner_ = max_iter_inner
        self.delta1_ = delta1
        self.delta1_inner_ = delta1_inner
        if delta2 is not None:
            self.delta2_ = delta2
        else:
            if outcome_type == "continue":
                self.delta2_ = 1e-5
            else:
                self.delta2_ = 1e-3 if binary_solve == "lap" else 1e-2
        # self.delta2_ = (
        #     delta2 or {"continue": 1e-5, "binary": 1e-3}[outcome_type]
        # )
        self.delta2_inner_ = delta2_inner
        self.delta1_var_ = delta1_var
        self.delta2_var_ = delta2_var
        self.pbar_ = pbar
        self.ci_ = ci
        self.ci_method_ = ci_method
        self.ci_level_ = ci_level
        self.n_bootstrap_ = n_bootstrap
        self.device_ = device
        self.seed_ = seed  # np.random.default_rng(seed)
        self.gem_ = gem
        self.binary_solve_ = binary_solve

        self.quasi_mc_K_ = quasi_mc_K
        self.importance_sampling_minK = importance_sampling_minK
        self.importance_sampling_maxK = importance_sampling_maxK
        self.bootstrap_init_disturb = bootstrap_init_disturb

        if (device != "cpu") and (seed is not None):
            torch.random.manual_seed(seed)

    @property
    def result_columns(self):
        if not self.ci_:
            return ["estimate"]
        if self.ci_method_ == "bootstrap":
            return ["estimate", "CI_1", "CI_2"]
        return ["estimate", "variance(log)", "std(log)", "CI_1", "CI_2"]

    @property
    def result_index(self):
        return self._estimator.get_params_names()

    def fit(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ) -> None:
        S = S.astype(int)

        if self.outcome_type_ == "continue":
            self._estimator = ContinueEM(
                max_iter=self.max_iter_,
                max_iter_inner=self.max_iter_inner_,
                delta1=self.delta1_,
                delta1_inner=self.delta1_inner_,
                delta2=self.delta2_,
                delta2_inner=self.delta2_inner_,
                delta1_var=self.delta1_var_,
                delta2_var=self.delta2_var_,
                pbar=self.pbar_,
            )
        elif self.outcome_type_ == "binary":
            if self.binary_solve_ == "lap":
                self._estimator = LapBinaryEM(
                    max_iter=self.max_iter_,
                    max_iter_inner=self.max_iter_inner_,
                    delta1=self.delta1_,
                    delta1_inner=self.delta1_inner_,
                    delta1_var=self.delta1_var_,
                    delta2=self.delta2_,
                    delta2_inner=self.delta2_inner_,
                    delta2_var=self.delta2_var_,
                    pbar=self.pbar_,
                    random_seed=self.seed_,
                    K=self.quasi_mc_K_,
                    gem=self.gem_,
                )
            elif self.binary_solve_ == "vi":
                self._estimator = VIBinaryEM(
                    max_iter=self.max_iter_,
                    max_iter_inner=self.max_iter_inner_,
                    delta1=self.delta1_,
                    delta1_inner=self.delta1_inner_,
                    delta1_var=self.delta1_var_,
                    delta2=self.delta2_,
                    delta2_inner=self.delta2_inner_,
                    delta2_var=self.delta2_var_,
                    pbar=self.pbar_,
                    random_seed=self.seed_,
                    K=self.quasi_mc_K_,
                    gem=self.gem_,
                )
            elif self.binary_solve_ == "is":
                if self.device_ == "cpu":
                    self._estimator = ISBinaryEM(
                        max_iter=self.max_iter_,
                        max_iter_inner=self.max_iter_inner_,
                        delta1=self.delta1_,
                        delta1_inner=self.delta1_inner_,
                        delta1_var=self.delta1_var_,
                        delta2=self.delta2_,
                        delta2_inner=self.delta2_inner_,
                        delta2_var=self.delta2_var_,
                        pbar=self.pbar_,
                        random_seed=self.seed_,
                        gem=self.gem_,
                        min_nIS=self.importance_sampling_minK,
                        max_nIS=self.importance_sampling_maxK,
                    )
                else:
                    from .binary_is_gpu import ISBinaryEMTorch

                    self._estimator = ISBinaryEMTorch(
                        max_iter=self.max_iter_,
                        max_iter_inner=self.max_iter_inner_,
                        delta1=self.delta1_,
                        delta1_inner=self.delta1_inner_,
                        delta2=self.delta2_,
                        delta2_inner=self.delta2_inner_,
                        delta1_var=self.delta1_var_,
                        delta2_var=self.delta2_var_,
                        pbar=self.pbar_,
                        gem=self.gem_,
                        min_nIS=self.importance_sampling_minK,
                        max_nIS=self.importance_sampling_maxK,
                        device=self.device_,
                    )
        self._estimator.run(X, S, W, Y, Z)
        self.params_ = pd.DataFrame(
            {"estimate": self._estimator.parameters},
            index=self._estimator.parameter_names,
        )
        self.params_hist_ = pd.DataFrame(
            self._estimator.parameter_history,
            columns=self._estimator.parameter_names,
        )

        if not self.ci_:
            return

        quan1 = (1 - self.ci_level_) / 2
        quan2 = 1 - quan1

        if self.ci_method_ == "bootstrap":
            # 使用boostrap方法
            # if self.pbar_:
            #     print("Bootstrap: ")
            res_bootstrap = bootstrap_estimator(
                # 使用复制品，而非原始的estimator
                estimator=deepcopy(self._estimator),
                X=X,
                Y=Y,
                W=W,
                S=S,
                Z=Z,
                Y_type=self.outcome_type_,
                n_repeat=self.n_bootstrap_,
                seed=self.seed_,
                pbar=self.pbar_,
                init_disturb=self.bootstrap_init_disturb,
            )
            self.res_bootstrap_ = pd.DataFrame(
                res_bootstrap, columns=self._estimator.parameter_names
            )
            res_ci = np.quantile(
                res_bootstrap,
                q=[quan1, quan2],
                axis=0,
            )
            self.params_["CI_1"] = res_ci[0, :]
            self.params_["CI_2"] = res_ci[1, :]
        else:
            zalpha = ndtri(quan2)

            params_var_, ind_sigma2 = self._estimator.estimate_variance()
            self.params_["variance(log)"] = params_var_
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "invalid value encountered in sqrt")
                self.params_["std(log)"] = np.sqrt(params_var_)
            CI = np.stack(
                [
                    self.params_["estimate"] - zalpha * self.params_["std(log)"],
                    self.params_["estimate"] + zalpha * self.params_["std(log)"],
                ],
                axis=1,
            )
            CI[ind_sigma2] = np.exp(CI[ind_sigma2])
            self.params_["CI_1"] = CI[:, 0]
            self.params_["CI_2"] = CI[:, 1]
