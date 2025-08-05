from abc import abstractmethod
from typing import Optional, Dict

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
from numpy import ndarray
from numpy.random import Generator
import pandas as pd

from .logger import logger_embp
from .utils import ols


class EM:
    def __init__(
        self,
        max_iter: int = 100,
        max_iter_inner: int = 100,
        delta1: float = 1e-3,
        delta1_inner: float = 1e-4,
        delta2: float = 1e-4,
        delta2_inner: float = 1e-6,
        delta1_var: float = 1e-2,
        delta2_var: float = 1e-2,
        pbar: bool = True,
        random_seed: int | None | Generator = None,
    ) -> None:
        self._max_iter = max_iter
        self._max_iter_inner = max_iter_inner
        self._delta1 = delta1
        self._delta1_inner = delta1_inner
        self._delta2 = delta2
        self._delta2_inner = delta2_inner
        self._delta1_var = delta1_var
        self._delta2_var = delta2_var
        self._pbar = pbar
        # 如果是Generator，则default_rng会直接返回它自身
        self._seed = np.random.default_rng(random_seed)

    @property
    def parameters(self) -> ndarray:
        """返回EM算法估计的参数值"""
        return self.params_

    @property
    @abstractmethod
    def parameter_history(self) -> ndarray:
        """返回EM算法的参数更新历史"""

    @property
    @abstractmethod
    def parameter_names(self) -> ndarray:
        """返回参数值的名称"""

    def run(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
        init_params: ndarray | None = None,
    ):
        self.prepare(X, S, W, Y, Z)

        params = self.init() if init_params is None else init_params
        self.params_hist_ = [params]
        with logging_redirect_tqdm(loggers=[logger_embp]):
            for self._iter_i in tqdm(
                range(1, self._max_iter + 1),
                desc="EM: ",
                disable=not self._pbar,
                leave=False,
            ):
                self.e_step(params)
                params_new = self.m_step(params)

                rdiff = self.calc_rdiff(params_new, params)
                logger_embp.info(
                    f"EM iteration {self._iter_i}: relative difference is {rdiff: .4f}"
                )
                params = params_new  # 更新
                self.params_hist_.append(params)

                if rdiff < self._delta2:
                    # self.iter_convergence_ = self._iter_i
                    break

                self.after_iter()
            else:
                logger_embp.warning(
                    f"EM iteration (max_iter={self._max_iter}) doesn't converge"
                )

        self.params_ = params
        # self.params_hist_ = np.stack(self.params_hist_)

    @abstractmethod
    def prepare(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ):
        """载入数据，并进行一些预先计算"""

    @abstractmethod
    def init(self) -> ndarray:
        """初始化参数"""

    @abstractmethod
    def e_step(self, params: ndarray):
        """E step，计算得到充分统计量"""

    @abstractmethod
    def m_step(self, params: ndarray) -> ndarray:
        """M step，更新参数值"""

    @abstractmethod
    def calc_rdiff(self, new: ndarray, old: ndarray) -> float:
        """计算相对差异，用于判断是否停止迭代"""

    def after_iter(self):
        """每次迭代后需要做的事情"""
        pass


class NumpyEM(EM):
    @property
    def parameter_history(self) -> ndarray:
        return np.stack(self.params_hist_)

    def prepare(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ):
        assert X.shape == S.shape == W.shape == Y.shape
        if Z is not None:
            assert X.shape == Z.shape[:-1]
        assert X.ndim == 1

        self._X = X
        self._S = S
        self._W = W
        self._Y = Y
        self._Z = Z

        # 准备后续步骤中会用到的array，预先计算，节省效率
        self._is_m = np.isnan(self._X)
        self._is_o = ~self._is_m
        self._ind_m = np.nonzero(self._is_m)[0]
        self._ind_o = np.nonzero(self._is_o)[0]

        self._Xo = self._X[self._ind_o]
        self._Yo = self._Y[self._ind_o]
        self._Wo = self._W[self._ind_o]
        self._Ym = self._Y[self._ind_m]
        self._Wm = self._W[self._ind_m]
        if self._Z is not None:
            self._Zo = self._Z[self._ind_o]
            self._Zm = self._Z[self._ind_m]
        else:
            self._Zo = self._Zm = None

        self._studies, self._ind_inv = np.unique(self._S, return_inverse=True)
        self._ind_m_inv = self._ind_inv[self._is_m]
        # self._ind_o_inv = self._ind_inv[self._is_o]

        # the transpose of 1-d array is still 1-d array
        self._ind_S = [np.nonzero(self._S == s)[0] for s in self._studies]
        self._ind_Sm = [
            np.nonzero((self._S.T == s) & self._is_m)[0] for s in self._studies
        ]
        self._ind_So = [
            np.nonzero((self._S.T == s) & self._is_o)[0] for s in self._studies
        ]

        self._n = self._Y.shape[-1]
        self._ns = self._studies.shape[-1]
        self._nz = 0 if self._Z is None else self._Z.shape[-1]
        self._n_o = self._is_o.sum(axis=-1)
        self._n_m = self._is_m.sum(axis=-1)
        self._n_s = np.array([indi.shape[-1] for indi in self._ind_S])

        self._wbar_s = np.stack([np.mean(self._W[ind], axis=-1) for ind in self._ind_S])
        self._wwbar_s = np.stack(
            [np.mean(self._W[ind] ** 2, axis=-1) for ind in self._ind_S]
        )

        self._sigma_ind = np.array(
            [1]
            + list(range(2 + 2 * self._ns, 2 + 2 * (self._ns + 1)))
            + list(range(3 + 4 * self._ns + self._nz, 3 + 5 * self._ns + self._nz))
        )
        self._params_ind = {
            "mu_x": slice(0, 1),
            "sigma2_x": slice(1, 2),
            "a": slice(2, 2 + self._ns),
            "b": slice(2 + self._ns, 2 + 2 * self._ns),
            "sigma2_w": slice(2 + 2 * self._ns, 2 + 3 * self._ns),
        }

        self._Xhat = np.copy(self._X)
        self._Xhat2 = self._Xhat**2

    def init(self) -> ndarray:
        """初始化参数

        这里仅初始化mu_x,sigma2_x,a,b,sigma2_w，其他和outcome相关的参数需要在子类
        中初始化

        Returns:
            dict: 参数组成的dict
        """
        mu_x = self._Xo.mean(axis=-1, keepdims=True)
        sigma2_x = np.var(self._Xo, ddof=1, axis=-1, keepdims=True)

        a, b, sigma2_w = [], [], []
        for ind_so_i in self._ind_So:
            if len(ind_so_i) == 0:
                a.append(0)
                b.append(0)
                sigma2_w.append(1)
                continue

            abi, sigma2_ws_i = ols(self._X[ind_so_i], self._W[ind_so_i])
            a.append(abi[1])
            b.append(abi[0])
            sigma2_w.append(sigma2_ws_i)
        a, b, sigma2_w = (
            np.stack(a, axis=-1),
            np.stack(b, axis=-1),
            np.stack(sigma2_w, axis=-1),
        )

        res = np.concatenate([mu_x, sigma2_x, a, b, sigma2_w], axis=-1)
        return res

    def calc_rdiff(self, new: ndarray, old: ndarray) -> float:
        return np.max(np.abs(old - new) / (np.abs(old) + self._delta1))

    def v_joint(self, params_log: ndarray) -> ndarray:
        """计算Vjoint，注意的时候是对应于log sigma2进行的"""
        raise NotImplementedError

    def estimate_variance(self) -> ndarray:
        params_w_log = self.parameters.copy()
        ind_sigma2 = np.nonzero(
            [pn.startswith("sigma2") for pn in self.parameter_names]
        )[0]
        params_w_log[ind_sigma2] = np.log(params_w_log[ind_sigma2])
        n_params = params_w_log.shape[0]

        params_hist = self.parameter_history

        rind_uncovg = list(range(n_params))
        R = []
        with logging_redirect_tqdm(loggers=[logger_embp]):
            for t in tqdm(
                range(params_hist.shape[0]),
                desc="Estimate Variance: ",
                disable=not self._pbar,
            ):
                params_t = params_hist[t, :]
                Rt = np.zeros((n_params, n_params)) if t == 0 else R[-1].copy()
                for j in rind_uncovg:
                    inpt = self.parameters.copy()
                    x = inpt[j] = params_t[j]
                    if j in ind_sigma2:
                        x = np.log(x)
                    # 计算差值比来作为导数的估计
                    dx = x - params_w_log[j]
                    if dx == 0:  # 如果dx=0了，就用上一个结果  TODO: 方差可能还没有收敛
                        continue

                    self.e_step(inpt)
                    oupt = self.m_step(inpt)
                    # 修改sigma2为log尺度
                    oupt[ind_sigma2] = np.log(oupt[ind_sigma2])

                    Rt[j, :] = (oupt - params_w_log) / dx

                # 看一下有哪些行完成了收敛
                if t > 0:
                    rdiff = np.max(
                        np.abs(Rt - R[-1]) / (np.abs(R[-1]) + self._delta1_var),
                        axis=1,
                    )
                    new_rind_uncovg = np.nonzero(rdiff >= self._delta2_var)[0]
                    if len(new_rind_uncovg) < len(rind_uncovg):
                        logger_embp.info("unfinished row ind:" + str(rind_uncovg))
                    rind_uncovg = new_rind_uncovg

                R.append(Rt)
                if len(rind_uncovg) == 0:
                    break
            else:
                logger_embp.warn("estimate variance does not converge.")

        self._R = np.stack(R, axis=0)
        DM = R[-1]

        v_joint = self.v_joint(self.params_)
        self.params_cov_ = v_joint + v_joint @ DM @ np.linalg.inv(
            np.diag(np.ones(n_params)) - DM
        )
        # TODO: 会出现一些方差<0，可能源于
        params_var = np.diag(self.params_cov_)
        # if np.any(params_var < 0.0):
        #     print(self.params_.index.values[params_var < 0.0])
        #     dv = params_var - np.diag(v_joint)
        #     import ipdb; ipdb.set_trace()
        return params_var, ind_sigma2


class BiomarkerPoolBase:
    def __init__(self) -> None:
        raise NotImplementedError

    def fit(
        X: np.ndarray,
        S: np.ndarray,
        W: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> None:
        raise NotImplementedError

    def summary() -> pd.DataFrame:
        raise NotImplementedError


def check_split_data(
    X: np.ndarray,
    S: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    Z: Optional[np.ndarray] = None,
) -> Dict:
    for arr in [X, S, W, Y]:
        assert arr.ndim == 1
    assert X.shape[0] == S.shape[0] == W.shape[0] == Y.shape[0]
    if Z is not None:
        assert Z.ndim == 2
        assert Z.shape[0] == X.shape[0]

    studies, ind_s_inv = np.unique(S, return_inverse=True)
    is_m = pd.isnull(X)
    is_o = np.logical_not(is_m)
    n_studies = len(studies)
    n_xKnow = int(is_o.sum())
    n_xUnKnow = int(is_m.sum())

    XWYZ_xKnow, WYZ_xUnKnow, n_ms, n_os = [], [], [], []
    ind_s = []
    for si in studies:
        ind_si = np.nonzero((S == si) & is_o)[0]
        n_os.append(len(ind_si))
        if ind_si.sum() == 0:
            XWYZ_xKnow.append(None)
        else:
            item = [ind_si, X[ind_si], W[ind_si], Y[ind_si], None]
            if Z is not None:
                item[-1] = (Z[ind_si, :],)
            XWYZ_xKnow.append(item)

        ind_si_n = np.nonzero((S == si) & is_m)[0]
        n_ms.append(len(ind_si_n))
        if ind_si_n.sum() == 0:
            WYZ_xUnKnow.append(None)
        else:
            item = [ind_si_n, W[ind_si_n], Y[ind_si_n], None]
            if Z is not None:
                item[-1] = Z[ind_si_n, :]
            WYZ_xUnKnow.append(item)

        ind_s.append(np.nonzero(S == si)[0])

    return {
        "ind_studies": studies,
        "n_studies": n_studies,
        "n_xKnow": n_xKnow,
        "n_xUnknow": n_xUnKnow,
        "n_ms": np.array(n_ms),
        "n_os": np.array(n_os),
        "ind_o": np.nonzero(is_o)[0],
        "ind_s": ind_s,
        "ind_s_inv": ind_s_inv,
        "XWYZ_xKnow": XWYZ_xKnow,
        "WYZ_xUnKnow": WYZ_xUnKnow,
    }
