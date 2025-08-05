import numpy as np
from numpy import ndarray

from .binary_lap import LapBinaryEM
from .logger import logger_embp


class VIBinaryEM(LapBinaryEM):
    def prepare(self, X, S, W, Y, Z=None):
        super().prepare(X, S, W, Y, Z)
        self._xi_m = np.zeros(self._n_m)
        self._Vm = np.ones(self._n_m)

    def e_step(self, params: ndarray):
        mu_x, sigma2_x, a, b, sigma2_w, beta_x, beta_0 = (
            params[..., self._params_ind[k]]
            for k in [
                "mu_x",
                "sigma2_x",
                "a",
                "b",
                "sigma2_w",
                "beta_x",
                "beta_0",
            ]
        )
        beta_z = params[..., self._params_ind["beta_z"]] if self._Z is not None else 0.0

        beta_0_m_long = beta_0[self._ind_m_inv]

        b2_sigma2_w = b**2 / sigma2_w
        b2_sigma2_w_m_long = b2_sigma2_w[self._ind_m_inv]
        bwa_sigma2_w_m_long = (
            b[self._ind_m_inv]
            * (self._Wm - a[self._ind_m_inv])
            / sigma2_w[self._ind_m_inv]
        )
        mu_sigma2_x = mu_x / sigma2_x
        Z_part_m = 0.0 if self._Z is None else self._Zm @ beta_z
        beta_mul_m_long = -2 * beta_x * (beta_0_m_long + Z_part_m)

        # 使用坐标上升优化来得到后验分布
        Xm_old = self._Xm.copy()
        Vm_old = self._Vm.copy()
        for i in range(1, self._max_iter_inner + 1):
            lam_xi = np.tanh(self._xi_m * 0.5) / (4 * self._xi_m)

            # 根据xi更新m和v
            self._Vm = 1 / (2 * lam_xi * beta_x**2 + b2_sigma2_w_m_long + 1 / sigma2_x)
            self._Xm = self._Vm * (
                (self._Ym - 0.5) * beta_x
                + beta_mul_m_long * lam_xi
                + bwa_sigma2_w_m_long
                + mu_sigma2_x
            )

            # 根据m和v更新xi
            self._xi_m = np.sqrt(
                beta_x**2 * self._Vm
                + (beta_0_m_long + Z_part_m + beta_x * self._Xm) ** 2
            )

            # 计算m和v的变化量，并判断是否收敛
            rdiff = np.max(
                np.abs(np.concatenate([self._Xm - Xm_old, self._Vm - Vm_old]))
                / (np.abs(np.concatenate([self._Xm, self._Vm])) + self._delta1_inner)
            )
            logger_embp.info(f"E step Newton-Raphson: iter={i} diff={rdiff:.4f}")
            if rdiff < self._delta2_inner:
                break
        else:
            logger_embp.warning(
                f"E step Newton-Raphson (max_iter={self._max_iter_inner})"
                " doesn't converge"
            )

        # 计算Xhat和Xhat2
        self._Xhat[self._is_m] = self._Xm
        self._Xhat2[self._is_m] = self._Xm**2 + self._Vm
