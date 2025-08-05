import numpy as np
from numpy import ndarray
from scipy.linalg import block_diag as sc_block_diag

from .logger import logger_embp
from .base import NumpyEM
from .utils import ols


class ContinueEM(NumpyEM):
    @property
    def parameter_names(self) -> list:
        return (
            ["mu_x", "sigma_x"]
            + [f"a_{si}" for si in self._studies]
            + [f"b_{si}" for si in self._studies]
            + [f"sigma2_w_{si}" for si in self._studies]
            + ["beta_x"]
            + [f"beta_0_{si}" for si in self._studies]
            + [f"beta_z_{i}" for i in range(self._nz)]
            + [f"sigma2_y_{si}" for si in self._studies]
        )

    def prepare(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ):
        super().prepare(X, S, W, Y, Z)

        self._ybar_s = np.array([self._Y[ind].mean(axis=-1) for ind in self._ind_S])
        self._yybar_s = np.array(
            [(self._Y[ind] ** 2).mean(axis=-1) for ind in self._ind_S]
        )

        if self._Z is not None:
            self._zzbar_s = []
            self._zbar_s = []
            self._yzbar_s = []
            for ind, n_s in zip(self._ind_S, self._n_s):
                Zs = self._Z[ind]
                self._zzbar_s.append(np.einsum("ij,ik->jk", Zs, Zs) / n_s)
                self._zbar_s.append(Zs.mean(axis=-2))
                self._yzbar_s.append(np.einsum("ij,i->j", Zs, self._Y[ind]))
            self._zzbar_s = np.stack(self._zzbar_s, axis=-3)
            self._zbar_s = np.stack(self._zbar_s, axis=-2)
            self._yzbar_s = np.stack(self._yzbar_s, axis=-2)
        else:
            self._zzbar_s = self._zbar_s = self._yzbar_s = 0

        self._params_ind.update(
            {
                "beta_x": slice(2 + 3 * self._ns, 3 + 3 * self._ns),
                "beta_0": slice(3 + 3 * self._ns, 3 + 4 * self._ns),
                "beta_z": slice(3 + 4 * self._ns, 3 + 4 * self._ns + self._nz),
                "sigma2_y": slice(
                    3 + 4 * self._ns + self._nz, 3 + 5 * self._ns + self._nz
                ),
            }
        )

    def init(self) -> ndarray:
        params = super().init()

        beta, sigma2_ys = ols(self._Xo, self._Yo, self._Zo)
        res = np.r_[
            params,
            beta[0],
            [beta[1]] * self._ns,
            beta[2:],
            [sigma2_ys] * self._ns,
        ]
        return res

    def e_step(self, params: ndarray):
        mu_x, sigma2_x, a, b, sigma2_w, beta_x, beta_0, sigma2_y = (
            params[self._params_ind[k]]
            for k in [
                "mu_x",
                "sigma2_x",
                "a",
                "b",
                "sigma2_w",
                "beta_x",
                "beta_0",
                "sigma2_y",
            ]
        )
        beta_z = params[self._params_ind["beta_z"]] if self._Z is not None else 0.0

        sigma2_denominator = (
            sigma2_w * sigma2_x * beta_x**2
            + sigma2_y * sigma2_x * b**2
            + sigma2_y * sigma2_w
        )
        sigma2 = sigma2_w * sigma2_x * sigma2_y / sigma2_denominator

        z_m_part = 0.0 if self._Z is None else np.einsum("ij,j->i", self._Zm, beta_z)
        beta_0_m_long = beta_0[self._ind_m_inv]
        sigma2_y_m_long = sigma2_y[self._ind_m_inv]
        a_m_long = a[self._ind_m_inv]
        b_m_long = b[self._ind_m_inv]
        sigma2_w_m_long = sigma2_w[self._ind_m_inv]
        sigma2_denominator_m_long = sigma2_denominator[self._ind_m_inv]
        sigma2_m_long = sigma2[self._ind_m_inv]

        xhat_m = (
            (self._Ym - beta_0_m_long - z_m_part) * beta_x * sigma2_w_m_long * sigma2_x
            + (self._Wm - a_m_long) * b_m_long * sigma2_y_m_long * sigma2_x
            + mu_x * sigma2_w_m_long * sigma2_y_m_long
        ) / sigma2_denominator_m_long

        self._Xhat[self._ind_m] = xhat_m
        self._Xhat2[self._ind_m] = xhat_m**2 + sigma2_m_long

    def m_step(self, params: ndarray) -> ndarray:
        vbar = self._Xhat2.mean(axis=-1)
        wxbar_s = np.stack(
            [np.mean(self._W[ind] * self._Xhat[ind], axis=-1) for ind in self._ind_S]
        )
        vbar_s = np.stack([np.mean(self._Xhat2[ind], axis=-1) for ind in self._ind_S])
        xbar_s = np.stack([np.mean(self._Xhat[ind], axis=-1) for ind in self._ind_S])
        xybar_s = np.array(
            [np.mean(self._Xhat[ind] * self._Y[ind], axis=-1) for ind in self._ind_S]
        )

        if self._Z is not None:
            # xzbar = np.mean(self._Xhat[:, None] * self._Z, axis=0)
            xzbar_s = np.stack(
                [
                    np.mean(self._Xhat[ind][..., None] * self._Z[ind], axis=-2)
                    for ind in self._ind_S
                ],
                axis=0,
            )
            # if self._batch_mode:
            #     xzbar_s = xzbar_s.swapaxes(0, 1)
        else:
            xzbar_s = 0.0

        # 3. M step，更新参数值
        mu_x = np.mean(self._Xhat, axis=-1)
        sigma2_x = vbar - mu_x**2
        b = (wxbar_s - self._wbar_s * xbar_s) / (vbar_s - xbar_s**2)
        a = self._wbar_s - b * xbar_s
        sigma2_w = (
            self._wwbar_s
            + a**2
            + b**2 * vbar_s
            - 2 * (a * self._wbar_s + b * wxbar_s - a * b * xbar_s)
        )
        # 迭代更新beta值
        beta_all = params[(2 + 3 * self._ns) :]
        for i in range(self._max_iter_inner):
            beta_all_new = beta_all.copy()

            # 关于z的一些项
            if self._Z is None:
                xzd = yzd = dzzd = zd = xzd = 0.0
            else:
                beta_z = beta_all_new[-(self._ns + self._nz) : -self._ns]
                xzd = np.einsum("ij,j->i", xzbar_s, beta_z)
                zd = np.einsum("ij,j->i", self._zbar_s, beta_z)
                yzd = np.einsum("ij,j->i", self._yzbar_s, beta_z)
                xzd = np.einsum("ij,j->i", xzbar_s, beta_z)
                dzzd = np.einsum("i,tij,j->t", beta_z, self._zzbar_s, beta_z)

            # beta_x
            # 为了避免zero variance的问题，需要使用一些通分技巧
            sigma2_y = beta_all_new[-self._ns :]
            sigma2_y_prod = np.stack(
                [
                    np.prod(np.delete(sigma2_y, i, axis=-1), axis=-1)
                    for i in range(self._ns)
                ]
            )
            beta_x_new = beta_all_new[0] = (
                self._n_s
                * (xybar_s - beta_all_new[1 : (1 + self._ns)] * xbar_s - xzd)
                * sigma2_y_prod
            ).sum(axis=-1, keepdims=True) / (self._n_s * vbar_s * sigma2_y_prod).sum(
                axis=-1, keepdims=True
            )
            # beta_0
            beta_0_new = beta_all_new[1 : (1 + self._ns)] = (
                self._ybar_s - zd - beta_x_new * xbar_s
            )
            # sigma2_y
            sigma2_y_new = beta_all_new[-self._ns :] = (
                self._yybar_s
                + beta_0_new**2
                + beta_x_new**2 * vbar_s
                + dzzd
                - 2 * beta_0_new * self._ybar_s
                - 2 * beta_x_new * xybar_s
                - 2 * yzd
                + 2 * beta_0_new * beta_x_new * xbar_s
                + 2 * beta_0_new * zd
                + 2 * beta_x_new * xzd
            )
            # beta_z
            if self._Z is not None:
                # TODO: sigma2_y_new =0 可能会导致问题
                tmp1 = np.linalg.inv(
                    np.sum(
                        self._n_s[..., None, None]  # (nbs,)ns
                        / sigma2_y_new[..., None, None]  # (nbs,)ns
                        * self._zzbar_s,  # (nbs,)ns,nz,nz
                        axis=-3,
                    )
                )
                tmp2 = np.sum(
                    (self._n_s / sigma2_y_new)[..., None]  # (nbs,)ns
                    * (
                        self._yzbar_s  # (nbs,)ns,nz
                        - beta_0_new[..., None] * self._zbar_s  # (nbs,)ns,nz
                        - beta_x_new[..., None] * xzbar_s  # (nbs,)1
                    ),
                    axis=-2,
                )
                beta_all_new[-(self._ns + self._nz) : -self._ns] = np.einsum(
                    "ij,j->i", tmp1, tmp2
                )

            # calculate the relative difference
            rdiff = np.max(
                np.abs(beta_all_new - beta_all)
                / (np.abs(beta_all) + self._delta1_inner),
                axis=-1,
            )
            logger_embp.info(
                f"Inner iteration {i + 1}: relative difference is {rdiff: .4f}"
            )

            # update parameters
            beta_all = beta_all_new

            if rdiff < self._delta2_inner:
                logger_embp.info(f"Inner iteration stop, stop iter: {i + 1}")
                break

        else:
            logger_embp.warn("Inner iteration does not converge")

        return np.concatenate(
            [
                np.expand_dims(mu_x, axis=-1),
                np.expand_dims(sigma2_x, axis=-1),
                a,
                b,
                sigma2_w,
                beta_all,
            ],
            axis=-1,
        )

    def v_joint(self, params: ndarray) -> ndarray:
        mu_x, sigma2_x, a, b, sigma2_w, beta_x, beta_0, sigma2_y = (
            params[self._params_ind[k]]
            for k in [
                "mu_x",
                "sigma2_x",
                "a",
                "b",
                "sigma2_w",
                "beta_x",
                "beta_0",
                "sigma2_y",
            ]
        )
        beta_z = params[self._params_ind["beta_z"]] if self._Z is not None else 0.0
        mu_x, sigma2_x = mu_x[0], sigma2_x[0]  # array->item

        self.e_step(params)

        xbar = self._Xhat.mean()
        vbar = self._Xhat2.mean()
        wxbar_s = np.array(
            [np.mean(self._W[ind] * self._Xhat[ind]) for ind in self._ind_S]
        )
        vbar_s = np.array([np.mean(self._Xhat2[ind]) for ind in self._ind_S])
        xbar_s = np.array([np.mean(self._Xhat[ind]) for ind in self._ind_S])

        xybar_s = np.array(
            [np.mean(self._Xhat[ind] * self._Y[ind]) for ind in self._ind_S]
        )
        if self._Z is not None:
            xzbar = np.mean(self._Xhat[:, None] * self._Z, axis=0)
            xzbar_s = np.stack(
                [
                    np.mean(self._Xhat[ind, None] * self._Z[ind, :], axis=0)
                    for ind in self._ind_S
                ],
                axis=0,
            )

        x12 = (xbar - mu_x) / sigma2_x
        V1 = self._n * np.array(
            [
                [1 / sigma2_x, x12],
                [x12, 0.5 * (vbar - mu_x**2) / sigma2_x],
            ]
        )

        temp_mul = self._n_s / sigma2_w
        A = np.diag(temp_mul)
        B = np.diag(temp_mul * xbar_s)
        C = np.diag(temp_mul * (self._wbar_s - a - b * xbar_s))
        D = np.diag(temp_mul * vbar_s)
        E = np.diag(temp_mul * (wxbar_s - a * xbar_s - b * vbar_s))
        F = np.diag(
            0.5
            * temp_mul
            * (
                self._wwbar_s
                + a**2
                + b**2 * vbar_s
                - 2 * a * self._wbar_s
                - 2 * b * wxbar_s
                + 2 * a * b * xbar_s
            )
        )
        V2 = np.block([[A, B, C], [B, D, E], [C, E, F]])

        sigma2_y_long = sigma2_y[self._ind_inv]
        temp_mul = self._n_s / sigma2_y
        if self._Z is not None:
            B = np.sum(self._n_s[None, :] * xzbar_s, axis=0, keepdims=True)
            E = temp_mul[None, :] * self._zbar_s
            G = (self._Z.T * sigma2_y_long) @ self._Z
            H = temp_mul * (
                self._yzbar_s.T
                - beta_0 * self._zbar_s.T
                - beta_x * xzbar_s.T
                - (self._zzbar_s @ beta_z).T
            )

            dz2 = np.sum(
                beta_z * beta_z[:, None] * self._zzbar_s,
                axis=(1, 2),
            )
            J_zpart = (
                dz2
                - 2 * self._yzbar_s @ beta_z
                + 2 * beta_0 * self._zbar_s @ beta_z
                + 2 * xzbar_s * beta_z * beta_x
            )
            C_zpart = -xzbar @ beta_z
            F_zpart = -self._zbar_s @ beta_z
        else:
            J_zpart = 0.0
            C_zpart = 0.0
            F_zpart = 0.0
        K = np.array([[np.sum(temp_mul * vbar_s)]])
        A = (temp_mul * xbar_s)[None, :]
        C = (temp_mul * (xybar_s - beta_0 * xbar_s - beta_x * vbar_s + C_zpart))[
            None, :
        ]
        D = np.diag(temp_mul)
        F = np.diag(temp_mul * (self._ybar_s - beta_0 - beta_x * xbar_s + F_zpart))
        J = np.diag(
            0.5
            * temp_mul
            * (
                self._yybar_s
                + beta_0**2
                + beta_x**2 * vbar_s
                - 2 * beta_0 * self._ybar_s
                - 2 * beta_x * xybar_s
                + 2 * beta_0 * beta_x * xbar_s
                + J_zpart
            )
        )
        V3 = [[K, A, C], [A.T, D, F], [C.T, F.T, J]]
        if self._Z is not None:
            V3[0].insert(2, B)
            V3[1].insert(2, E)
            V3[2].insert(2, H.T)
            V3.insert(2, [B.T, E.T, G, H])
        V3 = np.block(V3)

        return sc_block_diag(np.linalg.inv(V1), np.linalg.inv(V2), np.linalg.inv(V3))
