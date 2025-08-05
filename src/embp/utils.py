from typing import Tuple
from time import perf_counter

import numpy as np
from numpy import ndarray
from scipy.special import expit
from scipy.linalg import lstsq

from .logger import logger_embp
from scipy.optimize import minimize


def batch_nonzero(mask):
    if mask.ndim == 1:
        return np.nonzero(mask)[0]
    else:
        return np.arange(mask.shape[0])[:, None], np.stack(
            [np.nonzero(mask[i])[0] for i in range(mask.shape[0])],
        )


def ols(X: ndarray, Y: ndarray, Z: ndarray | None = None) -> Tuple[ndarray, ndarray]:
    X_des = np.stack([X, np.ones_like(X)], axis=-1)
    if Z is not None:
        X_des = np.concatenate([X_des, Z], axis=-1)

    beta, resid, _, _ = lstsq(X_des, Y)
    return beta, resid


def logistic(
    X: ndarray,
    Y: ndarray,
    Z: ndarray | None = None,
    lr: float = 1.0,
    delta1: float = 1e-3,
    delta2: float = 1e-7,
    max_iter: int = 100,
) -> ndarray:
    X_des = np.stack([X, np.ones_like(X)], axis=-1)
    if Z is not None:
        X_des = np.concatenate([X_des, Z], axis=-1)

    beta_ = np.zeros(X_des.shape[1])
    for i in range(max_iter):
        p = expit(X_des @ beta_)
        grad = X_des.T @ (p - Y)
        hess = np.einsum("ij,i,ik->jk", X_des, p * (1 - p), X_des)

        beta_delta = lr * np.linalg.solve(hess, grad)
        beta_ -= beta_delta

        rdiff = np.max(np.abs(beta_delta) / (np.abs(beta_) + delta1))
        logger_embp.info(f"Init step Newton-Raphson: iter={i + 1} diff={rdiff:.4f}")
        if rdiff < delta2:
            break
    else:
        logger_embp.warning(
            f"Init Newton-Raphson (max_iter={max_iter}) doesn't converge"
        )

    return beta_


def loss_grad(beta, X, y):
    p = expit(X @ beta)
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    loss = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    grad = X.T @ (p - y)
    return loss, grad


def logistic_bfgs(
    X: ndarray,
    Y: ndarray,
    Z: ndarray | None = None,
    lr: float = 1.0,
    delta1: float = 1e-3,
    delta2: float = 1e-7,
    max_iter: int = 100,
) -> ndarray:
    X_des = np.stack([X, np.ones_like(X)], axis=-1)
    if Z is not None:
        X_des = np.concatenate([X_des, Z], axis=-1)

    beta_ = np.zeros(X_des.shape[1])

    result = minimize(
        fun=loss_grad,  # ✅ 目标函数+梯度
        x0=beta_,  # ✅ 初始值
        args=(X_des, Y),
        jac=True,  # ✅ 目标函数的一阶导（梯度）
        method="BFGS",  # 拟牛顿法
        # options={"disp": True}
    )
    beta_ = result.x

    return beta_


def logistic_bayes(
    X: ndarray,
    Y: ndarray,
    Z: ndarray | None = None,
    lr: float = 1.0,
    delta1: float = 1e-3,
    delta2: float = 1e-7,
    max_iter: int = 100,
) -> ndarray:
    X_des = np.stack([X, np.ones_like(X)], axis=-1)
    if Z is not None:
        X_des = np.concatenate([X_des, Z], axis=-1)

    beta_ = np.zeros(X_des.shape[1])
    alpha = 1
    for i in range(max_iter):
        p = expit(X_des @ beta_)
        grad = X_des.T @ (p - Y) + alpha * beta_
        hess = np.einsum("ij,i,ik->jk", X_des, p * (1 - p), X_des)
        hess += alpha * np.eye(hess.shape[0])

        beta_delta = lr * np.linalg.solve(hess, grad)
        beta_ -= beta_delta

        rdiff = np.max(np.abs(beta_delta) / (np.abs(beta_) + delta1))
        logger_embp.info(f"Init step Newton-Raphson: iter={i + 1} diff={rdiff:.4f}")
        if rdiff < delta2:
            break
    else:
        logger_embp.warning(
            f"Init Newton-Raphson (max_iter={max_iter}) doesn't converge"
        )

    return beta_


class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *args):
        self.end = perf_counter()
        self.interval = self.end - self.start
