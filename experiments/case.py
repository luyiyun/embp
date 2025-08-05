from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

from embp import EMBP


def method_xonly(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray | None,
    type_outcome: Literal["binary", "continue"],
) -> np.ndarray:
    notnone = ~pd.isnull(X)
    X, Y = X[notnone], Y[notnone]
    if Z is not None:
        X = np.concatenate([X[:, None], Z[notnone]], axis=1)
    X = sm.add_constant(X)
    if type_outcome == "continue":
        model = sm.OLS(Y, X)
    else:
        model = sm.GLM(Y, X, sm.families.Binomial())
    res = model.fit()
    return np.r_[res.params[1], res.conf_int()[1, :]]


def main():
    files = [
        "./data/BRCA1_simu_all.csv",
        "./data/BRCA1_simu_0.1.csv",
        "./data/BRCA1_simu_0.2.csv",
    ]

    for fn in files:
        print("=" * 20 + f"Analyzing {fn}" + "=" * 20)

        dat = pd.read_csv(fn, index_col=0)
        X = dat["X"].values
        S = dat["S"].values
        W = dat["W"].values
        Y = dat["Y"].values
        Z = None

        #
        resi = method_xonly(X, Y, Z, "binary")
        print(f"Xonly method, beta_x={resi[0]:.3f} ({resi[1]:.3f}-{resi[2]:.3f})")

        if fn.endswith("all.csv"):
            continue

        # fit model
        embp_kwargs = {
            "ci": True,
            "ci_method": "bootstrap",
            "pbar": True,
            "max_iter": None,
            "seed": 0,
            "n_bootstrap": 200,
            "gem": False,
            "quasi_mc_K": 100,
            "delta2": None,
            "binary_solve": "lap",
            "device": "cpu",
            "importance_sampling_maxK": 5000,
            "bootstrap_init_disturb": 0.01,
        }
        estimator = EMBP("binary", **embp_kwargs)
        estimator.fit(X, S, W, Y, Z)
        res_beta_x = estimator.params_.loc["beta_x", :].values
        print(
            f"EMBP method, beta_x={res_beta_x[0]:.3f} ({res_beta_x[1]:.3f}-{res_beta_x[2]:.3f})"
        )
        # print(estimator.params_)
        # print(estimator.res_bootstrap_)


if __name__ == "__main__":
    main()
