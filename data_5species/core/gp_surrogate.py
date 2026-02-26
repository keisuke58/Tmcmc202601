from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float, variance: float) -> np.ndarray:
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    diff = x1[:, None, :] - x2[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    return variance * np.exp(-0.5 * sqdist / (length_scale**2))


@dataclass
class GPSurrogate:
    x_train: np.ndarray
    y_train: np.ndarray
    length_scale: float
    variance: float
    noise: float
    _k_inv: np.ndarray

    @classmethod
    def fit(
        cls,
        x_train: np.ndarray,
        y_train: np.ndarray,
        length_scale: float = 1.0,
        variance: float = 1.0,
        noise: float = 1e-6,
    ) -> GPSurrogate:
        x_train = np.asarray(x_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        k = _rbf_kernel(x_train, x_train, length_scale, variance)
        n = k.shape[0]
        k[np.diag_indices(n)] += noise
        k_inv = np.linalg.inv(k)
        return cls(
            x_train=x_train,
            y_train=y_train,
            length_scale=length_scale,
            variance=variance,
            noise=noise,
            _k_inv=k_inv,
        )

    def predict(self, x_test: np.ndarray, return_std: bool = False):
        x_test = np.asarray(x_test, dtype=np.float64)
        k_star = _rbf_kernel(x_test, self.x_train, self.length_scale, self.variance)
        mean = k_star @ (self._k_inv @ self.y_train)
        if not return_std:
            return mean
        k_test = _rbf_kernel(x_test, x_test, self.length_scale, self.variance)
        cov = k_test - k_star @ self._k_inv @ k_star.T
        var = np.clip(np.diag(cov), a_min=0.0, a_max=None)
        std = np.sqrt(var)
        return mean, std


def build_maxstress_surrogate(theta: np.ndarray, responses: np.ndarray) -> GPSurrogate:
    return GPSurrogate.fit(theta, responses)


def surrogate_log_likelihood(
    gp: GPSurrogate,
    theta: np.ndarray,
    y_obs: float,
    sigma_obs: float,
) -> float:
    mu, std = gp.predict(theta.reshape(1, -1), return_std=True)
    var_total = std[0] ** 2 + sigma_obs**2
    if not np.isfinite(var_total) or var_total <= 0.0:
        return -1e20
    resid2 = (y_obs - mu[0]) ** 2
    return -0.5 * np.log(2.0 * np.pi * var_total) - 0.5 * resid2 / var_total
