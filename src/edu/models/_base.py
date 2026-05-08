"""Shared interface for parametric demand and discount models.

Every model in :mod:`edu.models` exposes the same four methods:

``value(params, x)``
    Predicted dependent variable (Q for demand, SV for discounting).
``log_likelihood(params, x, y)``
    Gaussian log-likelihood of observed ``y`` given predicted ``value``.
``negative_log_likelihood(params, x, y)``
    Convenience wrapper for minimisation.
``fit(x, y, ...)``
    Nonlinear least-squares fit; returns a :class:`FitResult`.

The shared interface lets us swap models freely in cross-model comparison
(Phase 1 notebook 02) and keeps the unified-model code in Phase 2 short.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class FitResult:
    """Outcome of a single nonlinear least-squares fit.

    Attributes
    ----------
    params : dict[str, float]
        Estimated parameter values keyed by name.
    stderr : dict[str, float]
        One-sigma standard errors from the covariance matrix. ``nan`` when
        the covariance could not be estimated.
    aic, bic : float
        Information criteria computed from the residual sum of squares under
        a Gaussian-noise assumption.
    rss, r_squared : float
        Residual sum of squares and coefficient of determination.
    n_obs : int
        Number of observations used in the fit.
    success : bool
        Whether the optimiser reported convergence.
    message : str
        Optimiser status message.
    extra : dict[str, Any]
        Backend-specific extras (e.g. lmfit ``MinimizerResult``).
    """

    params: dict[str, float]
    stderr: dict[str, float]
    aic: float
    bic: float
    rss: float
    r_squared: float
    n_obs: int
    success: bool
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class ParametricModel(ABC):
    """Abstract base for all demand and discount models.

    Subclasses set :attr:`param_names` (declaration order = canonical order
    used by every method that takes a positional parameter vector) and
    implement :meth:`_value`. The base class provides the shared
    log-likelihood, NLS fitting loop, and information-criterion utilities.
    """

    #: Canonical parameter order. Subclasses override.
    param_names: ClassVar[tuple[str, ...]] = ()

    # ---- core evaluation ----------------------------------------------------

    @abstractmethod
    def _value(self, params: FloatArray, x: FloatArray) -> FloatArray:
        """Predicted response for parameter vector ``params`` at points ``x``.

        Subclasses must vectorise over ``x``; never loop.
        """

    def value(self, params: dict[str, float] | FloatArray, x: ArrayLike) -> FloatArray:
        """Public entry point. Accepts a dict or array of parameters."""
        p = self._coerce_params(params)
        x_arr = np.asarray(x, dtype=float)
        return self._value(p, x_arr)

    # ---- likelihood ---------------------------------------------------------

    def log_likelihood(
        self,
        params: dict[str, float] | FloatArray,
        x: ArrayLike,
        y: ArrayLike,
        sigma: float | None = None,
    ) -> float:
        """Gaussian log-likelihood under iid noise with std ``sigma``.

        When ``sigma`` is ``None``, the MLE estimate ``sqrt(rss / n)`` is used,
        which yields a profile likelihood comparable across models with the
        same dependent variable.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        pred = self.value(params, x_arr)
        resid = y_arr - pred
        n = y_arr.size
        if sigma is None:
            sigma = float(np.sqrt(np.sum(resid**2) / n))
        sigma = max(sigma, 1e-12)
        ll = -0.5 * n * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(resid**2) / sigma**2
        return float(ll)

    def negative_log_likelihood(
        self,
        params: dict[str, float] | FloatArray,
        x: ArrayLike,
        y: ArrayLike,
        sigma: float | None = None,
    ) -> float:
        return -self.log_likelihood(params, x, y, sigma=sigma)

    # ---- helpers ------------------------------------------------------------

    def _coerce_params(self, params: dict[str, float] | FloatArray) -> FloatArray:
        if isinstance(params, dict):
            try:
                return np.array([params[name] for name in self.param_names], dtype=float)
            except KeyError as exc:
                missing = set(self.param_names) - params.keys()
                msg = f"Missing parameters: {sorted(missing)}"
                raise ValueError(msg) from exc
        arr = np.asarray(params, dtype=float)
        if arr.shape != (len(self.param_names),):
            msg = (
                f"Parameter vector has shape {arr.shape}; expected "
                f"({len(self.param_names)},) for params {self.param_names}"
            )
            raise ValueError(msg)
        return arr

    @staticmethod
    def _information_criteria(rss: float, n: int, k: int) -> tuple[float, float]:
        """AIC and BIC under Gaussian noise with profiled variance.

        Returns ``(aic, bic)``. The constant terms cancel in model comparison.
        """
        if rss <= 0 or n <= 0:
            return float("nan"), float("nan")
        ll_max = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
        aic = 2 * k - 2 * ll_max
        bic = k * np.log(n) - 2 * ll_max
        return float(aic), float(bic)
