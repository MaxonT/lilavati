"""
Financial Machine Learning Core Engine.

Implements financial ML primitives from López de Prado (2018), *Advances in Financial Machine
Learning* (AFML):
1. Fractional differentiation with a fixed-width window (FFD, AFML ch. 5): stationarise a series
   while keeping as much memory as possible.
2. The triple-barrier method (AFML ch. 3): path-dependent labels from take-profit, stop-loss and
   expiration barriers.

Purged and embargoed cross-validation (AFML ch. 7 and 12) and the backtest-overfitting statistics are in
``lilavati.core.backtest_overfitting``.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from lilavati.core.econometrics import EconometricEngine

# Minimum number of FFD output points for a d candidate to be ADF-tested.
_MIN_FFD_POINTS = 25
# Absolute tolerance on returns when testing a barrier touch, so a move that lands exactly on a
# barrier (e.g. 100 -> 110 with a 10% take-profit) counts despite floating-point rounding.
_BARRIER_RETURN_TOL = 1e-12


class FractionalDiffResult(BaseModel):
    """Result of optimal fractional differentiation."""

    model_config = ConfigDict(frozen=True)

    optimal_d: float = Field(
        ...,
        description=(
            "Smallest d on the search grid {0, step, 2*step, ..., 1} whose FFD series has "
            "ADF p-value <= target_p_value"
        ),
    )
    adf_p_value: float = Field(
        ..., description="ADF p-value of the fractionally differentiated series at optimal_d"
    )
    correlation_with_original: float = Field(
        ...,
        description=(
            "Pearson correlation between the FFD series and the original series at the same "
            "time indices (the last len(differentiated_series) input points)"
        ),
    )
    differentiated_series: list[float] = Field(
        ...,
        description=(
            "FFD series at optimal_d; element j is aligned with input index "
            "len(series) - len(differentiated_series) + j"
        ),
    )


class TripleBarrierLabel(BaseModel):
    """A path-dependent label generated via the Triple-Barrier Method."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(..., description="Index of the entry bar")
    label: int = Field(
        ..., description="1 for take-profit, -1 for stop-loss, 0 for vertical expiration"
    )
    touch_index: int = Field(
        ...,
        description="Index of the first barrier touch (entry index + holding_horizon if expired)",
    )
    return_at_touch: float = Field(
        ...,
        description="Simple return (P[touch] - P[entry]) / P[entry] as a decimal",
    )


def _finite_1d(series: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values; clean or fill them first.")
    return arr


class FinancialMLEngine:
    """Financial machine learning methods (López de Prado, 2018)."""

    @staticmethod
    def get_fractional_weights(d: float, size: int, threshold: float = 1e-4) -> np.ndarray:
        """
        Fixed-width-window weights for fractional differencing (AFML snippet 5.3).

        w_0 = 1 and w_k = -w_{k-1} * (d - k + 1) / k, the coefficients of (1 - B)^d. Weights are
        generated until |w_k| < threshold (that weight is dropped) or `size` weights exist.

        Returned order is oldest lag first: result[-1] = w_0 = 1 multiplies the current value
        and result[0] multiplies the oldest value in the window, so
        np.dot(result, x[t - width + 1 : t + 1]) = sum_k w_k * x_{t-k}.
        d = 0 gives [1]; d = 1 gives [-1, 1] (first difference).
        """
        if not math.isfinite(d):
            raise ValueError("d must be finite.")
        if size < 1:
            raise ValueError("size must be at least 1.")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and non-negative.")
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            w.append(w_k)
        return np.array(w[::-1], dtype=np.float64)

    @staticmethod
    def fractionally_differentiate(
        series: Sequence[float],
        d: float,
        threshold: float = 1e-4,
    ) -> np.ndarray:
        """
        Fixed-width-window fractional differencing (FFD, AFML snippet 5.3): (1 - B)^d X_t with
        the weight series truncated at |w_k| < threshold.

        With width = len(get_fractional_weights(d, len(series), threshold)), the output has
        len(series) - width + 1 points; output[j] corresponds to series[width - 1 + j] (the first
        width - 1 points have an incomplete window and are dropped). Units are those of the input;
        AFML applies FFD to log prices, which callers must take themselves. Non-finite inputs raise
        ValueError (AFML forward-fills them instead).
        """
        arr = _finite_1d(series, "series")
        n = len(arr)
        weights = FinancialMLEngine.get_fractional_weights(d, size=n, threshold=threshold)
        width = len(weights)

        res = []
        for i in range(width, n + 1):
            window = arr[i - width : i]
            res.append(np.dot(weights, window))

        return np.asarray(res, dtype=np.float64)

    @staticmethod
    def find_optimal_fractional_d(
        series: Sequence[float],
        step: float = 0.05,
        target_p_value: float = 0.01,
    ) -> FractionalDiffResult:
        """
        Smallest d on the grid {0, step, 2*step, ..., 1} (1 is always included) whose FFD series
        (threshold 1e-4) has ADF p-value <= target_p_value (AFML section 5.6, snippet 5.4).

        ADF: `EconometricEngine.test_stationarity` (statsmodels `adfuller`, constant only,
        lag order by AIC); AFML snippet 5.4 uses maxlag=1 without autolag, so p-values differ.
        d = 0 means the input itself meets the target. Candidates whose FFD output has fewer than
        25 points are skipped, so on short series small d values are never tested. Raises
        ValueError if no grid point meets the target (instead of reporting a non-solution).
        correlation_with_original is the Pearson correlation between the FFD series and the input
        at the same time indices.
        """
        arr = _finite_1d(series, "series")
        if not math.isfinite(step) or not 0.0 < step <= 1.0:
            raise ValueError("step must be in (0, 1].")
        if not math.isfinite(target_p_value) or not 0.0 < target_p_value < 1.0:
            raise ValueError("target_p_value must be in (0, 1).")

        # d = 0: the series as given. Use the p-value only; is_stationary uses its own 5% level.
        initial_check = EconometricEngine.test_stationarity(arr.tolist())
        if initial_check.p_value <= target_p_value:
            return FractionalDiffResult(
                optimal_d=0.0,
                adf_p_value=float(initial_check.p_value),
                correlation_with_original=1.0,
                differentiated_series=arr.tolist(),
            )

        # Grid k*step for k = 1..floor(1/step), plus d = 1 if step does not divide 1. Rounding
        # removes float noise (0.15000000000000002 -> 0.15) without changing grid points.
        n_steps = math.floor(1.0 / step + 1e-9)
        grid = [round(k * step, 12) for k in range(1, n_steps + 1)]
        if not grid or grid[-1] < 1.0 - 1e-12:
            grid.append(1.0)

        best_p_tested = math.inf
        for d_candidate in grid:
            diff = FinancialMLEngine.fractionally_differentiate(arr.tolist(), d=d_candidate)
            if len(diff) < _MIN_FFD_POINTS:
                continue
            p_val = float(EconometricEngine.test_stationarity(diff.tolist()).p_value)
            best_p_tested = min(best_p_tested, p_val)
            if p_val <= target_p_value:
                # FFD output ends at the last input point, so align on the tail.
                aligned = arr[-len(diff) :]
                corr = float(np.corrcoef(aligned, diff)[0, 1])
                return FractionalDiffResult(
                    optimal_d=d_candidate,
                    adf_p_value=p_val,
                    correlation_with_original=corr,
                    differentiated_series=diff.tolist(),
                )

        if math.isinf(best_p_tested):
            raise ValueError(
                f"Series too short: no d on the grid leaves {_MIN_FFD_POINTS} FFD points."
            )
        raise ValueError(
            f"No d in [0, 1] (step {step}) reaches ADF p-value <= {target_p_value}; "
            f"smallest p-value tested was {best_p_tested:.4g}."
        )

    @staticmethod
    def apply_triple_barriers(
        prices: Sequence[float],
        holding_horizon: int,
        take_profit_pct: float,
        stop_loss_pct: float,
    ) -> list[TripleBarrierLabel]:
        """
        Triple-barrier labels (AFML section 3.4, snippets 3.2 and 3.5) with fixed return barriers.

        For each entry bar i with a complete window (i = 0..len(prices) - holding_horizon - 1),
        bars j = i+1..i+holding_horizon are scanned for the first touch of
        - upper barrier: return (P[j] - P[i]) / P[i] >= take_profit_pct  -> label +1
        - lower barrier: return <= -stop_loss_pct                         -> label -1
        and if neither is touched the vertical barrier at i + holding_horizon gives label 0 with
        the return at that bar (AFML section 3.4; snippet 3.5 instead labels a vertical touch by
        the sign of its return). The entry bar itself is not tested. Barriers are decimal simple
        returns (0.03 = 3%); for AFML's volatility-scaled barriers pass multiples of the
        volatility target. Touches are inclusive with a 1e-12 return tolerance (AFML uses strict
        inequalities). Because both barriers are tested on one price per bar and both are
        strictly positive, a bar cannot touch both. Entries whose vertical barrier would fall
        after the last bar are dropped (as AFML getBins drops events without t1). Pass
        math.inf for take_profit_pct to disable the upper barrier; stop_loss_pct >= 1 disables
        the lower one for positive prices. (In AFML a zero multiplier disables a barrier; here
        zero is rejected.)
        """
        p = _finite_1d(prices, "prices")
        if np.any(p <= 0):
            raise ValueError("prices must be strictly positive.")
        try:
            if isinstance(holding_horizon, bool):
                raise TypeError
            holding_horizon = operator.index(holding_horizon)
        except TypeError:
            raise ValueError("holding_horizon must be an integer number of bars.") from None
        if holding_horizon < 1:
            raise ValueError("holding_horizon must be at least 1 bar.")
        if math.isnan(take_profit_pct) or take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive (math.inf disables the barrier).")
        if math.isnan(stop_loss_pct) or stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive (>= 1 disables the barrier).")

        n = len(p)
        upper = take_profit_pct - _BARRIER_RETURN_TOL
        lower = -stop_loss_pct + _BARRIER_RETURN_TOL
        labels: list[TripleBarrierLabel] = []

        for i in range(n - holding_horizon):
            entry_price = p[i]
            label_val = 0
            touch_idx = i + holding_horizon

            for j in range(i + 1, i + holding_horizon + 1):
                ret = (p[j] - entry_price) / entry_price
                if ret >= upper:
                    label_val = 1
                    touch_idx = j
                    break
                if ret <= lower:
                    label_val = -1
                    touch_idx = j
                    break

            realized_return = (p[touch_idx] - entry_price) / entry_price
            labels.append(
                TripleBarrierLabel(
                    index=i,
                    label=label_val,
                    touch_index=touch_idx,
                    return_at_touch=float(realized_return),
                )
            )

        return labels
