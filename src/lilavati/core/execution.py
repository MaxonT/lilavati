"""
Trading Microstructure, Optimal Execution, and Portfolio Optimization Engine.

Implements:
1. Limit Order Book (LOB) metrics: Microprice, Spread, and Order Flow Imbalance (OFI).
2. The Almgren-Chriss (2000) optimal liquidation trajectory framework.
3. Black-Litterman asset allocation combining market equilibrium priors with agent views.
4. Hierarchical Risk Parity (HRP) using tree clustering without matrix inversion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cvxpy as cp
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from lilavati.schemas.execution import (
    AlmgrenChrissResult,
    BlackLittermanResult,
    HRPAllocationResult,
    LOBMetrics,
)


class ExecutionEngine:
    """Deterministic analytics for execution scheduling, microstructure, and portfolio optimization."""

    @staticmethod
    def analyze_lob_snapshot(
        best_bid: float,
        best_ask: float,
        bid_size: float,
        ask_size: float,
        prev_bid_size: float | None = None,
        prev_ask_size: float | None = None,
    ) -> LOBMetrics:
        """
        Computes microstructure metrics from an L1/L2 limit order book quote.
        Microprice = (bid_size * ask_price + ask_size * bid_price) / (bid_size + ask_size)
        """
        if best_bid >= best_ask:
            raise ValueError(
                f"Crossed or locked book: best_bid ({best_bid}) >= best_ask ({best_ask})"
            )
        if bid_size <= 0 or ask_size <= 0:
            raise ValueError("Bid and ask sizes must be positive.")

        spread = best_ask - best_bid
        mid_price = (best_ask + best_bid) / 2.0
        micro_price = (bid_size * best_ask + ask_size * best_bid) / (bid_size + ask_size)

        # Simplified Order Flow Imbalance (OFI) proxy
        ofi = 0.0
        if prev_bid_size is not None and prev_ask_size is not None:
            delta_bid = bid_size - prev_bid_size
            delta_ask = ask_size - prev_ask_size
            ofi = delta_bid - delta_ask

        return LOBMetrics(
            best_bid=float(best_bid),
            best_ask=float(best_ask),
            bid_size=float(bid_size),
            ask_size=float(ask_size),
            spread=float(spread),
            mid_price=float(mid_price),
            micro_price=float(micro_price),
            order_flow_imbalance=float(ofi),
        )

    @staticmethod
    def compute_almgren_chriss_schedule(
        total_shares: float,
        time_horizon_hours: float,
        n_intervals: int,
        daily_volatility: float,
        *,
        arrival_price: float,
        permanent_impact_gamma: float = 2.5e-7,
        temporary_impact_eta: float = 2.5e-6,
        risk_aversion_lambda: float = 1e-6,
        hours_per_day: float = 6.5,
        fixed_cost_epsilon: float = 0.0,
    ) -> AlmgrenChrissResult:
        """
        Discrete-time Almgren-Chriss optimal liquidation trajectory.

        Reference: Almgren, R. and Chriss, N. (2000), "Optimal Execution of Portfolio
        Transactions", Journal of Risk 3(2), 5-39. Equation numbers below are from the
        December 2000 version of the paper.

        Units follow the paper (Table 1), with the trading day as the time unit:

        * ``daily_volatility`` is a fraction (e.g. 0.019). It is converted to price units,
          sigma = daily_volatility * arrival_price, in ($/share)/day^(1/2).
        * ``permanent_impact_gamma`` is in $/share^2.
        * ``temporary_impact_eta`` is in ($/share)/(share/day).
        * ``fixed_cost_epsilon`` is in $/share (e.g. half the bid-ask spread).
        * ``risk_aversion_lambda`` is in 1/$.
        * ``time_horizon_hours`` is converted to days with ``hours_per_day``.

        With tau = T/N and eta_tilde = eta - gamma*tau/2 (eq. 8), kappa solves
        (2/tau^2)(cosh(kappa*tau) - 1) = lambda*sigma^2/eta_tilde (eq. 16), and the holdings are
        x_j = sinh(kappa(T - t_j)) / sinh(kappa*T) * X (eq. 17).

        The returned cost and variance are evaluated on the trajectory:
        E = gamma*X^2/2 + epsilon*sum|n_k| + (eta_tilde/tau)*sum n_k^2 (eq. 8) and
        V = sigma^2 * sum tau*x_k^2 (eq. 5), both in dollars (V in $^2).
        """
        if n_intervals < 1:
            raise ValueError("n_intervals must be at least 1.")
        if time_horizon_hours <= 0.0 or hours_per_day <= 0.0:
            raise ValueError("time_horizon_hours and hours_per_day must be positive.")
        if arrival_price <= 0.0:
            raise ValueError("arrival_price must be positive.")
        if daily_volatility < 0.0:
            raise ValueError("daily_volatility must be non-negative.")
        if risk_aversion_lambda < 0.0:
            raise ValueError("risk_aversion_lambda must be non-negative.")
        if temporary_impact_eta <= 0.0:
            raise ValueError("temporary_impact_eta must be positive.")

        x0 = float(total_shares)
        n = int(n_intervals)
        t_days = time_horizon_hours / hours_per_day
        tau = t_days / n

        sigma = daily_volatility * arrival_price  # ($/share)/day^(1/2)
        sigma2 = sigma * sigma
        gamma = permanent_impact_gamma
        eta = temporary_impact_eta
        lam = risk_aversion_lambda

        eta_tilde = eta - 0.5 * gamma * tau
        if eta_tilde <= 0.0:
            raise ValueError(
                f"eta_tilde = eta - gamma*tau/2 = {eta_tilde:.6g} <= 0: the expected cost is not "
                "strictly convex (Almgren-Chriss 2000, eq. 8). Use more intervals or a smaller "
                "permanent impact gamma."
            )

        # (2/tau^2)(cosh(kappa*tau) - 1) = kappa_tilde^2. Since cosh(y) - 1 = 2 sinh^2(y/2),
        # kappa = (2/tau) * asinh(kappa_tilde*tau/2), which is accurate for small and large kappa.
        kappa_tilde = math.sqrt(lam * sigma2 / eta_tilde)
        kappa = (2.0 / tau) * math.asinh(0.5 * kappa_tilde * tau)

        # x_j / X = sinh(kappa(T - t_j)) / sinh(kappa T), evaluated without overflow as
        # exp(-kappa t_j) * (1 - exp(-2 kappa (T - t_j))) / (1 - exp(-2 kappa T)).
        trajectory: list[float] = [x0]
        if kappa * t_days == 0.0:
            # lambda = 0 (or sigma = 0): the limit kappa -> 0 is the linear (TWAP) schedule.
            trajectory.extend(x0 * (n - j) / n for j in range(1, n + 1))
        else:
            denom = -math.expm1(-2.0 * kappa * t_days)
            for j in range(1, n + 1):
                t_j = j * tau
                t_rem = (n - j) * tau
                ratio = math.exp(-kappa * t_j) * (-math.expm1(-2.0 * kappa * t_rem)) / denom
                trajectory.append(x0 * ratio)
        trajectory[-1] = 0.0

        trade_sizes = [trajectory[j - 1] - trajectory[j] for j in range(1, n + 1)]

        expected_cost = (
            0.5 * gamma * x0 * x0
            + fixed_cost_epsilon * sum(abs(nk) for nk in trade_sizes)
            + (eta_tilde / tau) * sum(nk * nk for nk in trade_sizes)
        )
        variance = sigma2 * tau * sum(xk * xk for xk in trajectory[1:])

        half_life_hours = hours_per_day / kappa if kappa > 0.0 else float("inf")

        return AlmgrenChrissResult(
            total_shares=total_shares,
            time_horizon_hours=time_horizon_hours,
            n_intervals=n_intervals,
            trading_trajectory=trajectory,
            trade_sizes=trade_sizes,
            expected_shortfall=float(expected_cost),
            shortfall_variance=float(variance),
            half_life_hours=float(half_life_hours),
        )

    @staticmethod
    def optimize_black_litterman(
        assets: list[str],
        covariance_matrix: Sequence[Sequence[float]],
        market_prior_weights: Sequence[float],
        views_matrix: Sequence[Sequence[float]],
        views_returns: Sequence[float],
        views_confidence_std: Sequence[float],
        risk_aversion: float = 2.5,
        tau: float = 0.05,
    ) -> BlackLittermanResult:
        """
        Solves the Black-Litterman model fusing market equilibrium priors with views.
        E[R] = [ (tau*Sigma)^(-1) + P^T Omega^(-1) P ]^(-1) * [ (tau*Sigma)^(-1) Pi + P^T Omega^(-1) Q ]
        """
        n = len(assets)
        sigma = np.asarray(covariance_matrix, dtype=np.float64)
        w_mkt = np.asarray(market_prior_weights, dtype=np.float64)
        p = np.asarray(views_matrix, dtype=np.float64)
        q = np.asarray(views_returns, dtype=np.float64)
        omega_diag = np.asarray(views_confidence_std, dtype=np.float64) ** 2
        omega = np.diag(omega_diag)

        # 1. Reverse-engineer implied equilibrium returns (Pi = delta * Sigma * w_mkt)
        pi = risk_aversion * (sigma @ w_mkt)

        # 2. Black-Litterman posterior return calculation
        tau_sigma_inv = np.linalg.inv(tau * sigma)
        omega_inv = np.linalg.inv(omega)

        post_cov_inv = tau_sigma_inv + p.T @ omega_inv @ p
        post_cov = np.linalg.inv(post_cov_inv)
        post_er = post_cov @ (tau_sigma_inv @ pi + p.T @ omega_inv @ q)

        # 3. Solve for constrained optimal weights (Long-only, sum to 1.0) using CVXPY
        w_var = cp.Variable(n)
        port_ret = post_er.T @ w_var
        port_risk = cp.quad_form(w_var, sigma)  # type: ignore[attr-defined]
        objective = cp.Maximize(port_ret - 0.5 * risk_aversion * port_risk)
        constraints = [cp.sum(w_var) == 1.0, w_var >= 0.0]  # type: ignore[attr-defined]

        problem = cp.Problem(objective, constraints)
        problem.solve()  # type: ignore[no-untyped-call]

        if w_var.value is not None:
            optimal_weights = [max(0.0, float(wi)) for wi in w_var.value]
        else:
            optimal_weights = [1.0 / n] * n

        # Renormalize to exact 1.0 sum
        s = sum(optimal_weights)
        optimal_weights = [w / s for w in optimal_weights]

        return BlackLittermanResult(
            assets=assets,
            market_prior_weights=[float(w) for w in w_mkt],
            optimal_weights=optimal_weights,
            expected_returns=[float(r) for r in post_er],
            implied_prior_returns=[float(r) for r in pi],
        )

    @staticmethod
    def optimize_hierarchical_risk_parity(
        assets: list[str],
        returns_matrix: Sequence[Sequence[float]],
    ) -> HRPAllocationResult:
        """
        Solves Hierarchical Risk Parity (HRP) portfolio optimization (Marcos López de Prado 2016).
        Does not require matrix inversion; stable under high correlation and collinearity.
        """
        r_arr = np.asarray(returns_matrix, dtype=np.float64)
        cov = np.cov(r_arr, rowvar=False)
        corr = np.corrcoef(r_arr, rowvar=False)

        n = len(assets)
        if n == 1:
            return HRPAllocationResult(assets=assets, weights=[1.0], ordered_indices=[0])

        # Step 1: Distance matrix D_ij = sqrt(0.5 * (1 - rho_ij))
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        # Step 2: Hierarchical tree clustering
        dist_condensed = squareform(dist)
        link = linkage(dist_condensed, method="single")

        # Step 3: Quasi-diagonalization
        ordered_idx = list(leaves_list(link))

        # Step 4: Recursive Bisection
        weights = np.ones(n, dtype=np.float64)
        clusters = [ordered_idx]

        while len(clusters) > 0:
            new_clusters = []
            for c in clusters:
                if len(c) > 1:
                    mid = len(c) // 2
                    c1, c2 = c[:mid], c[mid:]

                    cov1 = cov[np.ix_(c1, c1)]
                    cov2 = cov[np.ix_(c2, c2)]

                    # Inverse-variance allocation between the two clusters
                    inv_v1 = 1.0 / np.trace(cov1)
                    inv_v2 = 1.0 / np.trace(cov2)
                    alpha1 = inv_v1 / (inv_v1 + inv_v2)
                    alpha2 = 1.0 - alpha1

                    weights[c1] *= alpha1
                    weights[c2] *= alpha2

                    new_clusters.extend([c1, c2])
            clusters = new_clusters

        # Normalize weights
        weights = weights / np.sum(weights)

        return HRPAllocationResult(
            assets=assets,
            weights=[float(w) for w in weights],
            ordered_indices=ordered_idx,
        )
