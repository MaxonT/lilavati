"""
Canonical Pydantic V2 Schemas for Derivatives, Options, Volatility Surfaces, and Greeks.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class OptionType(StrEnum):
    """Option exercise type."""

    CALL = "CALL"
    PUT = "PUT"


class ExerciseStyle(StrEnum):
    """Exercise style."""

    EUROPEAN = "EUROPEAN"
    AMERICAN = "AMERICAN"


class OptionGreeksResult(BaseModel):
    """First- and second-order Greeks sensitivities."""

    model_config = ConfigDict(frozen=True)

    delta: float = Field(
        ..., description="dV/dS - Change in option value per 1 unit change in spot price"
    )
    gamma: float = Field(
        ..., description="d^2V/dS^2 - Change in delta per 1 unit change in spot price"
    )
    vega: float = Field(
        ...,
        description=(
            "0.01 * dV/dsigma - Change in option value for a 1 vol-point (0.01) rise in volatility"
        ),
    )
    theta: float = Field(
        ...,
        description=(
            "-(dV/dT) / 365 - Change in option value per calendar day passing (per-year theta / 365)"
        ),
    )
    rho: float = Field(
        ...,
        description=(
            "0.01 * dV/dr - Change in option value for a 1 percentage-point (0.01) rise in the rate"
        ),
    )
    vanna: float = Field(
        ...,
        description=(
            "d(delta)/dsigma = d^2V/(dS dsigma) with sigma in decimal, unscaled (per 1.0 of vol)"
        ),
    )
    volga: float = Field(
        ...,
        description=(
            "d(vega)/dsigma for the reported per-vol-point vega, i.e. 0.01 * d^2V/dsigma^2 (vomma)"
        ),
    )


class OptionPricingResult(BaseModel):
    """Complete option valuation result including pricing, implied vol, and Greeks."""

    model_config = ConfigDict(frozen=True)

    spot: float
    strike: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float
    option_type: OptionType
    exercise_style: ExerciseStyle
    price: float = Field(..., description="Theoretical fair value of the option")
    greeks: OptionGreeksResult


class SABRCalibrationResult(BaseModel):
    """Calibrated parameters of the SABR stochastic volatility model."""

    model_config = ConfigDict(frozen=True)

    alpha: float = Field(..., description="Initial volatility level")
    beta: float = Field(..., description="CEV elasticity parameter (fixed in [0, 1])")
    rho: float = Field(
        ..., description="Correlation between underlying price and volatility Wiener processes"
    )
    nu: float = Field(..., description="Volatility of volatility parameter")
    rmse: float = Field(
        ...,
        description="Root Mean Squared Error of fitted vs market volatilities (decimal vol units)",
    )
    is_arbitrage_free: bool = Field(
        ...,
        description=(
            "True if Black-76 call prices from the fitted smile show no butterfly or call-spread "
            "arbitrage across the quoted strike range (calendar arbitrage is not checked)"
        ),
    )


class ArbitrageVerificationResult(BaseModel):
    """SMT-verified validation of no-arbitrage bounds across an option strike/expiry surface."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    violations: list[str] = Field(default_factory=list)
