"""Shared Bayesian logistic-regression model interface.

Dataset-specific pipelines should prepare ``X`` and binary ``y`` and call
this module.  Sampling backends are intentionally kept behind ``fit`` so the
same model definition can serve the respiratory and HIGGS applications.
"""

import time
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from scipy.optimize import minimize


@dataclass(frozen=True)
class BayesianLogisticConfig:
    """Likelihood, prior, and sampling settings; callers may override every field."""

    prior_scale: float = 2.0
    chains: int = 4
    warmup: int = 1_000
    post_warmup: int = 10_000
    seed: int = 0
    parallel_chains: bool = True
    dense_mass: bool = False
    map_initialization: bool = False
    map_max_iterations: int = 500


@dataclass
class BayesianLogisticResult:
    """Container for persisted posterior draws and run metadata."""

    draws: Any
    runtime_seconds: float
    diagnostics: dict[str, Any]
    feature_names: tuple[str, ...]


# Convert HMC diagnostics into JSON-safe per-parameter metadata.
def diagnostics_payload(result: BayesianLogisticResult) -> dict[str, object]:
    summary = result.diagnostics["summary"]["beta"]
    effective_sample_sizes = np.asarray(summary["n_eff"], dtype=float)
    split_r_hats = np.asarray(summary["r_hat"], dtype=float)
    parameter_diagnostics = {
        name: {
            "effective_sample_size": float(effective_sample_sizes[index])
            if np.isfinite(effective_sample_sizes[index])
            else None,
            "split_r_hat": float(split_r_hats[index]) if np.isfinite(split_r_hats[index]) else None,
        }
        for index, name in enumerate(result.feature_names)
    }
    return {
        "chains": result.diagnostics["num_chains"],
        "warmup": result.diagnostics["num_warmup"],
        "post_warmup": result.diagnostics["num_samples"],
        "backend": result.diagnostics["backend"],
        "parallel_chains": result.diagnostics["parallel_chains"],
        "dense_mass": result.diagnostics.get("dense_mass", False),
        "initialization": result.diagnostics.get("initialization", "default"),
        "map_max_iterations": result.diagnostics.get("map_max_iterations", 500),
        "jax_enable_x64": result.diagnostics.get("jax_enable_x64", False),
        "effective_sample_size_min": float(np.nanmin(effective_sample_sizes))
        if np.isfinite(effective_sample_sizes).any()
        else None,
        "effective_sample_size_median": float(np.nanmedian(effective_sample_sizes))
        if np.isfinite(effective_sample_sizes).any()
        else None,
        "split_r_hat_max": float(np.nanmax(split_r_hats))
        if np.isfinite(split_r_hats).any()
        else None,
        "parameters": parameter_diagnostics,
    }


# Compute posterior diagnostics after draws have been safely persisted.
def add_diagnostics(result: BayesianLogisticResult) -> BayesianLogisticResult:
    result.diagnostics["summary"] = numpyro.diagnostics.summary(result.draws)
    return result


def validate_inputs(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize a design matrix and binary response."""

    matrix = np.asarray(x, dtype=float)
    response = np.asarray(y).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")
    if response.ndim != 1 or len(response) != len(matrix):
        raise ValueError("y must have one entry for every row of X.")
    if not np.isfinite(matrix).all():
        raise ValueError("X must contain only finite values.")
    if not np.isin(response, (0, 1)).all():
        raise ValueError("y must be binary and coded as 0 or 1.")
    return matrix, response.astype(int)


def validate_config(config: BayesianLogisticConfig) -> None:
    """Reject invalid sampling and prior settings before a run starts."""

    if config.prior_scale <= 0:
        raise ValueError("prior_scale must be positive.")
    if config.chains < 1 or config.warmup < 0 or config.post_warmup < 1:
        raise ValueError("Invalid MCMC chain or draw counts.")
    if config.map_max_iterations < 1:
        raise ValueError("map_max_iterations must be positive.")


# Evaluate the negative log posterior and its analytical gradient.
def _negative_log_posterior_and_gradient(
    beta: np.ndarray,
    matrix: np.ndarray,
    response: np.ndarray,
    prior_scale: float,
) -> tuple[float, np.ndarray]:
    linear_predictor = matrix @ beta
    objective = np.sum(np.logaddexp(0.0, linear_predictor) - response * linear_predictor)
    objective += 0.5 * np.dot(beta, beta) / prior_scale**2
    probabilities = np.exp(-np.logaddexp(0.0, -linear_predictor))
    gradient = matrix.T @ (probabilities - response) + beta / prior_scale**2
    return float(objective), np.asarray(gradient, dtype=float)


# Compute a deterministic posterior mode for sampler initialization.
def find_map_estimate(
    x: Any,
    y: Any,
    config: BayesianLogisticConfig,
) -> np.ndarray:
    validate_config(config)
    matrix, response = validate_inputs(x, y)
    initial = np.zeros(matrix.shape[1], dtype=float)
    result = minimize(
        _negative_log_posterior_and_gradient,
        initial,
        args=(matrix, response, config.prior_scale),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": config.map_max_iterations},
    )
    estimate = np.asarray(result.x, dtype=float)
    if not result.success:
        raise RuntimeError(f"MAP optimization failed: {result.message}")
    if estimate.shape != initial.shape or not np.isfinite(estimate).all():
        raise RuntimeError("MAP optimization returned an invalid estimate.")
    objective, gradient = _negative_log_posterior_and_gradient(
        estimate,
        matrix,
        response,
        config.prior_scale,
    )
    if not np.isfinite(objective) or not np.isfinite(gradient).all():
        raise RuntimeError("MAP optimization returned a non-finite objective or gradient.")
    return estimate


def log_posterior(beta: Any, x: Any, y: Any, config: BayesianLogisticConfig) -> float:
    """Evaluate the shared Bayesian logistic log posterior.

    The likelihood is ``y_i ~ Bernoulli(sigmoid(X_i beta))`` and the prior is
    independent ``beta_j ~ Normal(0, prior_scale²)``.
    """

    validate_config(config)
    matrix, response = validate_inputs(x, y)
    coefficients = np.asarray(beta, dtype=float).reshape(-1)
    if coefficients.size != matrix.shape[1]:
        raise ValueError("beta must have one coefficient per column of X.")
    if not np.isfinite(coefficients).all():
        raise ValueError("beta must contain only finite values.")

    linear_predictor = matrix @ coefficients
    log_likelihood = np.sum(
        response * (-np.logaddexp(0.0, -linear_predictor))
        + (1 - response) * (-np.logaddexp(0.0, linear_predictor))
    )
    log_prior = -0.5 * np.sum(coefficients**2) / (config.prior_scale**2)
    return float(log_likelihood + log_prior)


def _model(x: Any, y: Any, prior_scale: float) -> None:
    """NumPyro model shared by every dataset adapter."""
    beta = numpyro.sample("beta", dist.Normal(0.0, prior_scale).expand((x.shape[1],)).to_event(1))
    with numpyro.plate("observations", x.shape[0]):
        numpyro.sample("y", dist.Bernoulli(logits=x @ beta), obs=y)


def fit(
    x: Any,
    y: Any,
    config: BayesianLogisticConfig | None = None,
    feature_names: tuple[str, ...] | None = None,
) -> BayesianLogisticResult:
    """Run NumPyro HMC and return chain-grouped posterior draws."""
    config = config or BayesianLogisticConfig()
    validate_config(config)
    matrix, response = validate_inputs(x, y)
    names = feature_names or tuple(f"x_{i}" for i in range(matrix.shape[1]))
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names must match the number of X columns.")
    if config.parallel_chains:
        numpyro.set_host_device_count(config.chains)
    kernel_options: dict[str, Any] = {"dense_mass": config.dense_mass}
    if config.map_initialization:
        map_estimate = find_map_estimate(matrix, response, config)
        kernel_options["init_strategy"] = init_to_value(values={"beta": map_estimate})
    sampler = MCMC(
        NUTS(_model, **kernel_options),
        num_warmup=config.warmup,
        num_samples=config.post_warmup,
        num_chains=config.chains,
        progress_bar=False,
    )
    started = time.perf_counter()
    sampler.run(jax.random.key(config.seed), x=matrix, y=response, prior_scale=config.prior_scale)
    runtime = time.perf_counter() - started
    draws = {
        name: np.asarray(jax.device_get(values))
        for name, values in sampler.get_samples(group_by_chain=True).items()
    }
    return BayesianLogisticResult(
        draws=draws,
        runtime_seconds=runtime,
        diagnostics={
            "num_chains": config.chains,
            "num_warmup": config.warmup,
            "num_samples": config.post_warmup,
            "backend": jax.default_backend(),
            "parallel_chains": config.parallel_chains,
            "dense_mass": config.dense_mass,
            "initialization": "map" if config.map_initialization else "default",
            "map_max_iterations": config.map_max_iterations,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
        feature_names=tuple(names),
    )
