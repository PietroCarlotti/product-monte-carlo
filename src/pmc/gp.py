##############################
# Import necessary libraries #
##############################

# Standard libraries
from dataclasses import dataclass
from numbers import Real

# Third-party libraries
import numpy as np
import torch
from scipy.optimize import minimize

try:
    import pyro
    import pyro.distributions as pyro_distributions
    from pyro.infer import MCMC, NUTS
except ModuleNotFoundError as error:
    if error.name != "pyro":
        raise
    pyro = None
    pyro_distributions = None
    MCMC = None
    NUTS = None

########################
# Model configuration  #
########################

_KERNELS = {"rbf", "matern_32"}
_MEAN_PRIOR_LOCATION = 0.0
_MEAN_PRIOR_SCALE = 1.0
_LENGTHSCALE_PRIOR_LOCATION = 0.0
_LENGTHSCALE_PRIOR_SCALE = 1.0
_AMPLITUDE_PRIOR_LOCATION = 0.0
_AMPLITUDE_PRIOR_SCALE = 1.0
_NOISE_PRIOR_LOCATION = -1.0
_NOISE_PRIOR_SCALE = 1.0

# Validate a positive integer configuration value.
def _validate_positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

# Validate a nonnegative integer random seed.
def _validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")

# Validate a finite real configuration value.
def _validate_real(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

@dataclass(frozen=True)
class ExactBayesianGPConfig:
    """Configure exact Bayesian Gaussian-process regression."""

    kernel: str = "matern_32"
    warmup_steps: int = 200
    num_samples: int = 500
    num_chains: int = 1
    target_accept_probability: float = 0.8
    max_tree_depth: int = 10
    jitter: float = 1e-6

    # Validate every exact Bayesian GP configuration value.
    def __post_init__(self):
        if not isinstance(self.kernel, str) or self.kernel not in _KERNELS:
            raise ValueError("kernel must be rbf or matern_32.")
        for name in ("warmup_steps", "num_samples", "num_chains", "max_tree_depth"):
            _validate_positive_integer(name, getattr(self, name))
        _validate_real("target_accept_probability", self.target_accept_probability)
        if not 0 < self.target_accept_probability < 1:
            raise ValueError("target_accept_probability must be between zero and one.")
        _validate_real("jitter", self.jitter)
        if self.jitter <= 0:
            raise ValueError("jitter must be positive.")

@dataclass(frozen=True)
class ExactFixedHyperparameterGPConfig:
    """Configure exact Gaussian-process regression with fixed hyperparameters."""

    kernel: str = "matern_32"
    mean: float = 0.0
    lengthscale: float = 1.0
    amplitude: float = 1.0
    noise_standard_deviation: float = 1.0
    jitter: float = 1e-6

    # Validate every exact fixed-hyperparameter GP configuration value.
    def __post_init__(self):
        if not isinstance(self.kernel, str) or self.kernel not in _KERNELS:
            raise ValueError("kernel must be rbf or matern_32.")
        for name in (
            "mean",
            "lengthscale",
            "amplitude",
            "noise_standard_deviation",
            "jitter",
        ):
            _validate_real(name, getattr(self, name))
        for name in (
            "lengthscale",
            "amplitude",
            "noise_standard_deviation",
            "jitter",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")

@dataclass(frozen=True)
class ExactOptimizedHyperparameterGPConfig:
    """Configure exact GP marginal-likelihood hyperparameter optimization."""

    kernel: str = "matern_32"
    initial_mean: float = 0.0
    initial_lengthscale: float = 1.0
    initial_amplitude: float = 1.0
    initial_noise_standard_deviation: float = 1.0
    jitter: float = 1e-6
    maximum_iterations: int = 100
    optimization_tolerance: float = 1e-6
    objective: str = "map"

    # Validate every exact optimized-hyperparameter GP configuration value.
    def __post_init__(self):
        if not isinstance(self.kernel, str) or self.kernel not in _KERNELS:
            raise ValueError("kernel must be rbf or matern_32.")
        if self.objective not in {"ml", "map"}:
            raise ValueError("objective must be ml or map.")
        for name in (
            "initial_mean",
            "initial_lengthscale",
            "initial_amplitude",
            "initial_noise_standard_deviation",
            "jitter",
            "optimization_tolerance",
        ):
            _validate_real(name, getattr(self, name))
        for name in (
            "initial_lengthscale",
            "initial_amplitude",
            "initial_noise_standard_deviation",
            "jitter",
            "optimization_tolerance",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        _validate_positive_integer("maximum_iterations", self.maximum_iterations)

########################
# Numerical operations #
########################

# Convert an array-like value to a finite real float64 tensor.
def _as_real_tensor(values, name):
    try:
        tensor = torch.as_tensor(values)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must contain real numeric values.") from error
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise ValueError(f"{name} must contain real numeric values.")
    tensor = tensor.to(device="cpu", dtype=torch.float64).detach().clone()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor

# Validate and standardize one training data set.
def _standardize_training_data(inputs, response):
    input_tensor = _as_real_tensor(inputs, "inputs")
    response_tensor = _as_real_tensor(response, "response")
    if input_tensor.ndim == 1:
        input_tensor = input_tensor.unsqueeze(1)
    elif input_tensor.ndim != 2:
        raise ValueError("inputs must have shape (n,) or (n, d).")
    if input_tensor.shape[0] == 0 or input_tensor.shape[1] == 0:
        raise ValueError("inputs must contain at least one row and coordinate.")
    if response_tensor.ndim != 1:
        raise ValueError("response must have shape (n,).")
    if response_tensor.shape[0] != input_tensor.shape[0]:
        raise ValueError("inputs and response must contain the same number of rows.")
    input_mean = input_tensor.mean(dim=0)
    input_scale = input_tensor.std(dim=0, unbiased=False)
    if not torch.isfinite(input_mean).all() or not torch.isfinite(input_scale).all():
        raise ValueError("inputs cannot be standardized to finite values.")
    if torch.any(input_scale == 0):
        raise ValueError("inputs must not contain constant coordinates.")
    response_mean = response_tensor.mean()
    response_scale = response_tensor.std(unbiased=False)
    if not torch.isfinite(response_mean) or not torch.isfinite(response_scale):
        raise ValueError("response cannot be standardized to finite values.")
    if response_scale == 0:
        raise ValueError("response must not be constant.")
    standardized_inputs = (input_tensor - input_mean) / input_scale
    standardized_response = (response_tensor - response_mean) / response_scale
    return (
        standardized_inputs,
        standardized_response,
        input_mean,
        input_scale,
        response_mean,
        response_scale,
    )

# Validate and standardize query points for a fitted input dimension.
def _standardize_query(query, input_mean, input_scale):
    query_tensor = _as_real_tensor(query, "query")
    num_features = input_mean.numel()
    if query_tensor.ndim == 1 and num_features == 1:
        query_tensor = query_tensor.unsqueeze(1)
    elif query_tensor.ndim != 2 or query_tensor.shape[1] != num_features:
        if num_features == 1:
            raise ValueError("query must have shape (q,) or (q, 1).")
        raise ValueError(f"query must have shape (q, {num_features}).")
    if query_tensor.shape[0] == 0:
        raise ValueError("query must contain at least one point.")
    standardized_query = (query_tensor - input_mean) / input_scale
    if not torch.isfinite(standardized_query).all():
        raise ValueError("query cannot be standardized to finite values.")
    return standardized_query

# Compute a supported stationary kernel covariance matrix.
def _kernel_covariance(first, second, lengthscale, amplitude, kernel):
    if kernel not in _KERNELS:
        raise ValueError("kernel must be rbf or matern_32.")
    distances = torch.cdist(first / lengthscale, second / lengthscale)
    if kernel == "rbf":
        correlation = torch.exp(-0.5 * distances.square())
    else:
        scaled = np.sqrt(3.0) * distances
        correlation = (1.0 + scaled) * torch.exp(-scaled)
    return amplitude.square() * correlation

# Compute exact conditional moments for the latent function at query points.
def _conditional_latent_moments(
    train_inputs,
    train_response,
    query_inputs,
    mean,
    lengthscale,
    amplitude,
    noise_standard_deviation,
    kernel,
    jitter,
):
    num_train = train_inputs.shape[0]
    train_covariance = _kernel_covariance(
        train_inputs, train_inputs, lengthscale, amplitude, kernel
    )
    train_covariance = train_covariance + (
        noise_standard_deviation.square() + jitter
    ) * torch.eye(num_train, dtype=torch.float64)
    train_cholesky = torch.linalg.cholesky(train_covariance)
    train_query_covariance = _kernel_covariance(
        train_inputs, query_inputs, lengthscale, amplitude, kernel
    )
    query_covariance = _kernel_covariance(
        query_inputs, query_inputs, lengthscale, amplitude, kernel
    )
    residual = (train_response - mean).unsqueeze(1)
    weights = torch.cholesky_solve(residual, train_cholesky)
    conditional_mean = mean + train_query_covariance.transpose(0, 1) @ weights
    solved_cross_covariance = torch.cholesky_solve(
        train_query_covariance, train_cholesky
    )
    conditional_covariance = query_covariance - (
        train_query_covariance.transpose(0, 1) @ solved_cross_covariance
    )
    conditional_covariance = 0.5 * (
        conditional_covariance + conditional_covariance.transpose(0, 1)
    )
    return conditional_mean.squeeze(1), conditional_covariance

# Compute the exact negative log marginal likelihood for one parameter vector.
def _negative_log_marginal_likelihood(
    parameter_vector, train_inputs, train_response, kernel, jitter
):
    mean = torch.tensor(parameter_vector[0], dtype=torch.float64)
    lengthscale = torch.tensor(np.exp(parameter_vector[1]), dtype=torch.float64)
    amplitude = torch.tensor(np.exp(parameter_vector[2]), dtype=torch.float64)
    noise_standard_deviation = torch.tensor(
        np.exp(parameter_vector[3]), dtype=torch.float64
    )
    covariance = _kernel_covariance(
        train_inputs, train_inputs, lengthscale, amplitude, kernel
    )
    covariance = covariance + (
        noise_standard_deviation.square() + jitter
    ) * torch.eye(train_inputs.shape[0], dtype=torch.float64)
    try:
        cholesky = torch.linalg.cholesky(covariance)
    except torch.linalg.LinAlgError:
        return np.inf
    residual = train_response - mean
    weights = torch.cholesky_solve(residual.unsqueeze(1), cholesky)
    quadratic = residual.unsqueeze(0) @ weights
    log_determinant = 2.0 * torch.log(torch.diagonal(cholesky)).sum()
    value = 0.5 * (quadratic.squeeze() + log_determinant)
    value = value + 0.5 * train_inputs.shape[0] * np.log(2.0 * np.pi)
    return float(value.item()) if torch.isfinite(value) else np.inf

# Compute the exact negative log posterior for one parameter vector.
def _negative_log_posterior(
    parameter_vector, train_inputs, train_response, kernel, jitter
):
    negative_log_likelihood = _negative_log_marginal_likelihood(
        parameter_vector, train_inputs, train_response, kernel, jitter
    )
    if not np.isfinite(negative_log_likelihood):
        return np.inf
    mean, log_lengthscale, log_amplitude, log_noise = parameter_vector
    negative_log_prior = 0.5 * mean**2
    negative_log_prior += 0.5 * log_lengthscale**2 + log_lengthscale
    negative_log_prior += 0.5 * log_amplitude**2 + log_amplitude
    negative_log_prior += 0.5 * (log_noise + 1.0) ** 2 + log_noise
    return float(negative_log_likelihood + negative_log_prior)

########################
# Exact Bayesian model #
########################

class ExactBayesianGaussianProcessRegressor:
    """Fit an exact Bayesian GP and sample its latent posterior."""

    # Initialize an unfitted exact Bayesian GP regressor.
    def __init__(self, config=None):
        if config is None:
            config = ExactBayesianGPConfig()
        if not isinstance(config, ExactBayesianGPConfig):
            raise TypeError("config must be an ExactBayesianGPConfig.")
        self.config = config
        self._train_inputs = None
        self._train_response = None
        self._input_mean = None
        self._input_scale = None
        self._response_mean = None
        self._response_scale = None
        self._posterior_samples = None

    # Define the analytically marginalized exact GP likelihood.
    def _model(self, inputs, response):
        reference = inputs.new_tensor
        mean = pyro.sample(
            "mean",
            pyro_distributions.Normal(
                reference(_MEAN_PRIOR_LOCATION), reference(_MEAN_PRIOR_SCALE)
            ),
        )
        lengthscale = pyro.sample(
            "lengthscale",
            pyro_distributions.LogNormal(
                reference(_LENGTHSCALE_PRIOR_LOCATION),
                reference(_LENGTHSCALE_PRIOR_SCALE),
            ),
        )
        amplitude = pyro.sample(
            "amplitude",
            pyro_distributions.LogNormal(
                reference(_AMPLITUDE_PRIOR_LOCATION),
                reference(_AMPLITUDE_PRIOR_SCALE),
            ),
        )
        noise_standard_deviation = pyro.sample(
            "noise_standard_deviation",
            pyro_distributions.LogNormal(
                reference(_NOISE_PRIOR_LOCATION), reference(_NOISE_PRIOR_SCALE)
            ),
        )
        covariance = _kernel_covariance(
            inputs,
            inputs,
            lengthscale,
            amplitude,
            self.config.kernel,
        )
        covariance = covariance + (
            noise_standard_deviation.square() + self.config.jitter
        ) * torch.eye(inputs.shape[0], dtype=torch.float64)
        pyro.sample(
            "response",
            pyro_distributions.MultivariateNormal(
                mean.expand(inputs.shape[0]), covariance_matrix=covariance
            ),
            obs=response,
        )

    # Fit hyperparameters with NUTS on standardized training data.
    def fit(self, inputs, response, seed=0):
        _validate_seed(seed)
        standardized = _standardize_training_data(inputs, response)
        if pyro is None:
            raise ImportError("Pyro is required to fit ExactBayesianGaussianProcessRegressor.")
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)
        nuts_kernel = NUTS(
            self._model,
            target_accept_prob=self.config.target_accept_probability,
            max_tree_depth=self.config.max_tree_depth,
        )
        sampler = MCMC(
            nuts_kernel,
            warmup_steps=self.config.warmup_steps,
            num_samples=self.config.num_samples,
            num_chains=self.config.num_chains,
            disable_progbar=True,
        )
        sampler.run(standardized[0], standardized[1])
        posterior_samples = sampler.get_samples(group_by_chain=False)
        self._train_inputs = standardized[0]
        self._train_response = standardized[1]
        self._input_mean = standardized[2]
        self._input_scale = standardized[3]
        self._response_mean = standardized[4]
        self._response_scale = standardized[5]
        self._posterior_samples = {
            name: posterior_samples[name].detach().clone()
            for name in (
                "mean",
                "lengthscale",
                "amplitude",
                "noise_standard_deviation",
            )
        }
        return self

    # Return defensive copies of retained hyperparameter draws.
    @property
    def posterior_samples(self):
        self._require_fitted()
        return {
            name: samples.detach().clone()
            for name, samples in self._posterior_samples.items()
        }

    # Require fitted training state and posterior draws.
    def _require_fitted(self):
        if self._posterior_samples is None:
            raise RuntimeError("The regressor must be fitted before prediction.")

    # Draw the latent function at arbitrary query points.
    def sample_latent(self, query, seed=0):
        self._require_fitted()
        _validate_seed(seed)
        query_inputs = _standardize_query(query, self._input_mean, self._input_scale)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        draws = []
        num_query = query_inputs.shape[0]
        identity = torch.eye(num_query, dtype=torch.float64)
        for index in range(self._posterior_samples["mean"].shape[0]):
            conditional_mean, conditional_covariance = _conditional_latent_moments(
                self._train_inputs,
                self._train_response,
                query_inputs,
                self._posterior_samples["mean"][index],
                self._posterior_samples["lengthscale"][index],
                self._posterior_samples["amplitude"][index],
                self._posterior_samples["noise_standard_deviation"][index],
                self.config.kernel,
                self.config.jitter,
            )
            cholesky = torch.linalg.cholesky(
                conditional_covariance + self.config.jitter * identity
            )
            standard_normal = torch.randn(
                num_query, dtype=torch.float64, generator=generator
            )
            standardized_draw = conditional_mean + cholesky @ standard_normal
            draws.append(
                standardized_draw * self._response_scale + self._response_mean
            )
        return torch.stack(draws).detach().cpu().numpy()

###############################
# Exact fixed-parameter model #
###############################

class ExactFixedHyperparameterGaussianProcessRegressor:
    """Fit an exact GP with fixed hyperparameters and sample its latent posterior."""

    # Initialize an unfitted exact fixed-hyperparameter GP regressor.
    def __init__(self, config=None):
        if config is None:
            config = ExactFixedHyperparameterGPConfig()
        if not isinstance(config, ExactFixedHyperparameterGPConfig):
            raise TypeError(
                "config must be an ExactFixedHyperparameterGPConfig."
            )
        self.config = config
        self._train_inputs = None
        self._train_response = None
        self._input_mean = None
        self._input_scale = None
        self._response_mean = None
        self._response_scale = None
        self._train_cholesky = None

    # Fit the exact fixed-hyperparameter GP on standardized training data.
    def fit(self, inputs, response):
        standardized = _standardize_training_data(inputs, response)
        self._train_inputs = standardized[0]
        self._train_response = standardized[1]
        self._input_mean = standardized[2]
        self._input_scale = standardized[3]
        self._response_mean = standardized[4]
        self._response_scale = standardized[5]
        covariance = _kernel_covariance(
            self._train_inputs,
            self._train_inputs,
            torch.tensor(self.config.lengthscale, dtype=torch.float64),
            torch.tensor(self.config.amplitude, dtype=torch.float64),
            self.config.kernel,
        )
        covariance = covariance + (
            self.config.noise_standard_deviation**2 + self.config.jitter
        ) * torch.eye(self._train_inputs.shape[0], dtype=torch.float64)
        self._train_cholesky = torch.linalg.cholesky(covariance)
        return self

    # Require fitted training state before prediction.
    def _require_fitted(self):
        if self._train_cholesky is None:
            raise RuntimeError("The regressor must be fitted before prediction.")

    # Draw independent latent posterior replicates at arbitrary query points.
    def sample_latent(self, query, seed=0, num_draws=1):
        self._require_fitted()
        _validate_seed(seed)
        _validate_positive_integer("num_draws", num_draws)
        query_inputs = _standardize_query(query, self._input_mean, self._input_scale)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        conditional_mean, conditional_covariance = _conditional_latent_moments(
            self._train_inputs,
            self._train_response,
            query_inputs,
            torch.tensor(self.config.mean, dtype=torch.float64),
            torch.tensor(self.config.lengthscale, dtype=torch.float64),
            torch.tensor(self.config.amplitude, dtype=torch.float64),
            torch.tensor(self.config.noise_standard_deviation, dtype=torch.float64),
            self.config.kernel,
            self.config.jitter,
        )
        identity = torch.eye(query_inputs.shape[0], dtype=torch.float64)
        cholesky = torch.linalg.cholesky(
            conditional_covariance + self.config.jitter * identity
        )
        standard_normal = torch.randn(
            (num_draws, query_inputs.shape[0]),
            dtype=torch.float64,
            generator=generator,
        )
        standardized_draws = conditional_mean + standard_normal @ cholesky.transpose(0, 1)
        return (
            standardized_draws * self._response_scale + self._response_mean
        ).detach().cpu().numpy()

################################
# Exact optimized-parameter GP #
################################

class ExactOptimizedHyperparameterGaussianProcessRegressor:
    """Fit an exact GP by optimizing its marginal likelihood hyperparameters."""

    # Initialize an unfitted exact optimized-hyperparameter GP regressor.
    def __init__(self, config=None):
        if config is None:
            config = ExactOptimizedHyperparameterGPConfig()
        if not isinstance(config, ExactOptimizedHyperparameterGPConfig):
            raise TypeError(
                "config must be an ExactOptimizedHyperparameterGPConfig."
            )
        self.config = config
        self._regressor = None
        self._optimized_hyperparameters = None

    # Fit hyperparameters by minimizing the exact negative log marginal likelihood.
    def fit(self, inputs, response):
        standardized = _standardize_training_data(inputs, response)
        initial_parameters = np.array(
            [
                self.config.initial_mean,
                np.log(self.config.initial_lengthscale),
                np.log(self.config.initial_amplitude),
                np.log(self.config.initial_noise_standard_deviation),
            ],
            dtype=float,
        )
        objective = (
            _negative_log_posterior
            if self.config.objective == "map"
            else _negative_log_marginal_likelihood
        )
        optimization = minimize(
            objective,
            initial_parameters,
            args=(
                standardized[0],
                standardized[1],
                self.config.kernel,
                self.config.jitter,
            ),
            method="L-BFGS-B",
            options={
                "maxiter": self.config.maximum_iterations,
                "ftol": self.config.optimization_tolerance,
            },
        )
        if not optimization.success or not np.isfinite(optimization.fun):
            raise RuntimeError(
                "Exact GP hyperparameter optimization failed: "
                f"{optimization.message}"
            )
        parameters = optimization.x
        self._optimized_hyperparameters = {
            "mean": float(parameters[0]),
            "lengthscale": float(np.exp(parameters[1])),
            "amplitude": float(np.exp(parameters[2])),
            "noise_standard_deviation": float(np.exp(parameters[3])),
        }
        self._regressor = ExactFixedHyperparameterGaussianProcessRegressor(
            ExactFixedHyperparameterGPConfig(
                kernel=self.config.kernel,
                jitter=self.config.jitter,
                **self._optimized_hyperparameters,
            )
        )
        self._regressor.fit(inputs, response)
        return self

    # Return the optimized hyperparameters as a defensive copy.
    @property
    def optimized_hyperparameters(self):
        if self._optimized_hyperparameters is None:
            raise RuntimeError("The regressor must be fitted before inspection.")
        return dict(self._optimized_hyperparameters)

    # Draw independent latent posterior replicates at arbitrary query points.
    def sample_latent(self, query, seed=0, num_draws=1):
        if self._regressor is None:
            raise RuntimeError("The regressor must be fitted before prediction.")
        return self._regressor.sample_latent(
            query, seed=seed, num_draws=num_draws
        )
