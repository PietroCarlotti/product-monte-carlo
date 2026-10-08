##############################
# Import necessary libraries #
##############################

# Standard libraries
import copy
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from threading import Lock

# Third-party libraries
import gpytorch
import numpy as np
import torch
from linear_operator.utils.errors import NotPSDError
from scipy.stats import norm, qmc

# PMC library
from ._version import __version__

########################
# Persistence metadata #
########################

PMC_GP_MODEL_FORMAT_VERSION = 1


_TORCH_STATE_LOCK = Lock()


# Isolate process-global Torch state used during model fitting.
@contextmanager
def _isolated_torch_state(seed, num_threads, device):
    with _TORCH_STATE_LOCK:
        rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        )
        original_num_threads = torch.get_num_threads()
        try:
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.default_generators[device.index].manual_seed(seed)
            if num_threads is not None:
                torch.set_num_threads(num_threads)
            yield
        finally:
            try:
                torch.set_rng_state(rng_state)
            finally:
                try:
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state(cuda_rng_state, device)
                finally:
                    if torch.get_num_threads() != original_num_threads:
                        torch.set_num_threads(original_num_threads)

# Convert nested result metadata to values accepted by strict JSON encoders.
def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

########################
# Public configuration #
########################

@dataclass(frozen=True)
class PMCConfig:
    """Configure full-sample sparse variational GP regression."""

    inducing_count: int = 8
    maximum_updates: int | None = None
    batch_size: int = 8192
    learning_rate: float = 0.03
    natural_gradient_learning_rate: float = 0.03
    minimum_epochs: int = 10
    patience: int = 3
    relative_tolerance: float = 1e-4
    learn_inducing_locations: bool = True
    full_batch: bool = False
    fixed_hyperparameters: bool = False
    fixed_mean: float = 0.0
    fixed_lengthscale: float = 1.0
    fixed_outputscale: float = 1.0
    fixed_noise: float = 1.0
    dtype: str = "float32"
    torch_num_threads: int | None = None
    kernel: str = "rbf"
    device: str = "cpu"

    # Validate all sparse-GP configuration values.
    def __post_init__(self):
        positive_integers = {
            "inducing_count": self.inducing_count,
            "batch_size": self.batch_size,
            "minimum_epochs": self.minimum_epochs,
            "patience": self.patience,
        }
        for name, value in positive_integers.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.maximum_updates is not None and (
            not isinstance(self.maximum_updates, int)
            or isinstance(self.maximum_updates, bool)
            or self.maximum_updates <= 0
        ):
            raise ValueError("maximum_updates must be a positive integer or None.")
        if self.torch_num_threads is not None and (
            not isinstance(self.torch_num_threads, int)
            or isinstance(self.torch_num_threads, bool)
            or self.torch_num_threads <= 0
        ):
            raise ValueError("torch_num_threads must be a positive integer or None.")
        if not isinstance(self.full_batch, bool):
            raise ValueError("full_batch must be boolean.")
        if not isinstance(self.fixed_hyperparameters, bool):
            raise ValueError("fixed_hyperparameters must be boolean.")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64.")
        positive_values = {
            "learning_rate": self.learning_rate,
            "natural_gradient_learning_rate": self.natural_gradient_learning_rate,
            "relative_tolerance": self.relative_tolerance,
        }
        for name, value in positive_values.items():
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite.")
        fixed_values = {
            "fixed_lengthscale": self.fixed_lengthscale,
            "fixed_outputscale": self.fixed_outputscale,
            "fixed_noise": self.fixed_noise,
        }
        for name, value in fixed_values.items():
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite.")
        if (
            not isinstance(self.fixed_mean, Real)
            or isinstance(self.fixed_mean, bool)
            or not np.isfinite(self.fixed_mean)
        ):
            raise ValueError("fixed_mean must be finite.")
        if self.kernel not in {"rbf", "matern_32", "matern_52"}:
            raise ValueError("kernel must be rbf, matern_32, or matern_52.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda.")


@dataclass(frozen=True)
class PMCResult:
    """Store one PMC-GP estimate and its approximate posterior uncertainty."""

    estimate: float
    posterior_standard_deviation: float
    interval_lower: float
    interval_upper: float
    confidence_level: float
    query_discrepancy: float | list[float]
    interval_method: str
    training_time_seconds: float
    diagnostics: dict
    configuration: dict

    # Export the estimate and its provenance as a strict JSON-compatible mapping.
    def to_dict(self):
        return _json_safe(asdict(self))


class PMCOptimizationError(RuntimeError):
    """Report a numerical failure during PMC-GP optimization or prediction."""


# Return the empirically validated target-specific PMC-GP configuration.
def recommended_pmc_gp_config(target, **overrides):
    kernels = {"identity": "matern_32", "square": "rbf"}
    if target not in kernels:
        raise ValueError("target must be identity or square.")
    return PMCConfig(kernel=kernels[target], **overrides)

#############################
# Variational process model #
#############################

class _SparseVariationalGP(gpytorch.models.ApproximateGP):
    # Initialize the natural variational distribution and RBF covariance kernel.
    def __init__(self, inducing_points, learn_inducing_locations, kernel):
        distribution = gpytorch.variational.NaturalVariationalDistribution(
            inducing_points.size(0)
        )
        strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == "matern_32":
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel == "matern_52":
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        else:
            base_kernel = gpytorch.kernels.RBFKernel()
        self.covariance_module = gpytorch.kernels.ScaleKernel(
            base_kernel
        )

    # Evaluate the latent Gaussian process at supplied inputs.
    def forward(self, inputs):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(inputs), self.covariance_module(inputs)
        )

########################
# Public PMC estimator #
########################

class GaussianProcessPMC:
    """Estimate E[h(C) | D=d] with a full-sample variational Gaussian process."""

    # Initialize an unfitted PMC-GP estimator.
    def __init__(self, config=None):
        self.config = PMCConfig() if config is None else config
        if not isinstance(self.config, PMCConfig):
            raise TypeError(
                "config must be a PMCConfig instance or None."
            )
        self._model = None
        self._x_mean = None
        self._x_scale = None
        self._y_mean = None
        self._y_scale = None
        self._training_time_seconds = None
        self._diagnostics = None
        self._seed = None
        self._standardized_discrepancy = None

    # Resolve an explicitly requested Torch device.
    @staticmethod
    def _resolve_device(device):
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested for PMC-GP, but CUDA is not available."
                )
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    # Describe the active Torch device for estimator diagnostics.
    @staticmethod
    def _device_metadata(device):
        return {
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime_version": torch.version.cuda,
            "cuda_device_index": (
                device.index if device.type == "cuda" else None
            ),
            "cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        }

    # Synchronize pending work on the selected device.
    @staticmethod
    def _synchronize_device(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Compute the complete-epoch training budget for a sample size.
    def _update_budget(self, n_observations):
        batch_size = (
            n_observations
            if self.config.full_batch
            else min(self.config.batch_size, n_observations)
        )
        batches_per_epoch = (n_observations + batch_size - 1) // batch_size
        minimum_updates = self.config.minimum_epochs * batches_per_epoch
        if self.config.maximum_updates is None:
            effective_maximum_updates = max(
                150,
                (self.config.minimum_epochs + self.config.patience)
                * batches_per_epoch,
            )
        else:
            effective_maximum_updates = self.config.maximum_updates
            if minimum_updates > effective_maximum_updates:
                raise ValueError(
                    "maximum_updates must permit minimum_epochs complete epochs; "
                    f"at least {minimum_updates} updates are required."
                )
        return batch_size, batches_per_epoch, effective_maximum_updates

    # Return the device holding the fitted model parameters.
    def _model_device(self):
        return next(self._model.parameters()).device

    # Return the configured floating-point Torch dtype.
    def _torch_dtype(self):
        return torch.float64 if self.config.dtype == "float64" else torch.float32

    # Save a fitted estimator in the versioned PMC-GP model format.
    def save(self, path):
        if self._model is None:
            raise RuntimeError("fit must be called before save.")
        payload = {
            "model_type": "pmc.GaussianProcessPMC",
            "model_format_version": PMC_GP_MODEL_FORMAT_VERSION,
            "pmc_version": __version__,
            "configuration": asdict(self.config),
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in self._model.state_dict().items()
            },
            "inducing_points": (
                self._model.variational_strategy.inducing_points.detach().cpu().clone()
            ),
            "x_mean": self._x_mean,
            "x_scale": self._x_scale,
            "y_mean": self._y_mean,
            "y_scale": self._y_scale,
            "training_time_seconds": self._training_time_seconds,
            "diagnostics": self._diagnostics,
            "seed": self._seed,
            "standardized_discrepancy": self._standardized_discrepancy,
        }
        torch.save(payload, Path(path))

    # Load a fitted estimator from the versioned PMC-GP model format.
    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("The saved PMC-GP model must contain a mapping.")
        if payload.get("model_type") != "pmc.GaussianProcessPMC":
            raise ValueError("The saved file is not a PMC-GP estimator.")
        version = payload.get("model_format_version")
        if version != PMC_GP_MODEL_FORMAT_VERSION:
            raise ValueError(
                "Unsupported PMC-GP model format version "
                f"{version}; expected {PMC_GP_MODEL_FORMAT_VERSION}."
            )
        required_fields = {
            "configuration",
            "model_state_dict",
            "inducing_points",
            "x_mean",
            "x_scale",
            "y_mean",
            "y_scale",
            "training_time_seconds",
            "diagnostics",
            "seed",
            "standardized_discrepancy",
        }
        missing_fields = sorted(required_fields.difference(payload))
        if missing_fields:
            raise ValueError(
                "The saved PMC-GP model is missing fields: "
                + ", ".join(missing_fields)
                + "."
            )
        configuration = dict(payload["configuration"])
        if device is not None:
            configuration["device"] = device
        estimator = cls(PMCConfig(**configuration))
        model_device = estimator._resolve_device(estimator.config.device)
        estimator._model = _SparseVariationalGP(
            payload["inducing_points"],
            estimator.config.learn_inducing_locations,
            estimator.config.kernel,
        )
        estimator._model.load_state_dict(payload["model_state_dict"])
        estimator._model.to(
            device=model_device, dtype=estimator._torch_dtype()
        )
        estimator._model.eval()
        estimator._x_mean = payload["x_mean"]
        estimator._x_scale = payload["x_scale"]
        estimator._y_mean = payload["y_mean"]
        estimator._y_scale = payload["y_scale"]
        estimator._training_time_seconds = payload["training_time_seconds"]
        estimator._diagnostics = dict(payload["diagnostics"])
        estimator._diagnostics.update(estimator._device_metadata(model_device))
        estimator._diagnostics["requested_device"] = estimator.config.device
        estimator._seed = payload["seed"]
        estimator._standardized_discrepancy = payload[
            "standardized_discrepancy"
        ]
        return estimator

    # Convert one-dimensional finite training values to a flat array.
    @staticmethod
    def _training_array(values, name):
        array = np.asarray(values, dtype=float)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional or have one column.")
        if len(array) < 2:
            raise ValueError(f"{name} must contain at least two observations.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array

    # Convert finite discrepancy values to a scalar or feature matrix.
    @staticmethod
    def _discrepancy_array(values):
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            if len(array) < 2:
                raise ValueError(
                    "discrepancy must contain at least two observations."
                )
        elif array.ndim == 2:
            if array.shape[1] == 0:
                raise ValueError(
                    "discrepancy must contain at least one feature."
                )
            if array.shape[0] < 2:
                raise ValueError(
                    "discrepancy must contain at least two observations."
                )
            if array.shape[1] == 1:
                array = array[:, 0]
        else:
            raise ValueError(
                "discrepancy must be one-dimensional or a two-dimensional "
                "feature matrix."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("discrepancy must contain only finite values.")
        return array

    # Initialize inducing inputs at empirical discrepancy quantiles.
    def _inducing_points(self, standardized_discrepancy):
        count = min(self.config.inducing_count, len(standardized_discrepancy))
        probabilities = np.linspace(0, 1, count)
        if standardized_discrepancy.ndim == 1:
            values = np.quantile(standardized_discrepancy, probabilities)
            return torch.tensor(values, dtype=self._torch_dtype()).unsqueeze(-1)
        num_features = standardized_discrepancy.shape[1]
        ranks = np.arange(count)
        values = np.empty((count, num_features))
        golden_ratio = (1 + np.sqrt(5)) / 2
        legacy_feature_count = min(num_features, 64)
        stable_ranks = None
        if num_features > legacy_feature_count:
            exponent = (count - 1).bit_length()
            sobol_points = qmc.Sobol(
                d=num_features - legacy_feature_count,
                scramble=False,
            ).random_base2(exponent)
            stable_ranks = np.argsort(
                sobol_points[:count], axis=0, kind="stable"
            )
        for feature in range(num_features):
            if feature == 0:
                feature_ranks = ranks
            elif feature >= legacy_feature_count:
                feature_ranks = stable_ranks[:, feature - legacy_feature_count]
            else:
                fractional_ranks = np.mod(
                    (ranks + 0.5) * golden_ratio**feature, 1
                )
                feature_ranks = np.argsort(fractional_ranks, kind="stable")
            values[:, feature] = np.quantile(
                standardized_discrepancy[:, feature],
                probabilities[feature_ranks],
            )
        values[0] = 0
        return torch.tensor(values, dtype=self._torch_dtype())

    # Initialize and optionally freeze the GP and likelihood hyperparameters.
    def _initialize_hyperparameters(self, model, likelihood):
        if not self.config.fixed_hyperparameters:
            return
        model.mean_module.initialize(constant=self.config.fixed_mean)
        model.covariance_module.base_kernel.initialize(
            lengthscale=self.config.fixed_lengthscale
        )
        model.covariance_module.initialize(outputscale=self.config.fixed_outputscale)
        likelihood.initialize(noise=self.config.fixed_noise)
        for parameter in list(model.hyperparameters()) + list(likelihood.parameters()):
            parameter.requires_grad_(False)

    # Check that every fitted model and likelihood parameter remains finite.
    @staticmethod
    def _parameters_are_finite(model, likelihood):
        return all(
            torch.isfinite(parameter).all().item()
            for parameter in list(model.parameters()) + list(likelihood.parameters())
        )

    # Retry one GP calculation with increasing Cholesky jitter.
    @staticmethod
    def _with_cholesky_jitter(function):
        jitter_values = (1e-6, 1e-5, 1e-4, 1e-3)
        last_error = None
        for jitter in jitter_values:
            try:
                with gpytorch.settings.cholesky_jitter(float_value=jitter):
                    return function(), jitter
            except NotPSDError as error:
                last_error = error
        raise PMCOptimizationError(
            "PMC-GP covariance remained non-positive-definite after jitter escalation."
        ) from last_error

    # Fit the variational GP while processing every observation in every epoch.
    def fit(self, discrepancy, response, seed=0, sample_weight=None):
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer.")
        device = self._resolve_device(self.config.device)
        with _isolated_torch_state(
            seed, self.config.torch_num_threads, device
        ):
            return self._fit_isolated(
                discrepancy, response, seed, sample_weight, device
            )

    # Fit the variational GP after isolating process-global Torch state.
    def _fit_isolated(self, discrepancy, response, seed, sample_weight, device):
        discrepancy = self._discrepancy_array(discrepancy)
        response = self._training_array(response, "response")
        if len(discrepancy) != len(response):
            raise ValueError("discrepancy and response must have the same length.")
        if sample_weight is not None:
            sample_weight = self._training_array(sample_weight, "sample_weight")
            if len(sample_weight) != len(discrepancy):
                raise ValueError(
                    "sample_weight and discrepancy must have the same length."
                )
            if np.any(sample_weight <= 0):
                raise ValueError("sample_weight must contain only positive values.")
            scaled_weight = sample_weight / sample_weight.max()
            sample_weight = scaled_weight / scaled_weight.mean()
            if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight <= 0):
                raise ValueError(
                    "sample_weight normalization must produce positive finite values."
                )
        self._seed = seed
        with np.errstate(over="ignore", invalid="ignore"):
            x_mean = discrepancy.mean(axis=0)
            x_scale = discrepancy.std(axis=0)
            self._y_mean = float(response.mean())
            self._y_scale = float(response.std())
        if not np.all(np.isfinite(x_scale)) or np.any(x_scale == 0):
            raise ValueError(
                "every discrepancy feature must have positive finite "
                "empirical variance."
            )
        if not np.isfinite(self._y_scale) or self._y_scale == 0:
            raise ValueError("response must have positive finite empirical variance.")

        if discrepancy.ndim == 1:
            self._x_mean = float(x_mean)
            self._x_scale = float(x_scale)
        else:
            self._x_mean = x_mean.tolist()
            self._x_scale = x_scale.tolist()
        standardized_discrepancy = (
            discrepancy - np.asarray(self._x_mean)
        ) / np.asarray(self._x_scale)
        train_x = torch.tensor(
            standardized_discrepancy,
            dtype=self._torch_dtype(),
        )
        if train_x.ndim == 1:
            train_x = train_x.unsqueeze(-1)
        train_y = torch.tensor(
            (response - self._y_mean) / self._y_scale,
            dtype=self._torch_dtype(),
        )
        train_weight = (
            None
            if sample_weight is None
            else torch.tensor(sample_weight, dtype=self._torch_dtype())
        )
        if train_weight is not None and (
            not torch.isfinite(train_weight).all().item()
            or not torch.all(train_weight > 0).item()
        ):
            raise ValueError(
                "sample_weight normalization must remain positive and finite "
                "in float32."
            )
        if train_x.shape[1] == 1:
            self._standardized_discrepancy = (
                train_x[:, 0].detach().cpu().clone()
            )
        else:
            self._standardized_discrepancy = train_x.detach().cpu().clone()
        inducing_points = self._inducing_points(
            self._standardized_discrepancy.numpy()
        ).to(device)
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        if train_weight is not None:
            train_weight = train_weight.to(device)
        self._model = _SparseVariationalGP(
            inducing_points,
            self.config.learn_inducing_locations,
            self.config.kernel,
        ).to(device=device, dtype=self._torch_dtype())
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            device=device, dtype=self._torch_dtype()
        )
        self._initialize_hyperparameters(self._model, likelihood)
        trainable_hyperparameters = [
            parameter
            for parameter in list(self._model.hyperparameters())
            + list(likelihood.parameters())
            if parameter.requires_grad
        ]
        optimizer = (
            None
            if not trainable_hyperparameters
            else torch.optim.Adam(
                trainable_hyperparameters,
                lr=self.config.learning_rate,
            )
        )
        variational_optimizer = gpytorch.optim.NGD(
            self._model.variational_parameters(),
            num_data=len(train_x),
            lr=self.config.natural_gradient_learning_rate,
        )
        objective = gpytorch.mlls.VariationalELBO(
            likelihood, self._model, num_data=len(train_x)
        )

        self._model.train()
        likelihood.train()
        batch_size, batches_per_epoch, maximum_updates = self._update_budget(
            len(train_x)
        )
        maximum_epochs = maximum_updates // batches_per_epoch
        generator = torch.Generator(device=device).manual_seed(seed)
        best_loss = np.inf
        best_epoch = None
        best_model_state = None
        best_likelihood_state = None
        last_epoch_loss = np.nan
        stale_epochs = 0
        updates_completed = 0
        epochs_completed = 0
        observations_processed = 0
        used_all_observations_per_epoch = True
        stopped_early = False
        maximum_training_jitter = 0.0
        self._synchronize_device(device)
        started = time.perf_counter()
        for _ in range(maximum_epochs):
            epoch_loss = 0.0
            epoch_observations = 0
            if batch_size == len(train_x):
                batches = [(train_x, train_y, train_weight)]
            else:
                permutation = torch.randperm(
                    len(train_x), generator=generator, device=device
                )
                batches = (
                    (
                        train_x[index],
                        train_y[index],
                        None if train_weight is None else train_weight[index],
                    )
                    for start in range(0, len(train_x), batch_size)
                    for index in [permutation[start : start + batch_size]]
                )
            for batch_x, batch_y, batch_weight in batches:
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                variational_optimizer.zero_grad()

                # Evaluate one weighted or unweighted variational loss.
                def evaluate_loss():
                    model_output = self._model(batch_x)
                    if batch_weight is None:
                        return -objective(model_output, batch_y)
                    expected_log_probability = likelihood.expected_log_prob(
                        batch_y, model_output
                    )
                    weighted_log_probability = torch.mean(
                        batch_weight * expected_log_probability
                    )
                    kl_divergence = (
                        self._model.variational_strategy.kl_divergence().sum()
                        / len(train_x)
                    )
                    return -(weighted_log_probability - kl_divergence)

                loss, training_jitter = self._with_cholesky_jitter(evaluate_loss)
                maximum_training_jitter = max(
                    maximum_training_jitter, training_jitter
                )
                if not torch.isfinite(loss).item():
                    raise PMCOptimizationError(
                        "PMC-GP optimization produced a nonfinite loss."
                    )
                loss.backward()
                if optimizer is not None:
                    optimizer.step()
                variational_optimizer.step()
                batch_observations = len(batch_y)
                epoch_loss += float(loss.detach()) * batch_observations
                epoch_observations += batch_observations
                observations_processed += batch_observations
                updates_completed += 1
            epochs_completed += 1
            used_all_observations_per_epoch &= epoch_observations == len(train_x)
            epoch_loss /= epoch_observations
            last_epoch_loss = epoch_loss
            if not np.isfinite(epoch_loss):
                raise PMCOptimizationError(
                    "PMC-GP optimization produced a nonfinite epoch loss."
                )
            if not self._parameters_are_finite(self._model, likelihood):
                raise PMCOptimizationError(
                    "PMC-GP optimization produced nonfinite parameters."
                )
            required_improvement = self.config.relative_tolerance * max(
                abs(best_loss), 1.0
            )
            improved = epoch_loss < best_loss - required_improvement
            if not np.isfinite(best_loss) or improved:
                best_loss = epoch_loss
                best_epoch = epochs_completed
                best_model_state = copy.deepcopy(self._model.state_dict())
                best_likelihood_state = copy.deepcopy(likelihood.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if (
                epochs_completed >= self.config.minimum_epochs
                and stale_epochs >= self.config.patience
            ):
                stopped_early = True
                break
        self._synchronize_device(device)
        self._training_time_seconds = time.perf_counter() - started
        self._model.load_state_dict(best_model_state)
        likelihood.load_state_dict(best_likelihood_state)
        self._model.eval()
        self._diagnostics = {
            "pmc_version": __version__,
            "model_format_version": PMC_GP_MODEL_FORMAT_VERSION,
            "torch_version": str(torch.__version__),
            "gpytorch_version": str(gpytorch.__version__),
            **self._device_metadata(device),
            "requested_device": self.config.device,
            "fit_device": str(device),
            "fit_cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "dtype": self.config.dtype,
            "n_observations": len(train_x),
            "seed": self._seed,
            "requested_maximum_updates": self.config.maximum_updates,
            "effective_maximum_updates": maximum_updates,
            "batches_per_epoch": batches_per_epoch,
            "updates_completed": updates_completed,
            "epochs_completed": epochs_completed,
            "observations_processed": observations_processed,
            "used_all_observations_per_epoch": used_all_observations_per_epoch,
            "used_sample_weights": sample_weight is not None,
            "minimum_sample_weight": (
                np.nan if sample_weight is None else float(sample_weight.min())
            ),
            "raw_uq_small_sample_warning": (
                "Raw empirical-Bayes GP intervals were not coverage-reliable "
                "below n=1000 in held-out validation."
                if len(train_x) < 1000
                else None
            ),
            "stopped_early": stopped_early,
            "termination_reason": (
                "early_stopping" if stopped_early else "maximum_epochs"
            ),
            "best_epoch": best_epoch,
            "best_epoch_loss": best_loss,
            "last_epoch_loss": last_epoch_loss,
            "final_epoch_loss": best_loss,
            "final_lengthscale": float(
                self._model.covariance_module.base_kernel.lengthscale.detach().item()
            ),
            "final_outputscale": float(
                self._model.covariance_module.outputscale.detach().item()
            ),
            "final_noise": float(likelihood.noise.detach().item()),
            "maximum_training_jitter": maximum_training_jitter,
        }
        return self

    # Predict the conditional target and an approximate latent posterior interval.
    def predict(self, discrepancy=0.0, confidence_level=0.95):
        if self._model is None:
            raise RuntimeError("fit must be called before predict.")
        if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one.")
        feature_count = (
            1
            if self._standardized_discrepancy.ndim == 1
            else self._standardized_discrepancy.shape[1]
        )
        query_discrepancy = np.asarray(discrepancy, dtype=float)
        if feature_count == 1:
            if query_discrepancy.ndim != 0:
                raise ValueError("discrepancy query must be scalar.")
            query_discrepancy = float(query_discrepancy)
            if not np.isfinite(query_discrepancy):
                raise ValueError("discrepancy must be finite.")
            standardized_query = (
                query_discrepancy - self._x_mean
            ) / self._x_scale
            query_values = [[standardized_query]]
        else:
            if query_discrepancy.ndim != 1 or len(query_discrepancy) != feature_count:
                raise ValueError(
                    f"discrepancy query must contain exactly {feature_count} features."
                )
            if not np.all(np.isfinite(query_discrepancy)):
                raise ValueError("discrepancy must contain only finite values.")
            standardized_query = (
                query_discrepancy - np.asarray(self._x_mean)
            ) / np.asarray(self._x_scale)
            query_values = standardized_query[np.newaxis, :]
        query = torch.tensor(
            query_values,
            dtype=self._torch_dtype(),
            device=self._model_device(),
        )
        # Evaluate the latent posterior with controlled covariance jitter.
        def evaluate_latent():
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                return self._model(query)

        latent, prediction_jitter = self._with_cholesky_jitter(evaluate_latent)
        estimate = float(latent.mean.item() * self._y_scale + self._y_mean)
        posterior_standard_deviation = float(latent.stddev.item() * self._y_scale)
        if not np.isfinite(estimate):
            raise PMCOptimizationError(
                "PMC-GP prediction produced a nonfinite estimate."
            )
        if (
            not np.isfinite(posterior_standard_deviation)
            or posterior_standard_deviation <= 0
        ):
            raise PMCOptimizationError(
                "PMC-GP prediction produced an invalid posterior standard deviation."
            )
        quantile = float(norm.ppf((1 + confidence_level) / 2))
        lengthscale = self._diagnostics["final_lengthscale"]
        if feature_count == 1:
            standardized_query = float(query.item())
            raw_distances = torch.abs(
                self._standardized_discrepancy - standardized_query
            )
            query_outside_training_range = bool(
                standardized_query < self._standardized_discrepancy.min().item()
                or standardized_query
                > self._standardized_discrepancy.max().item()
            )
            query_metadata = standardized_query
            discrepancy_metadata = query_discrepancy
        else:
            standardized_query = query[0].detach().cpu()
            raw_distances = torch.linalg.vector_norm(
                self._standardized_discrepancy - standardized_query, dim=1
            )
            query_outside_training_range = bool(
                torch.any(
                    standardized_query < self._standardized_discrepancy.min(dim=0).values
                ).item()
                or torch.any(
                    standardized_query > self._standardized_discrepancy.max(dim=0).values
                ).item()
            )
            query_metadata = standardized_query.tolist()
            discrepancy_metadata = query_discrepancy.tolist()
        distances = raw_distances / lengthscale
        if self.config.kernel == "matern_32":
            scaled_distances = np.sqrt(3) * distances
            local_weights = (1 + scaled_distances) * torch.exp(-scaled_distances)
        elif self.config.kernel == "matern_52":
            scaled_distances = np.sqrt(5) * distances
            local_weights = (
                1 + scaled_distances + 5 * distances**2 / 3
            ) * torch.exp(-scaled_distances)
        else:
            local_weights = torch.exp(-0.5 * distances**2)
        local_weight_sum = float(local_weights.sum().item())
        squared_weight_sum = float(torch.square(local_weights).sum().item())
        local_effective_sample_size = (
            0.0
            if squared_weight_sum == 0
            else float(local_weight_sum**2 / squared_weight_sum)
        )
        diagnostics = dict(self._diagnostics)
        diagnostics.update(
            {
                "standardized_query_discrepancy": query_metadata,
                "query_outside_training_range": query_outside_training_range,
                "nearest_standardized_discrepancy_distance": float(
                    torch.min(raw_distances).item()
                ),
                "local_kernel_weight_sum": local_weight_sum,
                "local_effective_sample_size": local_effective_sample_size,
                "local_effective_sample_fraction": (
                    local_effective_sample_size / len(self._standardized_discrepancy)
                ),
                "prediction_jitter": prediction_jitter,
            }
        )
        return PMCResult(
            estimate=estimate,
            posterior_standard_deviation=posterior_standard_deviation,
            interval_lower=estimate - quantile * posterior_standard_deviation,
            interval_upper=estimate + quantile * posterior_standard_deviation,
            confidence_level=confidence_level,
            query_discrepancy=discrepancy_metadata,
            interval_method="variational_empirical_bayes_latent_posterior",
            training_time_seconds=self._training_time_seconds,
            diagnostics=diagnostics,
            configuration=asdict(self.config),
        )
