##############################
# Import necessary libraries #
##############################

# Standard libraries
import json
import os
import platform
import time
from collections.abc import Mapping
from numbers import Integral, Real

# Third-party libraries
import numpy as np
import torch

# PMC library
from .algorithms import (
    CNFPMC,
    MLPPMC,
    BetaOracleSampling,
    GaussianOracleSampling,
    GaussianPMC,
    LinearPMC,
    consensus_discrepancy,
    oracle_analytic_point_estimate,
    oracle_analytic_point_estimate_beta,
    selected_component_index,
)
from .config import (
    _COMPACT_RESULT_DIMENSION,
    _SCALAR_KEYS_BY_FORM,
    _ULTRA_COMPACT_RESULT_DIMENSION,
    _compute_dgp_schema,
    _semantic_fingerprint,
)
from .dgp import BetaDGP, BivariateGaussianDGP, MultivariateGaussianDGP, UnivariateGaussianDGP
from .metrics import check_convergence, kl, mse, wasserstein
from .pmc_gp import GaussianProcessPMC, PMCConfig, recommended_pmc_gp_config
from .utils import dgp_param_column, parameters_to_filename, weights_to_filename
from .visualization import (
    plot_discrepancy_distribution,
    plot_inference_distribution,
    plot_product_distribution,
    plot_training_loss,
)

###############
# Simulations #
###############

# Vector-per-node DGP-parameter keys for multidimensional Gaussian forms
_VECTOR_DGP_KEYS = ('mu', 'cov_diag')

# Symmetric-matrix-per-node DGP-parameter keys (cov for bivariate_Gaussian/multivariate_Gaussian)
_MATRIX_DGP_KEYS = ('cov',)

# Engine-only result columns, populated for sampling/PMC rows and None for other algorithms
_PMC_ENGINE_COLUMNS = (
    'algorithm', 'sampling_method', 'discrepancy_method', 'n_transforms',
    'hidden_features',
    'n_training_steps', 'optimizer', 'weight_decay', 'base_learning_rate',
    'learning_rate_schedule',
    'batch_size', 'h', 'degree',
    'input_degree', 'activation', 'discrepancy_centering', 'ensemble_size',
    'regression_loss', 'huber_delta', 'loss_weighting', 'target_bandwidth_quantile',
    'target_bandwidth_scale', 'alpha',
    'gp_kernel', 'gp_inducing_count', 'gp_maximum_updates',
    'gp_natural_gradient_learning_rate', 'gp_minimum_epochs', 'gp_patience',
    'gp_relative_tolerance', 'gp_learn_inducing_locations',
    'gp_torch_num_threads', 'gp_confidence_level',
    'gp_local_neighborhood_exponent', 'gp_minimum_local_count',
    'gp_maximum_local_count', 'gp_full_observation_count',
    'gp_requested_local_count', 'gp_actual_local_count', 'gp_local_fraction',
    'gp_neighborhood_radius',
    'gp_posterior_standard_deviation', 'gp_interval_lower', 'gp_interval_upper',
    'gp_interval_method', 'gp_diagnostics',
    'converged', 'relative_improvement', 'training_log_likelihood',
)

# Fusion-only result columns, always None on the Python side (populated by R's fusion runner)
_FUSION_ONLY_COLUMNS = (
    'coalescence_time', 'n_time_steps', 'rho_acceptance_probability',
    'Q_acceptance_probability', 'acceptance_probability', 'final_ESS', 'final_CESS',
    'beta_fitted_nodes', 'beta_jacobian_shares', 'beta_max_terminal_spread',
    'beta_preprocessing_seconds',
)

# Hardware/context columns shared by Python and R simulation rows
_HARDWARE_CONTEXT_COLUMNS = (
    'hostname', 'platform_system', 'platform_release', 'platform_machine',
    'cpu_count_logical', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'torch_cuda_available',
    'torch_cuda_device_name', 'simulation_device',
)


# Report whether a result schema omits quadratic parameter expansions.
def _is_compact_result_schema(schema):
    return schema.get(
        'compact', schema.get('max_p', 0) >= _COMPACT_RESULT_DIMENSION
    )

# Report whether a result schema uses only a semantic fingerprint for parameters
def _is_ultra_compact_result_schema(schema):
    return schema.get(
        'ultra_compact',
        schema.get('max_p', 0) >= _ULTRA_COMPACT_RESULT_DIMENSION,
    )


# Adapt the scalar public PMC-GP estimator to the experiment engine contract.
class _GaussianProcessPMCAdapter:
    # Initialize the transformed target, GP estimator, query interval, and seed.
    def __init__(
        self, h, configuration, confidence_level, seed,
        loss_weighting='uniform', target_bandwidth_quantile=None,
        target_bandwidth_scale=1.0, alpha=None,
        local_neighborhood_exponent=None, minimum_local_count=256,
        maximum_local_count=131072,
    ):
        self.h = h
        self.configuration = configuration
        self.estimator = GaussianProcessPMC(configuration)
        self.estimators = []
        self.confidence_level = confidence_level
        self.seed = 0 if seed is None else seed
        self.loss_weighting = loss_weighting
        self.target_bandwidth_quantile = target_bandwidth_quantile
        self.target_bandwidth_scale = target_bandwidth_scale
        self.alpha = alpha
        self.local_neighborhood_exponent = local_neighborhood_exponent
        self.minimum_local_count = minimum_local_count
        self.maximum_local_count = maximum_local_count
        self.training_loss = None
        self.last_result = None
        self.last_results = []
        self.local_neighborhood_diagnostics = None
        self.discrepancy_feature_count = None
        self._validate_local_neighborhood()

    # Validate the optional nearest-zero neighborhood configuration.
    def _validate_local_neighborhood(self):
        if self.local_neighborhood_exponent is not None and (
            isinstance(self.local_neighborhood_exponent, bool)
            or not isinstance(self.local_neighborhood_exponent, Real)
            or not 0 < self.local_neighborhood_exponent <= 1
        ):
            raise ValueError(
                "gp_local_neighborhood_exponent must be numeric and in (0, 1]."
            )
        for name, value in (
            ("gp_minimum_local_count", self.minimum_local_count),
            ("gp_maximum_local_count", self.maximum_local_count),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 2
            ):
                raise ValueError(f"{name} must be an integer of at least two.")
        if self.minimum_local_count > self.maximum_local_count:
            raise ValueError(
                "gp_minimum_local_count must not exceed "
                "gp_maximum_local_count."
            )

    # Select a deterministic growing neighborhood nearest zero discrepancy.
    def _local_indices(self, discrepancy):
        full_count = len(discrepancy)
        if self.local_neighborhood_exponent is None:
            return None, full_count
        requested_count = int(
            np.ceil(full_count ** self.local_neighborhood_exponent)
        )
        requested_count = max(requested_count, self.minimum_local_count)
        requested_count = min(
            requested_count, self.maximum_local_count, full_count
        )
        distances = self._distances(discrepancy)
        indices = np.argsort(distances, kind='stable')[:requested_count]
        return indices, requested_count

    # Measure scalar absolute or vector Euclidean distance from zero.
    @staticmethod
    def _distances(discrepancy):
        if discrepancy.shape[1] == 1:
            return np.abs(discrepancy[:, 0])
        return np.linalg.norm(discrepancy, axis=1)

    # Compute positive sample weights around the zero-discrepancy target.
    def _sample_weight(self, discrepancy):
        if self.loss_weighting == 'uniform':
            return None
        if self.loss_weighting not in ('gaussian', 'alpha_mixture'):
            raise ValueError(
                'loss_weighting must be "uniform", "gaussian", or '
                '"alpha_mixture".'
            )
        if (
            isinstance(self.target_bandwidth_quantile, bool)
            or not isinstance(self.target_bandwidth_quantile, Real)
            or not 0 < self.target_bandwidth_quantile <= 1
        ):
            raise ValueError(
                "target_bandwidth_quantile must be numeric and in (0, 1] "
                "for gaussian or alpha_mixture loss weighting."
            )
        if (
            isinstance(self.target_bandwidth_scale, bool)
            or not isinstance(self.target_bandwidth_scale, Real)
            or self.target_bandwidth_scale <= 0
        ):
            raise ValueError(
                "target_bandwidth_scale must be numeric and positive for "
                "gaussian or alpha_mixture loss weighting."
            )
        if self.loss_weighting == 'alpha_mixture' and (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, Real)
            or not 0 <= self.alpha <= 1
        ):
            raise ValueError(
                "alpha must be numeric and in [0, 1] for alpha_mixture "
                "loss weighting."
            )
        distances = self._distances(discrepancy)
        bandwidth = (
            np.quantile(distances, self.target_bandwidth_quantile)
            * self.target_bandwidth_scale
        )
        bandwidth = max(float(bandwidth), np.finfo(float).eps)
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        weights = np.maximum(weights, np.finfo(np.float32).tiny)
        if self.loss_weighting == 'alpha_mixture':
            weights = (1 - self.alpha) + self.alpha * weights
        return weights

    # Fit the scalar GP to every transformed consensus-discrepancy pair.
    def fit(self, consensus, discrepancy):
        self.last_result = None
        self.last_results = []
        discrepancy_array = np.asarray(
            discrepancy.detach().cpu() if torch.is_tensor(discrepancy) else discrepancy,
            dtype=float,
        )
        if discrepancy_array.ndim != 2 or discrepancy_array.shape[1] < 1:
            raise ValueError(
                "PMC-GP discrepancy must be a two-dimensional array with at "
                "least one feature."
            )
        if not np.all(np.isfinite(discrepancy_array)):
            raise ValueError("PMC-GP discrepancy must contain only finite values.")
        self.discrepancy_feature_count = discrepancy_array.shape[1]
        indices, requested_count = self._local_indices(discrepancy_array)
        if indices is None:
            selected_consensus = consensus
            selected_discrepancy = discrepancy_array
        else:
            selected_discrepancy = discrepancy_array[indices]
            if torch.is_tensor(consensus):
                selected_consensus = consensus[
                    torch.as_tensor(indices, device=consensus.device)
                ]
            else:
                selected_consensus = np.asarray(consensus)[indices]
        response = (
            selected_consensus
            if self.h is None
            else self.h(selected_consensus)
        )
        response_array = np.asarray(
            response.detach().cpu() if torch.is_tensor(response) else response,
            dtype=float,
        )
        if response_array.ndim != 2 or response_array.shape[1] < 1:
            raise ValueError(
                "PMC-GP response must be a two-dimensional array with at "
                "least one output."
            )
        if not np.all(np.isfinite(response_array)):
            raise ValueError("PMC-GP response must contain only finite values.")
        if len(response_array) != len(selected_discrepancy):
            raise ValueError(
                "PMC-GP discrepancy and response must have the same number "
                "of observations."
            )
        distances = self._distances(selected_discrepancy)
        self.local_neighborhood_diagnostics = {
            'local_neighborhood_strategy': (
                'full_sample'
                if self.local_neighborhood_exponent is None
                else 'nearest_zero_power'
            ),
            'local_neighborhood_exponent': self.local_neighborhood_exponent,
            'full_observation_count': len(discrepancy_array),
            'requested_local_count': requested_count,
            'actual_local_count': len(selected_discrepancy),
            'local_fraction': len(selected_discrepancy) / len(discrepancy_array),
            'neighborhood_radius': float(distances.max()),
        }
        sample_weight = self._sample_weight(selected_discrepancy)
        estimator_discrepancy = (
            selected_discrepancy[:, 0]
            if self.discrepancy_feature_count == 1
            else selected_discrepancy
        )
        self.estimators = [self.estimator]
        self.estimators.extend(
            GaussianProcessPMC(self.configuration)
            for _ in range(response_array.shape[1] - 1)
        )
        for dimension, estimator in enumerate(self.estimators):
            estimator.fit(
                estimator_discrepancy,
                response_array[:, dimension],
                seed=self.seed,
                sample_weight=sample_weight,
            )

    # Predict ordered conditional targets and retain per-dimension uncertainty.
    def predict(self, discrepancy_condition):
        if self.discrepancy_feature_count is None or not self.estimators:
            raise RuntimeError("fit must be called before predict.")
        query_values = (
            discrepancy_condition.detach().cpu()
            if torch.is_tensor(discrepancy_condition)
            else discrepancy_condition
        )
        query = np.asarray(query_values, dtype=float).reshape(-1)
        if len(query) != self.discrepancy_feature_count:
            raise ValueError(
                "PMC-GP prediction query must match the fitted discrepancy "
                "feature count."
            )
        if not np.all(np.isfinite(query)):
            raise ValueError("PMC-GP prediction query must contain only finite values.")
        estimator_query = float(query[0]) if len(query) == 1 else query
        self.last_results = [
            estimator.predict(
                discrepancy=estimator_query,
                confidence_level=self.confidence_level,
            )
            for estimator in self.estimators
        ]
        for result in self.last_results:
            result.diagnostics.update(self.local_neighborhood_diagnostics)
        self.last_result = self.last_results[0]
        return np.array([result.estimate for result in self.last_results])

# Build a diagnostic-plot filename encoding the config/DGP parameters
def _plot_filename(prefix, config, parameters_str, sampling_method, w, include_preprocessing=False):
    preprocessing_part = (
        f"preprocessing_{config['preprocessing']}_" if include_preprocessing else ''
    )
    return (
        f"{prefix}_{config['parametric_form']}_{preprocessing_part}n_{config['n']}_"
        f"{parameters_str}_sampling_{sampling_method}_{weights_to_filename(w)}.pdf"
    )


# Build the DGP and the corresponding product-distribution parameters
def _build_dgp(parametric_form, dgp_parameters):
    # Univariate Gaussian: product of independent per-node Gaussians
    if parametric_form == 'univariate_Gaussian':
        dgp = UnivariateGaussianDGP(mu=dgp_parameters['mu'], sigma=dgp_parameters['sigma'])
        product_parameters = {'mu': dgp.mu_product, 'sigma': dgp.sigma_product}
    # Bivariate Gaussian: product of independent per-node 2D Gaussian vectors
    elif parametric_form == 'bivariate_Gaussian':
        dgp = BivariateGaussianDGP(mu=dgp_parameters['mu'], cov=dgp_parameters['cov'])
        product_parameters = {'mu': dgp.mu_product, 'cov': dgp.cov_product}
    # Multivariate Gaussian: product of independent per-node Gaussian vectors
    elif parametric_form == 'multivariate_Gaussian':
        dgp = MultivariateGaussianDGP(
            mu=dgp_parameters['mu'],
            cov=dgp_parameters.get('cov'),
            cov_diag=dgp_parameters.get('cov_diag'),
        )
        if hasattr(dgp, 'cov_product_diag'):
            product_parameters = {
                'mu': dgp.mu_product,
                'cov_diag': dgp.cov_product_diag,
            }
        else:
            product_parameters = {'mu': dgp.mu_product, 'cov': dgp.cov_product}
    # Beta: product of independent per-node Beta distributions
    elif parametric_form == 'Beta':
        dgp = BetaDGP(a=dgp_parameters['a'], b=dgp_parameters['b'])
        product_parameters = {'a': dgp.a_product, 'b': dgp.b_product}
    else:
        raise ValueError(f"Unsupported parametric form: {parametric_form}")
    return dgp, product_parameters


# Compute the dimension-normalized coordinate sum for NumPy or Torch arrays
def _normalized_sum(consensus):
    if torch.is_tensor(consensus):
        return consensus.sum(dim=-1, keepdim=True) / np.sqrt(
            consensus.shape[-1]
        )
    return np.sum(consensus, axis=-1, keepdims=True) / np.sqrt(
        consensus.shape[-1]
    )


# Resolves the configured batch size from nominal and realized sample sizes
def _resolve_batch_size(batch_size, n, n_sample):
    if batch_size is None:
        return None
    if isinstance(batch_size, bool):
        raise ValueError(
            'batch_size must be a positive integer, "adaptive", or None.'
        )
    if isinstance(batch_size, Integral):
        if batch_size <= 0:
            raise ValueError(
                'batch_size must be a positive integer, "adaptive", or None.'
            )
        return batch_size
    if not isinstance(batch_size, str) or batch_size != 'adaptive':
        raise ValueError(
            'batch_size must be a positive integer, "adaptive", or None.'
        )
    batch_limit = 256 if n <= 10000 else 1024
    return int(min(n_sample, batch_limit))


# Select whether a config should use CUDA for Torch-backed estimation
def _uses_cuda(config):
    if config.get('algorithm') != 'PMC':
        return False
    if config.get('method') == 'MLP':
        return torch.cuda.is_available()
    if config.get('method') != 'GP' or config.get('gp_device', 'cpu') != 'cuda':
        return False
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PMC-GP requested CUDA, but PyTorch reports that CUDA is unavailable."
        )
    return True


# Build cheap hardware/context values for runtime comparisons
def _hardware_context_values(device):
    device_type = device.type
    torch_cuda_device_name = None
    if device_type == 'cuda':
        torch_cuda_device_name = torch.cuda.get_device_name(device)
    values = (
        platform.node(),
        platform.system(),
        platform.release(),
        platform.machine(),
        os.cpu_count(),
        os.environ.get('OMP_NUM_THREADS'),
        os.environ.get('MKL_NUM_THREADS'),
        os.environ.get('OPENBLAS_NUM_THREADS'),
        os.environ.get('CUDA_VISIBLE_DEVICES'),
        torch.cuda.is_available(),
        torch_cuda_device_name,
        device_type,
    )
    return dict(zip(_HARDWARE_CONTEXT_COLUMNS, values))


# Create CUDA events when GPU-side elapsed time should be recorded
def _cuda_timer(device):
    if device.type != 'cuda':
        return None
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


# Record the first CUDA event when one is available
def _cuda_timer_start(cuda_timer):
    if cuda_timer is None:
        return
    start_event, _ = cuda_timer
    start_event.record()


# Record the second CUDA event and return elapsed seconds
def _cuda_timer_stop(cuda_timer):
    if cuda_timer is None:
        return None
    start_event, end_event = cuda_timer
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / 1000


# Validate scalar or target-specific base learning rates before training
def _validate_base_learning_rate(base_learning_rate, h_values, algorithm):
    if isinstance(base_learning_rate, Mapping):
        if algorithm != 'PMC':
            raise ValueError(
                "base_learning_rate mappings are supported only for PMC."
            )
        if (
            len(base_learning_rate) != len(h_values)
            or set(base_learning_rate) != set(h_values)
        ):
            raise ValueError(
                "base_learning_rate mapping keys must exactly match h_values."
            )
        for h_name, value in base_learning_rate.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    f"base_learning_rate for {h_name!r} must be numeric."
                )
        return
    if base_learning_rate is not None and (
        isinstance(base_learning_rate, bool)
        or not isinstance(base_learning_rate, Real)
    ):
        raise ValueError("base_learning_rate must be numeric.")


# Instantiate the estimator for the given algorithm/method combination
def _build_estimator(
    algorithm, parametric_form, method_name, n_transforms, hidden_features,
    n_training_steps, optimizer, learning_rate, batch_size, preprocessing, device, h,
    use_cosine_schedule=False, degree=1, loss_weighting='uniform',
    target_bandwidth_quantile=None, target_bandwidth_scale=1.0, alpha=None,
    input_degree=1, activation='relu', discrepancy_centering='target',
    ensemble_size=1, weight_decay=0.0, regression_loss='mse', huber_delta=1.0,
    h_config=None, seed=None, gp_kernel=None, gp_inducing_count=8,
    gp_device='cpu', gp_maximum_updates=None,
    gp_natural_gradient_learning_rate=0.03,
    gp_minimum_epochs=10, gp_patience=3, gp_relative_tolerance=1e-4,
    gp_learn_inducing_locations=True, gp_torch_num_threads=None,
    gp_confidence_level=0.95, gp_local_neighborhood_exponent=None,
    gp_minimum_local_count=256, gp_maximum_local_count=131072,
    gp_full_batch=False, gp_fixed_hyperparameters=False, gp_fixed_mean=0.0,
    gp_fixed_lengthscale=1.0, gp_fixed_outputscale=1.0, gp_fixed_noise=1.0,
    gp_dtype='float32',
):
    # Oracle baseline: analytic product distribution, bypasses PMC entirely
    if algorithm == 'Oracle':
        if parametric_form == 'Beta':
            return BetaOracleSampling()
        return GaussianOracleSampling()

    # Kwargs shared by every estimator class below
    common_kwargs = dict(
        hidden_features=hidden_features,
        n_training_steps=n_training_steps,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        preprocessing=preprocessing,
        device=device
    )

    if algorithm == 'PMC' and method_name in {'Gaussian', 'CNF'}:
        # Analytic linear-Gaussian conditional of consensus on discrepancy
        if method_name == 'Gaussian':
            return GaussianPMC(**common_kwargs)
        # Conditional normalizing flow of consensus given discrepancy
        return CNFPMC(
            n_transforms=n_transforms, use_cosine_schedule=use_cosine_schedule,
            **common_kwargs
        )
    # PMC: linear (degree-adjustable polynomial) or MLP regression of consensus on discrepancy
    if method_name == 'linear':
        return LinearPMC(h=h, degree=degree, **common_kwargs)
    if method_name == 'GP':
        kernel = (
            recommended_pmc_gp_config(h_config).kernel
            if gp_kernel is None
            else gp_kernel
        )
        configuration = PMCConfig(
            device=gp_device,
            inducing_count=gp_inducing_count,
            maximum_updates=gp_maximum_updates,
            batch_size=8192 if batch_size is None else batch_size,
            learning_rate=0.03 if learning_rate is None else learning_rate,
            natural_gradient_learning_rate=gp_natural_gradient_learning_rate,
            minimum_epochs=gp_minimum_epochs,
            patience=gp_patience,
            relative_tolerance=gp_relative_tolerance,
            learn_inducing_locations=gp_learn_inducing_locations,
            full_batch=gp_full_batch,
            fixed_hyperparameters=gp_fixed_hyperparameters,
            fixed_mean=gp_fixed_mean,
            fixed_lengthscale=gp_fixed_lengthscale,
            fixed_outputscale=gp_fixed_outputscale,
            fixed_noise=gp_fixed_noise,
            dtype=gp_dtype,
            torch_num_threads=gp_torch_num_threads,
            kernel=kernel,
        )
        return _GaussianProcessPMCAdapter(
            h,
            configuration,
            gp_confidence_level,
            seed,
            loss_weighting=loss_weighting,
            target_bandwidth_quantile=target_bandwidth_quantile,
            target_bandwidth_scale=target_bandwidth_scale,
            alpha=alpha,
            local_neighborhood_exponent=gp_local_neighborhood_exponent,
            minimum_local_count=gp_minimum_local_count,
            maximum_local_count=gp_maximum_local_count,
        )
    return MLPPMC(
        h=h, use_cosine_schedule=use_cosine_schedule,
        loss_weighting=loss_weighting,
        target_bandwidth_quantile=target_bandwidth_quantile,
        target_bandwidth_scale=target_bandwidth_scale,
        alpha=alpha,
        input_degree=input_degree,
        activation=activation,
        discrepancy_centering=discrepancy_centering,
        ensemble_size=ensemble_size,
        weight_decay=weight_decay,
        regression_loss=regression_loss,
        huber_delta=huber_delta,
        **common_kwargs
    )


# Sample from the fitted PMC estimator, evaluate KL/Wasserstein, and return raw samples
def _evaluate_pmc_samples(
    estimator, algorithm, dgp, discrepancy_condition, n_inference,
    plots_directory, config, parameters_str, sampling_method, w
):
    # Sample consensus | discrepancy = 0 (unconditionally, for the oracle baseline)
    if algorithm == 'Oracle':
        consensus_inference, log_p_consensus_inference = estimator.sample(
            n_inference=n_inference
        )
    else:
        consensus_inference, log_p_consensus_inference = estimator.sample(
            discrepancy_condition=discrepancy_condition,
            n_inference=n_inference
        )

    if plots_directory is not None:
        # CPU copy of discrepancy_condition for the estimator-conditional plots below
        discrepancy_condition_cpu = (
            discrepancy_condition.cpu() if discrepancy_condition is not None else None
        )

        # Plot the raw sample distribution
        plot_inference_distribution(
            consensus_inference,
            mode='samples',
            dgp=dgp,
            show=False,
            save=True,
            filename=_plot_filename(
                'inference_distribution_samples', config, parameters_str, sampling_method, w,
                include_preprocessing=True
            ),
            plots_directory=plots_directory
        )

        # Plot the estimated conditional pdf; not applicable to the oracle baseline
        if algorithm != 'Oracle':
            plot_inference_distribution(
                consensus_inference,
                mode='pdf',
                estimator=estimator,
                discrepancy_condition=discrepancy_condition_cpu,
                dgp=dgp,
                show=False,
                save=True,
                filename=_plot_filename(
                    'inference_distribution_pdf', config, parameters_str, sampling_method, w,
                    include_preprocessing=True
                ),
                plots_directory=plots_directory
            )

        # Plot the empirical CDF of the samples
        plot_inference_distribution(
            consensus_inference,
            mode='cdf',
            dgp=dgp,
            show=False,
            save=True,
            filename=_plot_filename(
                'inference_distribution_empirical_cdf', config, parameters_str, sampling_method, w,
                include_preprocessing=True
            ),
            plots_directory=plots_directory
        )

        # Plot the estimated conditional CDF; not applicable to the oracle baseline
        if algorithm != 'Oracle':
            plot_inference_distribution(
                consensus_inference,
                mode='cdf',
                estimator=estimator,
                discrepancy_condition=discrepancy_condition_cpu,
                dgp=dgp,
                show=False,
                save=True,
                filename=_plot_filename(
                    'inference_distribution_cdf', config, parameters_str, sampling_method, w,
                    include_preprocessing=True
                ),
                plots_directory=plots_directory
            )

    # Evaluate the inference using KL divergence and Wasserstein distance
    kl_val = kl(
        dgp, consensus_inference, log_p_consensus_inference,
        estimator=estimator, discrepancy_condition=discrepancy_condition
    )
    w1_val, w2_val = wasserstein(
        dgp, consensus_inference, n_subsample=2000,
        estimator=estimator, discrepancy_condition=discrepancy_condition
    )
    eval_metrics = {
        'kl': kl_val, 'w1': w1_val, 'w2': w2_val, 'mse': None, 'C_hat': None, 'true_mean': None
    }
    return eval_metrics, consensus_inference


# Select the configured component from the full product-distribution mean
def _selected_product_mean(dgp, h, h_config, component_index):
    true_mean = np.array(
        dgp.mean_product(h=h, h_name=h_config)
    ).flatten()
    if component_index is None:
        return true_mean
    return true_mean[component_index:component_index + 1]


# Compute the point estimate for PMC or Oracle, and score it
def _pmc_oracle_point_estimate(
    algorithm, estimator, consensus_inference, dgp, h_config, h,
    component_index=None,
):
    if algorithm == 'Oracle':
        if dgp.parametric_form == 'Beta':
            consensus_hat = oracle_analytic_point_estimate_beta(
                h_config, estimator.a_product, estimator.b_product
            )
        else:
            consensus_hat = oracle_analytic_point_estimate(
                h_config, estimator.mu_product, estimator.cov_product
            )
    else:
        consensus_hat = (
            np.mean(h(consensus_inference), axis=0) if h is not None
            else np.mean(consensus_inference, axis=0)
        )

    # Keep every dimension of the point estimate, not just the first, for the results row
    true_mean_vec = _selected_product_mean(
        dgp, h, h_config, component_index
    )
    consensus_hat_vec = np.array(consensus_hat).flatten()
    if component_index is not None and algorithm == 'Oracle':
        consensus_hat_vec = consensus_hat_vec[
            component_index:component_index + 1
        ]
    mse_val = mse(dgp, consensus_hat_vec, true_mean=true_mean_vec)

    return {'mse': mse_val, 'C_hat': consensus_hat_vec, 'true_mean': true_mean_vec}


# Compute the PMC point estimate of E[h(consensus) | discrepancy = 0] and its MSE
def _evaluate_pmc(
    estimator, dgp, discrepancy_condition, h, h_config=None,
    component_index=None,
):
    # Predict E[h(consensus) | discrepancy = 0]
    consensus_hat = estimator.predict(discrepancy_condition)

    # Keep every dimension of the point estimate, not just the first, for the results row
    consensus_hat_vec = np.array(consensus_hat).flatten()
    true_mean_vec = _selected_product_mean(
        dgp, h, h_config, component_index
    )
    mse_val = mse(dgp, consensus_hat_vec, true_mean=true_mean_vec)

    return {
        'kl': None, 'w1': None, 'w2': None, 'mse': mse_val,
        'C_hat': consensus_hat_vec, 'true_mean': true_mean_vec
    }


# Yields (key, node, dims) tuples spanning every DGP-parameter column for the given schema
def _iter_dgp_param_specs(schema):
    if _is_ultra_compact_result_schema(schema):
        return
    for key in schema['scalar_keys']:
        for node in range(1, schema['max_num_nodes'] + 1):
            yield key, node, ()
    if _is_compact_result_schema(schema):
        return
    for key in schema.get('vector_keys', ('mu',)):
        for node in range(1, schema['max_num_nodes'] + 1):
            for dim in range(1, schema['max_p'] + 1):
                yield key, node, (dim,)
    if not _is_compact_result_schema(schema):
        for key in schema.get('matrix_keys', _MATRIX_DGP_KEYS):
            for node in range(1, schema['max_num_nodes'] + 1):
                for d1 in range(1, schema['max_p'] + 1):
                    for d2 in range(d1, schema['max_p'] + 1):
                        yield key, node, (d1, d2)


# Build the ordered list of DGP-parameter CSV column names for the given schema
def _dgp_param_columns(schema):
    return [dgp_param_column(key, node, *dims) for key, node, dims in _iter_dgp_param_specs(schema)]


# Build the DGP-parameter values (None where inapplicable), matching _dgp_param_columns order
def _dgp_param_values(schema, parametric_form, dgp_parameters):
    scalar_keys = _SCALAR_KEYS_BY_FORM.get(parametric_form, ())
    is_vector_form = parametric_form in ('bivariate_Gaussian', 'multivariate_Gaussian')
    mu_list = dgp_parameters.get('mu') if is_vector_form else None
    cov_diag_list = dgp_parameters.get('cov_diag') if is_vector_form else None
    cov_list = dgp_parameters.get('cov') if is_vector_form else None
    if is_vector_form:
        num_nodes = len(mu_list) if mu_list is not None else 0
    elif scalar_keys:
        num_nodes = len(dgp_parameters.get(scalar_keys[0], []))
    else:
        num_nodes = 0

    values = {}
    for key, node, dims in _iter_dgp_param_specs(schema):
        column = dgp_param_column(key, node, *dims)
        value = None
        if not dims and key in scalar_keys and node <= num_nodes:
            value = dgp_parameters[key][node - 1]
        elif len(dims) == 1 and is_vector_form and node <= num_nodes:
            dim = dims[0]
            vector_list = mu_list if key == 'mu' else cov_diag_list
            if vector_list is not None and dim <= len(vector_list[node - 1]):
                value = vector_list[node - 1][dim - 1]
        elif len(dims) == 2 and is_vector_form and node <= num_nodes:
            d1, d2 = dims
            node_cov = cov_list[node - 1]
            if d1 <= len(node_cov) and d2 <= len(node_cov[0]):
                value = node_cov[d1 - 1][d2 - 1]
        values[column] = value
    return values


# Build the ordered list of sampling-weight-matrix CSV column names for the given schema
def _weight_columns(schema):
    if _is_compact_result_schema(schema):
        return []
    max_dim = max(schema['max_p'], 1)
    return [
        dgp_param_column('w', node, d1, d2)
        for node in range(1, schema['max_num_nodes'] + 1)
        for d1 in range(1, max_dim + 1)
        for d2 in range(d1, max_dim + 1)
    ]


# Build the sampling-weight-matrix values, None beyond each node's actual matrix shape
def _weight_values(schema, w_list):
    if _is_compact_result_schema(schema):
        return {}
    max_dim = max(schema['max_p'], 1)
    values = {}
    for node in range(1, schema['max_num_nodes'] + 1):
        weight = np.asarray(w_list[node - 1]) if node <= len(w_list) else None
        for d1 in range(1, max_dim + 1):
            for d2 in range(d1, max_dim + 1):
                column = dgp_param_column('w', node, d1, d2)
                if weight is None:
                    value = None
                elif weight.ndim == 0:
                    value = float(weight) if d1 == d2 else 0.0
                elif weight.ndim == 1:
                    value = float(weight[d1 - 1]) if d1 == d2 else 0.0
                else:
                    value = float(weight[d1 - 1, d2 - 1])
                values[column] = value
    return values


# Build the PMC-engine-only result values, in the order defined by _PMC_ENGINE_COLUMNS
def _pmc_engine_values(
    config, h_config, converged, relative_improvement, estimator,
    result_dimension=None, compact_schema=False,
):
    is_mlp_pmc = config['algorithm'] == 'PMC' and config.get('method') == 'MLP'
    is_gp_pmc = config['algorithm'] == 'PMC' and config.get('method') == 'GP'
    uses_weighted_pmc = is_mlp_pmc or is_gp_pmc
    loss_weighting = (
        config.get('loss_weighting', 'uniform') if uses_weighted_pmc else None
    )
    uses_target_weights = loss_weighting in ('gaussian', 'alpha_mixture')
    gp_result = None
    if is_gp_pmc:
        gp_result = (
            estimator.last_result
            if result_dimension is None
            else estimator.last_results[result_dimension]
        )
    gp_configuration = estimator.estimator.config if is_gp_pmc else None
    gp_local_diagnostics = (
        gp_result.diagnostics
        if gp_result is not None
        else {}
    )
    gp_diagnostics = None
    if gp_result is not None:
        gp_diagnostics = dict(gp_result.to_dict()['diagnostics'])
        if compact_schema:
            gp_diagnostics.pop('standardized_query_discrepancy', None)
        gp_diagnostics = json.dumps(gp_diagnostics, allow_nan=False, sort_keys=True)
    pmc_engine_values = (
        config['algorithm'],
        config['sampling_method'],
        (
            None
            if config['algorithm'] == 'Oracle'
            else config.get('discrepancy_method', 'consecutive')
        ),
        config.get('n_transforms'),
        (
            None
            if config.get('hidden_features') is None
            else str(config['hidden_features'])
        ),
        config.get('n_training_steps'),
        config.get('optimizer'),
        config.get('weight_decay', 0.0) if is_mlp_pmc else None,
        config.get('base_learning_rate'),
        config.get('learning_rate_schedule'),
        config.get('batch_size'),
        h_config,
        config.get('degree'),
        config.get('input_degree', 1) if is_mlp_pmc else None,
        config.get('activation', 'relu') if is_mlp_pmc else None,
        config.get('discrepancy_centering', 'target') if is_mlp_pmc else None,
        config.get('ensemble_size', 1) if is_mlp_pmc else None,
        config.get('regression_loss', 'mse') if is_mlp_pmc else None,
        config.get('huber_delta', 1.0) if is_mlp_pmc else None,
        loss_weighting,
        config.get('target_bandwidth_quantile') if uses_target_weights else None,
        config.get('target_bandwidth_scale', 1.0) if uses_target_weights else None,
        config.get('alpha') if loss_weighting == 'alpha_mixture' else None,
        gp_configuration.kernel if is_gp_pmc else None,
        gp_configuration.inducing_count if is_gp_pmc else None,
        gp_configuration.maximum_updates if is_gp_pmc else None,
        gp_configuration.natural_gradient_learning_rate if is_gp_pmc else None,
        gp_configuration.minimum_epochs if is_gp_pmc else None,
        gp_configuration.patience if is_gp_pmc else None,
        gp_configuration.relative_tolerance if is_gp_pmc else None,
        gp_configuration.learn_inducing_locations if is_gp_pmc else None,
        gp_configuration.torch_num_threads if is_gp_pmc else None,
        gp_result.confidence_level if gp_result is not None else None,
        (
            estimator.local_neighborhood_exponent
            if is_gp_pmc
            else None
        ),
        estimator.minimum_local_count if is_gp_pmc else None,
        estimator.maximum_local_count if is_gp_pmc else None,
        gp_local_diagnostics.get('full_observation_count'),
        gp_local_diagnostics.get('requested_local_count'),
        gp_local_diagnostics.get('actual_local_count'),
        gp_local_diagnostics.get('local_fraction'),
        gp_local_diagnostics.get('neighborhood_radius'),
        (
            gp_result.posterior_standard_deviation
            if gp_result is not None
            else None
        ),
        gp_result.interval_lower if gp_result is not None else None,
        gp_result.interval_upper if gp_result is not None else None,
        gp_result.interval_method if gp_result is not None else None,
        gp_diagnostics,
        converged,
        relative_improvement,
        getattr(estimator, 'training_log_likelihood', None),
    )
    return dict(zip(_PMC_ENGINE_COLUMNS, pmc_engine_values))


# Build the fusion-only result values, always None on the Python side
def _fusion_only_values():
    return dict.fromkeys(_FUSION_ONLY_COLUMNS)


# Return the ordered result columns expected for a DGP schema.
def result_columns(schema):
    pmc_columns = list(_PMC_ENGINE_COLUMNS)
    if not schema.get('uses_cnf'):
        pmc_columns = [
            key for key in pmc_columns
            if key not in ('n_transforms', 'training_log_likelihood')
        ]
    return [
        'parametric_form', 'num_nodes', 'p', 'n', 'n_sample', 'method',
        'preprocessing', 'n_inference', 'seed', 'timestamp', 'iteration_id',
        'training_time_seconds', 'training_process_time_seconds',
        'training_cuda_time_seconds', 'simulation_time_seconds',
        'simulation_process_time_seconds', 'simulation_cuda_time_seconds',
        *_HARDWARE_CONTEXT_COLUMNS,
        *(['semantic_fingerprint'] if _is_ultra_compact_result_schema(schema) else []),
        *_dgp_param_columns(schema),
        *_weight_columns(schema),
        *pmc_columns,
        *_FUSION_ONLY_COLUMNS,
        'kl', 'w1', 'w2', 'mse', 'C_hat', 'true_mean',
    ]


# Expand a point-estimate dict's vector-valued C_hat/true_mean into one row per dimension
def _expand_per_dimension(eval_metrics):
    c_hat_vec = eval_metrics['C_hat']
    true_mean_vec = eval_metrics['true_mean']
    return [
        {**eval_metrics, 'C_hat': float(c_hat_i), 'true_mean': float(true_mean_i)}
        for c_hat_i, true_mean_i in zip(c_hat_vec, true_mean_vec)
    ]


# Assemble the unified results row (shared column schema with the R runner)
def _build_result_row(
    config, dgp_schema, dgp_parameters, method_name, h_config, n_sample,
    training_time_seconds, training_cpu_seconds, training_cuda_seconds,
    simulation_time_seconds, simulation_cpu_seconds, simulation_cuda_seconds,
    hardware_context, converged, relative_improvement, estimator, eval_metrics,
    result_dimension=None,
):
    parametric_form = config['parametric_form']
    p = 1 if parametric_form in ('univariate_Gaussian', 'Beta') else len(dgp_parameters['mu'][0])

    core = {
        'parametric_form': parametric_form,
        'num_nodes': config['num_nodes'],
        'p': p,
        'n': config['n'],
        'n_sample': n_sample,
        'method': method_name,
        'preprocessing': config['preprocessing'],
        'n_inference': config.get('n_inference'),
        'seed': config['seed'],
        'timestamp': config.get('timestamp'),
        'iteration_id': config.get('iteration_id'),
        'training_time_seconds': training_time_seconds,
        'training_process_time_seconds': training_cpu_seconds,
        'training_cuda_time_seconds': training_cuda_seconds,
        'simulation_time_seconds': simulation_time_seconds,
        'simulation_process_time_seconds': simulation_cpu_seconds,
        'simulation_cuda_time_seconds': simulation_cuda_seconds,
        **hardware_context,
    }

    pmc_engine_values = _pmc_engine_values(
        config, h_config, converged, relative_improvement, estimator,
        result_dimension=result_dimension,
        compact_schema=_is_compact_result_schema(dgp_schema),
    )
    # Drop the CNF-only columns entirely when no config in this batch uses CNF
    if not dgp_schema.get('uses_cnf'):
        for cnf_only_key in ('n_transforms', 'training_log_likelihood'):
            pmc_engine_values.pop(cnf_only_key, None)

    semantic_values = {}
    if _is_ultra_compact_result_schema(dgp_schema):
        semantic_values['semantic_fingerprint'] = config.get(
            'semantic_fingerprint'
        ) or _semantic_fingerprint(
            parametric_form,
            dgp_parameters,
            config.get('w', []),
        )

    return {
        **core,
        **semantic_values,
        **_dgp_param_values(dgp_schema, parametric_form, dgp_parameters),
        **_weight_values(dgp_schema, config.get('w', [])),
        **pmc_engine_values,
        **_fusion_only_values(),
        **eval_metrics,
    }


# Run one simulation end-to-end and return its unified results row
def simulate_config(config, plots_directory):

    # Seed numpy (32-bit range, forked workers need a fresh unseeded state) and torch
    seed = config.get('seed')
    if seed is not None:
        np.random.seed(int(seed) % (2**31))
        torch.manual_seed(seed)
    else:
        np.random.seed()

    # Fall back to a single-config schema for direct calls that bypass build_configs
    dgp_schema = config.get('dgp_schema') or _compute_dgp_schema([config])

    # Extract the parameters from the configuration
    parametric_form = config['parametric_form']
    dgp_parameters = config['DGP_parameters']
    n = config['n']
    sampling_method = config['sampling_method']
    discrepancy_method = config.get('discrepancy_method', 'consecutive')
    preprocessing = config['preprocessing']
    algorithm = config['algorithm']
    method_name = config['method']
    n_transforms = config.get('n_transforms')
    degree = config.get('degree', 1)
    loss_weighting = config.get('loss_weighting', 'uniform')
    target_bandwidth_quantile = config.get('target_bandwidth_quantile')
    target_bandwidth_scale = config.get('target_bandwidth_scale', 1.0)
    alpha = config.get('alpha')
    input_degree = config.get('input_degree', 1)
    activation = config.get('activation', 'relu')
    discrepancy_centering = config.get('discrepancy_centering', 'target')
    ensemble_size = config.get('ensemble_size', 1)
    weight_decay = config.get('weight_decay', 0.0)
    regression_loss = config.get('regression_loss', 'mse')
    huber_delta = config.get('huber_delta', 1.0)
    gp_kernel = config.get('gp_kernel')
    gp_inducing_count = config.get('gp_inducing_count', 8)
    gp_device = config.get('gp_device', 'cpu')
    gp_maximum_updates = config.get('gp_maximum_updates')
    gp_natural_gradient_learning_rate = config.get(
        'gp_natural_gradient_learning_rate', 0.03
    )
    gp_minimum_epochs = config.get('gp_minimum_epochs', 10)
    gp_patience = config.get('gp_patience', 3)
    gp_relative_tolerance = config.get('gp_relative_tolerance', 1e-4)
    gp_learn_inducing_locations = config.get('gp_learn_inducing_locations', True)
    gp_torch_num_threads = config.get('gp_torch_num_threads')
    gp_confidence_level = config.get('gp_confidence_level', 0.95)
    gp_local_neighborhood_exponent = config.get('gp_local_neighborhood_exponent')
    gp_minimum_local_count = config.get('gp_minimum_local_count', 256)
    gp_maximum_local_count = config.get('gp_maximum_local_count', 131072)
    gp_full_batch = config.get('gp_full_batch', False)
    gp_fixed_hyperparameters = config.get('gp_fixed_hyperparameters', False)
    gp_fixed_mean = config.get('gp_fixed_mean', 0.0)
    gp_fixed_lengthscale = config.get('gp_fixed_lengthscale', 1.0)
    gp_fixed_outputscale = config.get('gp_fixed_outputscale', 1.0)
    gp_fixed_noise = config.get('gp_fixed_noise', 1.0)
    gp_dtype = config.get('gp_dtype', 'float32')
    hidden_features = config.get('hidden_features')
    w = config['w']
    if _is_ultra_compact_result_schema(dgp_schema):
        config['semantic_fingerprint'] = _semantic_fingerprint(
            parametric_form,
            dgp_parameters,
            w,
        )
    n_inference = config.get('n_inference')
    optimizer = config.get('optimizer')
    base_learning_rate = (
        config.get('gp_base_learning_rate', 0.03)
        if method_name == 'GP' and 'gp_base_learning_rate' in config
        else config.get('base_learning_rate', 0.03)
    )
    learning_rate_schedule = config.get('learning_rate_schedule')
    if learning_rate_schedule is not None and learning_rate_schedule not in ('constant', 'cosine'):
        raise ValueError('learning_rate_schedule must be "constant" or "cosine".')
    use_cosine_schedule = learning_rate_schedule == 'cosine'
    n_training_steps = config.get('n_training_steps')
    batch_size = (
        config.get('gp_batch_size')
        if method_name == 'GP' and 'gp_batch_size' in config
        else config.get('batch_size')
    )
    h_values = config['h_values']
    selected_component = config.get('selected_component')
    if selected_component is not None and h_values != ['identity']:
        raise ValueError(
            "selected_component requires h_values to be ['identity']"
        )
    _validate_base_learning_rate(base_learning_rate, h_values, algorithm)
    _h_functions = {
        'identity': None,
        'normalized_sum': _normalized_sum,
        'square': lambda consensus: consensus ** 2,
    }

    # Set the device
    device = torch.device('cuda' if _uses_cuda(config) else 'cpu')
    hardware_context = _hardware_context_values(device)

    # Define the DGP based on the parametric form
    dgp, product_parameters = _build_dgp(parametric_form, dgp_parameters)

    # Generate the string representation of the parameters for the filename
    parameters_str = (
        _semantic_fingerprint(parametric_form, dgp_parameters, w)
        if _is_ultra_compact_result_schema(dgp_schema)
        else parameters_to_filename(dgp_parameters)
    )

    # Generate data from the DGP
    n_sample = np.random.poisson(lam=n)
    data = dgp.sample(n_sample)
    component_index = selected_component_index(
        selected_component, data.shape[2]
    )

    # Resolve adaptive batching from nominal and realized sizes for training and metadata
    batch_size = _resolve_batch_size(batch_size, n, n_sample)
    config['batch_size'] = batch_size

    # Sqrt-scale training steps with the realized sample size to balance under/overtraining
    if n_training_steps == 'adaptive':
        n_training_steps = int(np.clip(round(200 * np.sqrt(n_sample / 100)), 200, 8000))
        config['n_training_steps'] = n_training_steps

    if plots_directory is not None:
        os.makedirs(plots_directory, exist_ok=True)
        plot_product_distribution(
            parametric_form,
            parameters=product_parameters,
            show=False,
            save=True,
            filename=_plot_filename(
                'product_distribution', config, parameters_str, sampling_method, w
            ),
            plots_directory=plots_directory
        )

    # Compute Consensus and Discrepancy (not needed for the Oracle baseline)
    if algorithm != 'Oracle':
        consensus, discrepancy = consensus_discrepancy(
            data,
            w=w,
            sampling_method=sampling_method,
            discrepancy_method=discrepancy_method,
            selected_component=selected_component,
        )

        if plots_directory is not None:
            plot_discrepancy_distribution(
                discrepancy,
                show=False,
                save=True,
                filename=_plot_filename(
                    'discrepancy_distribution', config, parameters_str, sampling_method, w
                ),
                plots_directory=plots_directory
            )
    else:
        consensus, discrepancy = None, None

    discrepancy_condition = (
        torch.zeros(discrepancy.shape[1]).to(device) if discrepancy is not None else None
    )

    rows = []
    # Product sampling and Oracle fits do not depend on h; fit once and reuse them.
    # PMC regression methods (linear, MLP, and GP) remain in the h-dependent branch.
    if algorithm == 'Oracle' or (
        algorithm == 'PMC' and method_name in {'Gaussian', 'CNF'}
    ):
        estimator = _build_estimator(
            algorithm, parametric_form, method_name, n_transforms, hidden_features,
            n_training_steps, optimizer, base_learning_rate, batch_size, preprocessing,
            device, None, use_cosine_schedule=use_cosine_schedule
        )

        training_start_time = time.perf_counter()
        training_start_process_time = time.process_time()
        training_cuda_timer = _cuda_timer(device)
        _cuda_timer_start(training_cuda_timer)
        if algorithm == 'Oracle':
            estimator.fit(data)
        else:
            estimator.fit(consensus, discrepancy)
        training_cuda_seconds = _cuda_timer_stop(training_cuda_timer)
        training_cpu_seconds = time.process_time() - training_start_process_time
        training_time_seconds = time.perf_counter() - training_start_time

        if estimator.training_loss is not None:
            converged, relative_improvement = check_convergence(estimator.training_loss)
        else:
            converged, relative_improvement = None, None

        if plots_directory is not None and estimator.training_loss is not None:
            plot_training_loss(
                estimator.training_loss,
                show=False,
                save=True,
                filename=_plot_filename(
                    'training_loss', config, parameters_str, sampling_method, w
                ),
                plots_directory=plots_directory
            )

        # Sample consensus | discrepancy = 0 once; scored per h value below
        eval_metrics_shared, consensus_inference = _evaluate_pmc_samples(
            estimator, algorithm, dgp, discrepancy_condition, n_inference,
            plots_directory, config, parameters_str, sampling_method, w
        )

        for h_config in h_values:
            # Time only this h value's own point-estimate work, not the shared pre-loop setup
            row_start_time = time.perf_counter()
            row_start_process_time = time.process_time()
            simulation_cuda_timer = _cuda_timer(device)
            _cuda_timer_start(simulation_cuda_timer)
            h = _h_functions.get(h_config, h_config)
            point_estimate = _pmc_oracle_point_estimate(
                algorithm, estimator, consensus_inference, dgp, h_config, h,
                component_index=component_index,
            )
            eval_metrics = {**eval_metrics_shared, **point_estimate}

            simulation_cuda_seconds = _cuda_timer_stop(simulation_cuda_timer)
            simulation_cpu_seconds = time.process_time() - row_start_process_time
            simulation_time_seconds = time.perf_counter() - row_start_time
            # One row per consensus dimension, matching the R fusion runner's row expansion
            for dimension, dim_eval_metrics in enumerate(
                _expand_per_dimension(eval_metrics)
            ):
                rows.append(_build_result_row(
                    config, dgp_schema, dgp_parameters, method_name, h_config, n_sample,
                    training_time_seconds, training_cpu_seconds, training_cuda_seconds,
                    simulation_time_seconds, simulation_cpu_seconds,
                    simulation_cuda_seconds,
                    hardware_context, converged, relative_improvement, estimator,
                    dim_eval_metrics, result_dimension=dimension
                ))
    # PMC regression: the fit depends on h, so fit once per h value.
    else:
        for h_config in h_values:
            # Time only this h value's own fit-and-evaluate work, not the shared pre-loop setup
            row_start_time = time.perf_counter()
            row_start_process_time = time.process_time()
            h = _h_functions.get(h_config, h_config)
            learning_rate = (
                base_learning_rate[h_config]
                if isinstance(base_learning_rate, Mapping)
                else base_learning_rate
            )
            estimator = _build_estimator(
                algorithm, parametric_form, method_name, n_transforms, hidden_features,
                n_training_steps, optimizer, learning_rate, batch_size, preprocessing,
                device, h, use_cosine_schedule=use_cosine_schedule, degree=degree,
                loss_weighting=loss_weighting,
                target_bandwidth_quantile=target_bandwidth_quantile,
                target_bandwidth_scale=target_bandwidth_scale,
                alpha=alpha, input_degree=input_degree, activation=activation,
                discrepancy_centering=discrepancy_centering,
                ensemble_size=ensemble_size, weight_decay=weight_decay,
                regression_loss=regression_loss, huber_delta=huber_delta,
                h_config=h_config, seed=seed, gp_kernel=gp_kernel,
                gp_inducing_count=gp_inducing_count,
                gp_device=gp_device,
                gp_maximum_updates=gp_maximum_updates,
                gp_natural_gradient_learning_rate=gp_natural_gradient_learning_rate,
                gp_minimum_epochs=gp_minimum_epochs, gp_patience=gp_patience,
                gp_relative_tolerance=gp_relative_tolerance,
                gp_learn_inducing_locations=gp_learn_inducing_locations,
                gp_torch_num_threads=gp_torch_num_threads,
                gp_confidence_level=gp_confidence_level,
                gp_local_neighborhood_exponent=gp_local_neighborhood_exponent,
                gp_minimum_local_count=gp_minimum_local_count,
                gp_maximum_local_count=gp_maximum_local_count,
                gp_full_batch=gp_full_batch,
                gp_fixed_hyperparameters=gp_fixed_hyperparameters,
                gp_fixed_mean=gp_fixed_mean,
                gp_fixed_lengthscale=gp_fixed_lengthscale,
                gp_fixed_outputscale=gp_fixed_outputscale,
                gp_fixed_noise=gp_fixed_noise,
                gp_dtype=gp_dtype,
            )

            training_start_time = time.perf_counter()
            training_start_process_time = time.process_time()
            training_cuda_timer = _cuda_timer(device)
            simulation_cuda_timer = _cuda_timer(device)
            _cuda_timer_start(simulation_cuda_timer)
            _cuda_timer_start(training_cuda_timer)
            estimator.fit(consensus, discrepancy)
            training_cuda_seconds = _cuda_timer_stop(training_cuda_timer)
            training_cpu_seconds = time.process_time() - training_start_process_time
            training_time_seconds = time.perf_counter() - training_start_time

            if estimator.training_loss is not None:
                converged, relative_improvement = check_convergence(estimator.training_loss)
            else:
                converged, relative_improvement = None, None

            if plots_directory is not None and estimator.training_loss is not None:
                plot_training_loss(
                    estimator.training_loss,
                    show=False,
                    save=True,
                    filename=_plot_filename(
                        'training_loss', config, parameters_str, sampling_method, w
                    ),
                    plots_directory=plots_directory
                )

            eval_metrics = _evaluate_pmc(
                estimator, dgp, discrepancy_condition, h, h_config=h_config,
                component_index=component_index,
            )

            simulation_cuda_seconds = _cuda_timer_stop(simulation_cuda_timer)
            simulation_cpu_seconds = time.process_time() - row_start_process_time
            simulation_time_seconds = time.perf_counter() - row_start_time
            # One row per consensus dimension, matching the R fusion runner's row expansion
            for dimension, dim_eval_metrics in enumerate(
                _expand_per_dimension(eval_metrics)
            ):
                result_config = {
                    **config,
                    'base_learning_rate': learning_rate,
                }
                rows.append(_build_result_row(
                    result_config, dgp_schema, dgp_parameters, method_name,
                    h_config, n_sample,
                    training_time_seconds, training_cpu_seconds, training_cuda_seconds,
                    simulation_time_seconds, simulation_cpu_seconds,
                    simulation_cuda_seconds,
                    hardware_context, converged, relative_improvement, estimator,
                    dim_eval_metrics, result_dimension=dimension
                ))

    return rows
