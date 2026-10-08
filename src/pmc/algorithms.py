##############################
# Import necessary libraries #
##############################

# Standard libraries
import warnings
from functools import partial
from numbers import Integral, Real

# Third-party libraries
import numpy as np
import torch
import zuko
from scipy.stats import beta as beta_dist
from scipy.stats import multivariate_normal

# PMC library
from .utils import (
    MLP,
    ensure_batched_tensor,
    feature_dim,
    fit_linear_regression,
    polynomial_features,
    regression_huber_loss,
    regression_mse_loss,
    to_numpy,
    to_tensor,
    train_torch_model,
    validate_activation,
    weighted_regression_huber_loss,
    weighted_regression_mse_loss,
)

#######################
# Abstract interfaces #
#######################

class Sampling:
    # Draws samples from the fitted distribution; must be implemented by subclasses
    def sample(self, *args, **kwargs):
        raise NotImplementedError


class Regression:
    # Fits the regression estimator; must be implemented by subclasses
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    # Predicts outputs from the fitted regression estimator; must be implemented by subclasses
    def predict(self, *args, **kwargs):
        raise NotImplementedError


##################################
# Consensus and Discrepancy (CD) #
##################################

_WEIGHT_ROUNDOFF_FACTOR = 64

# Bound float64 matrix roundoff using dimension, scale, and accumulation count
def _weight_roundoff_tolerance(matrix, term_count=1):
    dimension = matrix.shape[0] if matrix.ndim > 0 else 1
    scale = max(1.0, float(np.max(np.abs(matrix))))
    coefficient = (
        _WEIGHT_ROUNDOFF_FACTOR
        * np.finfo(np.float64).eps
        * dimension
        * term_count
    )
    return coefficient * scale

# Validate consensus weights without modifying their supplied values
def _validate_consensus_weights(w, num_nodes, p):
    if not isinstance(w, (list, tuple, np.ndarray)) or (
        isinstance(w, np.ndarray) and w.ndim == 0
    ):
        raise ValueError('w must be a list, tuple, or array of weights')
    if len(w) != num_nodes:
        raise ValueError(
            f'Number of weights {len(w)} does not match '
            f'number of nodes num_nodes={num_nodes}'
        )

    weights = []
    weight_shapes = []
    for i, weight in enumerate(w):
        weight = np.asarray(weight)
        if weight.shape not in ((), (p,), (p, p)):
            raise ValueError(
                f'Weight w[{i}] has shape {weight.shape}, expected (), '
                f'({p},), or ({p}, {p})'
            )
        if (
            np.issubdtype(weight.dtype, np.bool_)
            or not (
                np.issubdtype(weight.dtype, np.integer)
                or np.issubdtype(weight.dtype, np.floating)
            )
        ):
            raise ValueError(f'Weight w[{i}] must contain real numeric values')
        weight = weight.astype(np.float64, copy=False)
        if not np.all(np.isfinite(weight)):
            raise ValueError(f'Weight w[{i}] must contain only finite values')

        if weight.ndim == 2:
            symmetry_tolerance = _weight_roundoff_tolerance(weight)
            if not np.allclose(
                weight,
                weight.T,
                rtol=0.0,
                atol=symmetry_tolerance,
            ):
                raise ValueError(
                    f'Weight matrix w[{i}] must be symmetric within its float64 '
                    f'roundoff tolerance {symmetry_tolerance}'
                )
            symmetric_weight = weight / 2 + weight.T / 2
            psd_tolerance = _weight_roundoff_tolerance(symmetric_weight)
            if np.linalg.eigvalsh(symmetric_weight)[0] < -psd_tolerance:
                raise ValueError(
                    f'Weight matrix w[{i}] must be positive semi-definite within '
                    f'its float64 roundoff tolerance {psd_tolerance}'
                )
        else:
            diagonal = np.atleast_1d(weight)
            psd_tolerance = _weight_roundoff_tolerance(weight)
            if np.any(diagonal < -psd_tolerance):
                raise ValueError(
                    f'Weight w[{i}] must be nonnegative within its float64 '
                    f'roundoff tolerance {psd_tolerance}'
                )
        weights.append(weight)
        weight_shapes.append(weight.shape)

    if len(set(weight_shapes)) != 1:
        raise ValueError('All consensus weights must use the same representation')

    with np.errstate(over='ignore', invalid='ignore'):
        weight_sum = np.sum(weights, axis=0)
    if not np.all(np.isfinite(weight_sum)):
        raise ValueError('The sum of the weights must be finite')
    expected_sum = np.eye(p) if weights[0].ndim == 2 else np.ones_like(weight_sum)
    sum_tolerance = _weight_roundoff_tolerance(
        weight_sum,
        term_count=num_nodes,
    )
    if not np.allclose(
        weight_sum,
        expected_sum,
        rtol=0.0,
        atol=sum_tolerance,
    ):
        raise ValueError(
            'The sum of the weights must equal the identity action '
            f'within its accumulated float64 roundoff tolerance '
            f'{sum_tolerance}. Sum is:\n{weight_sum}'
        )
    return weights

# Apply one scalar, diagonal, or dense consensus weight
def _apply_consensus_weight(samples, weight):
    if weight.ndim == 0:
        return samples * weight
    if weight.ndim == 1:
        return samples * weight[np.newaxis, :]
    return samples @ weight.T

# Convert and validate an optional one-based component selector
def selected_component_index(selected_component, p):
    if selected_component is None:
        return None
    if isinstance(selected_component, bool) or not isinstance(
        selected_component, Integral
    ):
        raise ValueError('selected_component must be an integer')
    if selected_component < 1 or selected_component > p:
        raise ValueError(
            f'selected_component must be between 1 and ambient p={p}'
        )
    return int(selected_component) - 1

# Project one consensus weight onto a selected raw component
def _project_consensus_weight(weight, component_index):
    if weight.ndim == 0:
        return weight
    if weight.ndim == 1:
        return weight[[component_index]]
    return weight[np.ix_([component_index], [component_index])]

# Build the deterministic orthonormal Helmert contrast
def _helmert_contrast(num_nodes, dtype):
    contrast = np.zeros((num_nodes - 1, num_nodes), dtype=dtype)
    for i in range(num_nodes - 1):
        scale = np.sqrt((i + 1) * (i + 2))
        contrast[i, :i + 1] = 1 / scale
        contrast[i, i + 1] = -(i + 1) / scale
    return contrast

# Build a balanced orthonormal Haar contrast for arbitrary node counts
def _haar_contrast(num_nodes, dtype):
    rows = []

    # Split each interval recursively with normalized unequal child blocks
    def append_interval(start, stop):
        size = stop - start
        if size == 1:
            return
        middle = start + (size + 1) // 2
        left_size = middle - start
        right_size = stop - middle
        row = np.zeros(num_nodes, dtype=dtype)
        row[start:middle] = np.sqrt(right_size / (left_size * size))
        row[middle:stop] = -np.sqrt(left_size / (right_size * size))
        rows.append(row)
        append_interval(start, middle)
        append_interval(middle, stop)

    append_interval(0, num_nodes)
    return np.stack(rows)

# Apply a linear node contrast and flatten basis rows before feature columns
def _apply_linear_contrast(data, contrast, n):
    return np.einsum('kn,snp->skp', contrast, data).reshape(n, -1)

# Compute weighted consensus and the configured discrepancy across IID node samples
def _iid_consensus_discrepancy(
    data, w, num_nodes, n, p, discrepancy_method='consecutive'
):
    # Linear interpolation: C = \sum^{K}_{i=1} w_i X_i
    consensus = np.zeros((n, p))
    for i in range(num_nodes):
        consensus += _apply_consensus_weight(data[:, i, :], w[i]).reshape(n, p)

    if discrepancy_method == 'consecutive':
        # Pairwise differences: D_i = X_i - X_{i+1} for i=1,...,K-1
        discrepancy_list = []
        for i in range(num_nodes - 1):
            discrepancy_list.append(
                (data[:, i, :] - data[:, i + 1, :]).reshape(n, p)
            )
        discrepancy = np.concatenate(discrepancy_list, axis=1)
    elif discrepancy_method == 'star':
        discrepancy = np.concatenate(
            [
                (data[:, 0, :] - data[:, i, :]).reshape(n, p)
                for i in range(1, num_nodes)
            ],
            axis=1,
        )
    elif discrepancy_method == 'helmert':
        contrast_dtype = np.result_type(data.dtype, np.float32)
        contrast = _helmert_contrast(num_nodes, contrast_dtype)
        discrepancy = _apply_linear_contrast(data, contrast, n)
    elif discrepancy_method == 'haar':
        contrast_dtype = np.result_type(data.dtype, np.float32)
        contrast = _haar_contrast(num_nodes, contrast_dtype)
        discrepancy = _apply_linear_contrast(data, contrast, n)
    elif discrepancy_method == 'spanning_tree':
        # Use child order with zero-based binary-heap parents
        discrepancy = np.concatenate(
            [
                (
                    data[:, (child - 1) // 2, :]
                    - data[:, child, :]
                ).reshape(n, p)
                for child in range(1, num_nodes)
            ],
            axis=1,
        )
    elif discrepancy_method == 'variance':
        centered = data - np.mean(data, axis=1, keepdims=True)
        discrepancy = np.mean(np.sum(centered ** 2, axis=2), axis=1, keepdims=True)
    elif discrepancy_method == 'max_distance':
        pairwise_distances = [
            np.linalg.norm(data[:, i, :] - data[:, j, :], axis=1)
            for i in range(num_nodes)
            for j in range(i + 1, num_nodes)
        ]
        discrepancy = np.max(np.stack(pairwise_distances, axis=1), axis=1, keepdims=True)
    else:
        raise ValueError(f'Unsupported discrepancy_method: {discrepancy_method}')

    return consensus, discrepancy


# Computes consensus (C in the paper) and discrepancy (D in the paper) from per-node samples
def consensus_discrepancy(
    data, w=None, sampling_method='iid', discrepancy_method='consecutive',
    selected_component=None,
):
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError('data must be a three-dimensional array')
    if not (
        np.issubdtype(data.dtype, np.integer)
        or np.issubdtype(data.dtype, np.floating)
    ):
        raise ValueError('data must contain real numeric values')
    if not np.all(np.isfinite(data)):
        raise ValueError('data must contain only finite values')

    # Derive the number of nodes (K in the paper) from the data shape
    n, num_nodes, p = data.shape
    if n < 1:
        raise ValueError('data must contain at least one sample')
    if num_nodes < 2:
        raise ValueError('data must contain at least two nodes')
    if p < 1:
        raise ValueError('data must contain at least one feature')
    if sampling_method == 'cartesian' and num_nodes != 2:
        raise ValueError("cartesian sampling_method is only supported for num_nodes=2")

    component_index = selected_component_index(selected_component, p)

    # If weights are not provided, use default weights
    if w is None:
        # Default: Each node gets a (1/num_nodes) * I weight
        w = [np.eye(p) / num_nodes for _ in range(num_nodes)]
    # If weights are provided, validate them
    else:
        w = _validate_consensus_weights(w, num_nodes, p)

    if component_index is not None:
        data = data[:, :, component_index:component_index + 1]
        w = [
            _project_consensus_weight(weight, component_index)
            for weight in w
        ]
        p = 1

    if sampling_method == 'iid':
        consensus, discrepancy = _iid_consensus_discrepancy(
            data, w, num_nodes, n, p, discrepancy_method=discrepancy_method
        )
    elif sampling_method == 'cartesian':
        if discrepancy_method != 'consecutive':
            raise ValueError(
                f'discrepancy_method={discrepancy_method!r} is only supported '
                'with sampling_method=\'iid\''
            )
        warnings.warn(
            "REMINDER: I DIDN'T DOUBLE CHECK THE CODE FOR THE NON-IID CASE. "
            "THE WAY I COMPUTE C AND D MIGHT NOT BE CORRECT."
        )
        # Get all i, j combinations
        consensus_matrix = (
            _apply_consensus_weight(data[:, 0, :], w[0])[:, None, :]
            + _apply_consensus_weight(data[:, 1, :], w[1])[None, :, :]
        )
        discrepancy_matrix = (
            data[:, 0, :][:, None, :] - data[:, 1, :][None, :, :]
        )

        # Flatten the grid to get n^2 samples
        consensus = consensus_matrix.reshape(-1, p)
        discrepancy = discrepancy_matrix.reshape(-1, p)
    elif sampling_method == 'greedy':
        if discrepancy_method != 'consecutive':
            raise ValueError(
                f'discrepancy_method={discrepancy_method!r} is only supported '
                'with sampling_method=\'iid\''
            )
        if num_nodes != 2:
            raise ValueError("greedy sampling_method is only supported for num_nodes=2")

        # Extract the two nodes' samples from the data
        samples_a = data[:, 0, :]
        samples_b = data[:, 1, :]

        if p == 1:
            # In 1D, sorting both arrays gives the exact Wasserstein-1 optimal coupling
            order_a = np.argsort(samples_a[:, 0])
            order_b = np.argsort(samples_b[:, 0])
            paired_a = samples_a[order_a]
            paired_b = samples_b[order_b]
        else:
            from scipy.spatial.distance import cdist
            # cdist avoids the intermediate n x n x p diff array, halving peak memory
            dist = cdist(samples_a, samples_b).astype(float)
            if not np.all(np.isfinite(dist)):
                raise ValueError('greedy pairwise distances must be finite')

            paired_a = np.zeros((n, p))
            paired_b = np.zeros((n, p))

            for k in range(n):
                i, j = np.unravel_index(np.argmin(dist), dist.shape)
                paired_a[k] = samples_a[i]
                paired_b[k] = samples_b[j]
                dist[i, :] = np.inf
                dist[:, j] = np.inf

        # Compute consensus and discrepancy from greedy pairs, same formula as iid
        paired_data = np.stack([paired_a, paired_b], axis=1)
        consensus, discrepancy = _iid_consensus_discrepancy(
            paired_data, w, num_nodes, n, p
        )
    else:
        raise ValueError(f"Unsupported sampling_method: {sampling_method}")

    return consensus, discrepancy


##############################
# Shared preprocessing mixin #
##############################

# Standardization/hyperparameter logic shared by sampling and PMC via multiple inheritance
class _PreprocessingMixin:
    # Initializes shared preprocessing/training hyperparameters
    def __init__(self, preprocessing=None, hidden_features=(64, 64), n_training_steps=500,
                 optimizer='Adam', learning_rate=1e-4, batch_size=None, device='cpu',
                 weight_decay=0.0):
        self.preprocessing = preprocessing
        self.hidden_features = hidden_features
        self.n_training_steps = n_training_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.consensus_transformed_mean = None
        self.consensus_transformed_std = None
        self.training_loss = None
        self.training_log_likelihood = None
        self.device = device

    # Applies the configured preprocessing transform (currently: optional logit) to consensus
    def _forward_transform(self, consensus):
        if self.preprocessing == 'logit':
            # Avoid extreme values of 0 and 1 for logit transformation by clipping consensus
            consensus_clipped = np.clip(consensus, 1e-6, 1 - 1e-6)
            return np.log(consensus_clipped / (1 - consensus_clipped))
        return consensus

    # Reverses the configured preprocessing transform, mapping back to the original consensus scale
    def _inverse_transform(self, consensus_transformed):
        if self.preprocessing == 'logit':
            return 1 / (1 + np.exp(-consensus_transformed))
        return consensus_transformed

    # Standardizes consensus and converts consensus, discrepancy to torch tensors
    def _preprocess(self, consensus, discrepancy):
        consensus_transformed = self._forward_transform(consensus)

        # Standardize consensus_transformed
        self.consensus_transformed_mean = np.mean(consensus_transformed, axis=0)
        self.consensus_transformed_std = np.std(consensus_transformed, axis=0)
        consensus_standardized = (
            (consensus_transformed - self.consensus_transformed_mean)
            / self.consensus_transformed_std
        )

        # Convert consensus and discrepancy to torch tensors
        consensus_tensor = torch.from_numpy(consensus_standardized).float().to(self.device)
        discrepancy_tensor = torch.from_numpy(discrepancy).float().to(self.device)

        return consensus_tensor, discrepancy_tensor


####################################
# Product Monte Carlo sampling       #
####################################

class ProductSampling(Sampling, _PreprocessingMixin):
    # Preprocesses consensus, discrepancy and fits the estimator, adjusting for standardization
    def fit(self, consensus, discrepancy):
        consensus, discrepancy = self._preprocess(consensus, discrepancy)
        self._fit_estimator(consensus, discrepancy)
        if self.training_log_likelihood is not None:
            self.training_log_likelihood -= float(
                np.sum(np.log(self.consensus_transformed_std))
            )

    # Fits the underlying estimator on preprocessed data; must be implemented by subclasses
    def _fit_estimator(self, consensus, discrepancy):
        raise NotImplementedError

    # Applies the already-fitted preprocessing transform to a query consensus, for use in log_prob()
    def _standardize_query(self, consensus):
        consensus_np = to_numpy(consensus)
        consensus_transformed = self._forward_transform(consensus_np)
        return (
            (consensus_transformed - self.consensus_transformed_mean)
            / self.consensus_transformed_std
        )

    # Unstandardizes inference outputs and reverses the preprocessing transform, if any
    def _postprocess(
        self, consensus_inference_standardized, log_p_consensus_inference_standardized
    ):
        # Unstandardize
        consensus_inference_transformed = (
            consensus_inference_standardized * self.consensus_transformed_std
            + self.consensus_transformed_mean
        )
        log_p_consensus_inference_transformed = (
            log_p_consensus_inference_standardized
            - np.sum(np.log(self.consensus_transformed_std))
        )

        # Reverse the preprocessing, if it was applied during preprocessing
        consensus_inference = self._inverse_transform(consensus_inference_transformed)
        if self.preprocessing == 'logit':
            log_p_consensus_inference = log_p_consensus_inference_transformed - np.sum(
                np.log(consensus_inference + 1e-6)
                + np.log(1 - consensus_inference + 1e-6), axis=-1
            )
        else:
            log_p_consensus_inference = log_p_consensus_inference_transformed

        return consensus_inference, log_p_consensus_inference

    # Draws n_inference samples of consensus given discrepancy_condition
    def sample(self, discrepancy_condition=None, n_inference=100000):
        if discrepancy_condition is None:
            raise ValueError(
                "discrepancy_condition must be provided for conditional sampling."
            )

        consensus_inference_standardized, log_p_consensus_inference_standardized = (
            self._sample_standardized(discrepancy_condition, n_inference)
        )
        return self._postprocess(
            consensus_inference_standardized, log_p_consensus_inference_standardized
        )

    # Draws standardized consensus samples given discrepancy_condition; implemented by subclasses
    def _sample_standardized(self, discrepancy_condition, n_inference):
        raise NotImplementedError


class GaussianPMC(ProductSampling):
    # Initializes the fitted Gaussian-conditional parameters (kwargs: see _PreprocessingMixin)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear_weights = None
        self.linear_bias = None
        self.cov_conditional = None

    # Returns the Gaussian of consensus given d; the conditional covariance is constant
    def _condition_gaussian(self, d):
        mu_consensus_cond = d @ self.linear_weights + self.linear_bias
        return mu_consensus_cond, self.cov_conditional

    # Fits the linear-Gaussian conditional of consensus on discrepancy and logs the likelihood
    def _fit_estimator(self, consensus, discrepancy):
        self.linear_weights, self.linear_bias, self.cov_conditional = fit_linear_regression(
            consensus, discrepancy
        )

        consensus_np = to_numpy(consensus)
        discrepancy_np = to_numpy(discrepancy)
        consensus_dim = consensus_np.shape[1]
        mu_consensus_cond_all = discrepancy_np @ self.linear_weights + self.linear_bias
        diff = consensus_np - mu_consensus_cond_all
        cov_consensus_cond_inv = np.linalg.inv(self.cov_conditional)
        _, log_det = np.linalg.slogdet(self.cov_conditional)
        log_liks = -0.5 * (consensus_dim * np.log(2 * np.pi) + log_det +
                            np.einsum('ni,ij,nj->n', diff, cov_consensus_cond_inv, diff))
        self.training_log_likelihood = float(np.mean(log_liks))

    # Computes the log-density of consensus under the Gaussian conditional on discrepancy_condition
    def log_prob(self, consensus, discrepancy_condition):
        d = to_numpy(discrepancy_condition).flatten()
        mu_consensus_cond, cov_consensus_cond = self._condition_gaussian(d)

        consensus_standardized = self._standardize_query(consensus)
        log_p_standardized = multivariate_normal.logpdf(
            consensus_standardized, mean=mu_consensus_cond, cov=cov_consensus_cond
        )
        _, log_p = self._postprocess(consensus_standardized, log_p_standardized)
        return log_p

    # Draws standardized consensus samples from the Gaussian conditional on discrepancy_condition
    def _sample_standardized(self, discrepancy_condition, n_inference):
        d = to_numpy(discrepancy_condition).flatten()
        mu_consensus_cond, cov_consensus_cond = self._condition_gaussian(d)

        consensus_inference_standardized = multivariate_normal.rvs(
            mean=mu_consensus_cond, cov=cov_consensus_cond, size=n_inference
        )
        log_p_consensus_inference_standardized = multivariate_normal.logpdf(
            consensus_inference_standardized, mean=mu_consensus_cond, cov=cov_consensus_cond
        )

        return consensus_inference_standardized, log_p_consensus_inference_standardized


class CNFPMC(ProductSampling):
    # Initializes the flow architecture and cosine-schedule params (kwargs: see _PreprocessingMixin)
    def __init__(self, n_transforms=5, use_cosine_schedule=False, **kwargs):
        super().__init__(**kwargs)
        self.n_transforms = n_transforms
        self.use_cosine_schedule = use_cosine_schedule
        self.flow_estimator = None

    # Trains the conditional normalizing-flow density estimator of consensus given discrepancy
    def _fit_estimator(self, consensus, discrepancy):
        n_flow_layers = self.n_transforms
        features = feature_dim(consensus)
        context = feature_dim(discrepancy)

        flow_estimator = zuko.flows.NSF(
            features=features,
            context=context,
            hidden_features=self.hidden_features,
            transforms=n_flow_layers
        ).to(self.device)

        training_loss = train_torch_model(
            flow_estimator, consensus, discrepancy,
            loss_fn=lambda m, c, d: -m(d).log_prob(c).mean(),
            n_training_steps=self.n_training_steps, optimizer_name=self.optimizer,
            learning_rate=self.learning_rate, batch_size=self.batch_size,
            use_cosine_schedule=self.use_cosine_schedule
        )

        self.flow_estimator = flow_estimator
        self.training_loss = training_loss
        with torch.no_grad():
            self.training_log_likelihood = float(
                flow_estimator(discrepancy).log_prob(consensus).mean().item()
            )

    # Computes the log-density of consensus under the fitted flow conditional on discrepancy
    def log_prob(self, consensus, discrepancy_condition):
        consensus_standardized = self._standardize_query(consensus)
        with torch.no_grad():
            consensus_tensor = to_tensor(consensus_standardized)
            discrepancy_condition = ensure_batched_tensor(discrepancy_condition)
            log_p_standardized = self.flow_estimator(
                discrepancy_condition
            ).log_prob(consensus_tensor).numpy()
        _, log_p = self._postprocess(consensus_standardized, log_p_standardized)
        return log_p

    # Draws standardized consensus samples from the fitted flow conditional on discrepancy_condition
    def _sample_standardized(self, discrepancy_condition, n_inference):
        with torch.no_grad():
            discrepancy_condition = ensure_batched_tensor(discrepancy_condition)
            if discrepancy_condition.ndim != 2 or discrepancy_condition.shape[0] != 1:
                raise ValueError(
                    "discrepancy_condition must have shape (context_dim,) or "
                    f"(1, context_dim), got {tuple(discrepancy_condition.shape)}"
                )

            consensus_samples = self.flow_estimator(discrepancy_condition).sample(
                (n_inference,)
            )
            log_p_consensus_samples = self.flow_estimator(
                discrepancy_condition
            ).log_prob(consensus_samples)

            if consensus_samples.ndim == 3 and consensus_samples.shape[1] == 1:
                consensus_samples = consensus_samples[:, 0, :]
            if log_p_consensus_samples.ndim == 2 and log_p_consensus_samples.shape[1] == 1:
                log_p_consensus_samples = log_p_consensus_samples[:, 0]

        return to_numpy(consensus_samples), to_numpy(log_p_consensus_samples)


###################
# Oracle baseline #
###################

# Minimal Sampling leaf that exploits true knowledge of the per-node distributions
class OracleSampling(Sampling):
    pass


# Compute a symmetric PSD Moore-Penrose pseudoinverse
def _symmetric_psd_pinv(matrix):
    symmetric = (np.asarray(matrix) + np.asarray(matrix).T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = np.max(np.abs(eigenvalues))
    tolerance = max(symmetric.shape) * np.finfo(eigenvalues.dtype).eps * scale
    retained = eigenvalues > tolerance
    if not np.any(retained):
        raise ValueError(
            "Matrix has no positive mode above the pseudoinverse tolerance."
        )
    inverse = (
        eigenvectors[:, retained]
        @ np.diag(1 / eigenvalues[retained])
        @ eigenvectors[:, retained].T
    )
    return (inverse + inverse.T) / 2


class GaussianOracleSampling(OracleSampling):
    # Initialize the fitted product-Gaussian parameters
    def __init__(self):
        self.mu_product = None
        self.cov_product = None
        self.training_loss = None
        self.training_log_likelihood = None

    # Fit the product-Gaussian analytically from per-node component data
    def fit(self, data):
        num_nodes = data.shape[1]
        p = data.shape[2]
        precision = np.zeros((p, p))
        precision_mean = np.zeros(p)
        for k in range(num_nodes):
            data_k = data[:, k, :]
            mu_k = np.mean(data_k, axis=0)
            cov_k = np.cov(data_k, rowvar=False).reshape(p, p)
            cov_inv = _symmetric_psd_pinv(cov_k)
            precision += cov_inv
            precision_mean += cov_inv @ mu_k
        self.cov_product = _symmetric_psd_pinv(precision)
        self.cov_product = (self.cov_product + self.cov_product.T) / 2
        self.mu_product = self.cov_product @ precision_mean

    # Draw samples from the fitted product-Gaussian distribution
    def sample(self, n_inference=100000):
        distribution = multivariate_normal(
            mean=self.mu_product,
            cov=self.cov_product,
            allow_singular=True,
        )
        samples = np.asarray(distribution.rvs(size=n_inference)).reshape(
            -1, self.mu_product.size
        )
        log_p = np.atleast_1d(distribution.logpdf(samples))
        return samples, log_p

    # Compute the log-density of the fitted product-Gaussian at given points
    def log_prob(self, consensus):
        consensus_np = to_numpy(consensus)
        return np.atleast_1d(
            multivariate_normal.logpdf(
                consensus_np,
                mean=self.mu_product,
                cov=self.cov_product,
                allow_singular=True,
            )
        )


class BetaOracleSampling(OracleSampling):
    # Initialize the fitted product-Beta shape parameters
    def __init__(self):
        self.a_product = None
        self.b_product = None
        self.training_loss = None
        self.training_log_likelihood = None

    # Fit the product-Beta analytically from per-node component data
    def fit(self, data):
        num_nodes = data.shape[1]
        a_sum = 0.0
        b_sum = 0.0
        for k in range(num_nodes):
            a_k, b_k, _, _ = beta_dist.fit(data[:, k, 0], floc=0, fscale=1)
            a_sum += a_k
            b_sum += b_k
        a_product = a_sum - (num_nodes - 1)
        b_product = b_sum - (num_nodes - 1)
        if a_product <= 0 or b_product <= 0:
            raise ValueError(
                "Resulting shape parameters for the product Beta "
                "distribution must be strictly positive."
            )
        self.a_product = a_product
        self.b_product = b_product

    # Draw samples from the fitted product-Beta distribution
    def sample(self, n_inference=100000):
        samples = beta_dist.rvs(self.a_product, self.b_product, size=n_inference)
        samples = samples.reshape(-1, 1)
        log_p = beta_dist.logpdf(samples, self.a_product, self.b_product).flatten()
        return samples, log_p

    # Compute the log-density of the fitted product-Beta at given points
    def log_prob(self, consensus):
        consensus_np = to_numpy(consensus)
        return beta_dist.logpdf(consensus_np, self.a_product, self.b_product).flatten()


# Compute the analytic Oracle estimate of E[h(consensus)] from the fitted product-Gaussian params
def oracle_analytic_point_estimate(h_config, mu_product, cov_product):
    if h_config == 'identity':
        return mu_product
    if h_config == 'square':
        return np.diag(cov_product) + mu_product ** 2
    if h_config == 'normalized_sum':
        return np.array([np.sum(mu_product) / np.sqrt(np.size(mu_product))])
    raise ValueError(f"No analytic Oracle point estimate available for h: {h_config}")


# Compute the analytic Oracle estimate of E[h(consensus)] from the fitted product-Beta params
def oracle_analytic_point_estimate_beta(h_config, a_product, b_product):
    if h_config == 'identity':
        return a_product / (a_product + b_product)
    if h_config == 'square':
        return (
            a_product * (a_product + 1)
            / ((a_product + b_product) * (a_product + b_product + 1))
        )
    raise ValueError(f"No analytic Oracle point estimate available for h: {h_config}")


######################################
# Product Monte Carlo (PMC)         #
######################################

class PMC(Regression, _PreprocessingMixin):
    # Adds the h transform (kwargs: see _PreprocessingMixin)
    def __init__(self, h=None, **kwargs):
        super().__init__(**kwargs)
        self.h = h

    # Applies h to consensus if set, then preprocesses consensus, discrepancy and fits the estimator
    def fit(self, consensus, discrepancy):
        if self.h is not None:
            consensus = self.h(consensus)
        consensus, discrepancy = self._preprocess(consensus, discrepancy)
        self._fit_estimator(consensus, discrepancy)

    # Fits the underlying estimator on preprocessed data; must be implemented by subclasses
    def _fit_estimator(self, consensus, discrepancy):
        raise NotImplementedError

    # Predicts consensus from discrepancy_condition, reversing preprocessing
    def predict(self, discrepancy_condition):
        discrepancy_condition = ensure_batched_tensor(discrepancy_condition)

        consensus_standardized = self._predict_standardized(discrepancy_condition)

        # Unstandardize
        consensus_transformed = (
            consensus_standardized * self.consensus_transformed_std
            + self.consensus_transformed_mean
        )

        # Reverse the preprocessing, if it was applied
        consensus = self._inverse_transform(consensus_transformed)

        return consensus

    # Predicts standardized consensus from discrepancy_condition; implemented by subclasses
    def _predict_standardized(self, discrepancy_condition):
        raise NotImplementedError


class LinearPMC(PMC):
    # Initializes the fitted polynomial regression parameters (kwargs: see _PreprocessingMixin)
    def __init__(self, degree=1, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.linear_weights = None
        self.linear_bias = None

    # Fits an ordinary least squares regression of consensus on a discrepancy polynomial
    def _fit_estimator(self, consensus, discrepancy):
        self.linear_weights, self.linear_bias, _ = fit_linear_regression(
            consensus, discrepancy, degree=self.degree
        )

    # Predicts standardized consensus from discrepancy_condition under the fitted polynomial model
    def _predict_standardized(self, discrepancy_condition):
        discrepancy_np = to_numpy(discrepancy_condition)
        discrepancy_poly = polynomial_features(discrepancy_np, self.degree)
        return discrepancy_poly @ self.linear_weights + self.linear_bias


class MLPPMC(PMC):
    # Initializes the MLP architecture, discrepancy statistics, schedule, and loss weighting
    def __init__(
        self, use_cosine_schedule=False, loss_weighting='uniform',
        target_bandwidth_quantile=None, target_bandwidth_scale=1.0,
        alpha=None, input_degree=1, activation='relu',
        discrepancy_centering='target', ensemble_size=1,
        regression_loss='mse', huber_delta=1.0, **kwargs
    ):
        super().__init__(**kwargs)
        validate_activation(activation)
        if discrepancy_centering not in ('mean', 'target'):
            raise ValueError(
                'discrepancy_centering must be "mean" or "target".'
            )
        if (
            isinstance(ensemble_size, bool)
            or not isinstance(ensemble_size, Integral)
            or ensemble_size <= 0
        ):
            raise ValueError("ensemble_size must be a positive integer.")
        if regression_loss not in ('mse', 'huber'):
            raise ValueError('regression_loss must be "mse" or "huber".')
        if (
            isinstance(huber_delta, bool)
            or not isinstance(huber_delta, Real)
            or huber_delta <= 0
        ):
            raise ValueError("huber_delta must be numeric and positive.")
        if isinstance(input_degree, bool) or not isinstance(input_degree, int) or input_degree <= 0:
            raise ValueError("input_degree must be a positive integer.")
        if loss_weighting not in ('uniform', 'gaussian', 'alpha_mixture'):
            raise ValueError(
                'loss_weighting must be "uniform", "gaussian", or '
                '"alpha_mixture".'
            )
        if loss_weighting in ('gaussian', 'alpha_mixture') and (
            isinstance(target_bandwidth_quantile, bool)
            or not isinstance(target_bandwidth_quantile, Real)
            or not 0 < target_bandwidth_quantile <= 1
        ):
            raise ValueError(
                "target_bandwidth_quantile must be numeric and in (0, 1] "
                "for gaussian or alpha_mixture loss weighting."
            )
        if loss_weighting in ('gaussian', 'alpha_mixture') and (
            isinstance(target_bandwidth_scale, bool)
            or not isinstance(target_bandwidth_scale, Real)
            or target_bandwidth_scale <= 0
        ):
            raise ValueError(
                "target_bandwidth_scale must be numeric and positive for "
                "gaussian or alpha_mixture loss weighting."
            )
        if loss_weighting == 'alpha_mixture' and (
            isinstance(alpha, bool)
            or not isinstance(alpha, Real)
            or not 0 <= alpha <= 1
        ):
            raise ValueError(
                "alpha must be numeric and in [0, 1] for alpha_mixture "
                "loss weighting."
            )
        self.use_cosine_schedule = use_cosine_schedule
        self.loss_weighting = loss_weighting
        self.target_bandwidth_quantile = target_bandwidth_quantile
        self.target_bandwidth_scale = target_bandwidth_scale
        self.alpha = alpha
        self.input_degree = input_degree
        self.activation = activation
        self.discrepancy_centering = discrepancy_centering
        self.ensemble_size = ensemble_size
        self.regression_loss = regression_loss
        self.huber_delta = huber_delta
        self.mlp_estimator = None
        self.mlp_estimators = []
        self.member_training_losses = []
        self.discrepancy_mean = None
        self.discrepancy_center = None
        self.discrepancy_std = None

    # Computes Gaussian weights around the standardized zero-discrepancy target
    def _gaussian_target_weights(self, discrepancy_standardized):
        target_standardized = -self.discrepancy_center / self.discrepancy_std
        distances = torch.linalg.vector_norm(
            discrepancy_standardized - target_standardized, dim=1
        )
        bandwidth = (
            torch.quantile(distances, self.target_bandwidth_quantile)
            * self.target_bandwidth_scale
        )
        bandwidth = torch.clamp(
            bandwidth, min=torch.finfo(discrepancy_standardized.dtype).eps
        )
        return torch.exp(-0.5 * (distances / bandwidth) ** 2)

    # Trains an MLP regression of consensus on standardized discrepancy via minibatch Adam
    def _fit_estimator(self, consensus, discrepancy):
        consensus_dim = feature_dim(consensus)
        # Standardize discrepancy so the MLP trains on inputs of comparable scale
        self.discrepancy_mean = discrepancy.mean(dim=0)
        self.discrepancy_std = discrepancy.std(dim=0)
        self.discrepancy_center = (
            self.discrepancy_mean
            if self.discrepancy_centering == 'mean'
            else torch.zeros_like(self.discrepancy_mean)
        )
        discrepancy_standardized = (
            discrepancy - self.discrepancy_center
        ) / self.discrepancy_std
        sample_weights = None
        weighted_loss_fn = None
        if self.loss_weighting == 'gaussian':
            sample_weights = self._gaussian_target_weights(discrepancy_standardized)
        elif self.loss_weighting == 'alpha_mixture':
            # Blend uniform and Gaussian target weights by the mixing parameter alpha
            sample_weights = (1 - self.alpha) + self.alpha * self._gaussian_target_weights(
                discrepancy_standardized
            )

        if self.regression_loss == 'mse':
            loss_fn = regression_mse_loss
            if sample_weights is not None:
                weighted_loss_fn = weighted_regression_mse_loss
        else:
            loss_fn = partial(
                regression_huber_loss, delta=self.huber_delta
            )
            if sample_weights is not None:
                weighted_loss_fn = partial(
                    weighted_regression_huber_loss, delta=self.huber_delta
                )

        discrepancy_features = polynomial_features(
            discrepancy_standardized, self.input_degree
        )
        discrepancy_feature_dim = feature_dim(discrepancy_features)
        self.mlp_estimators = []
        self.member_training_losses = []
        for _ in range(self.ensemble_size):
            mlp = MLP(
                discrepancy_feature_dim, consensus_dim, self.hidden_features,
                activation=self.activation
            ).to(self.device)
            training_loss = train_torch_model(
                mlp, consensus, discrepancy_features,
                loss_fn=loss_fn,
                n_training_steps=self.n_training_steps,
                optimizer_name=self.optimizer,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                use_cosine_schedule=self.use_cosine_schedule,
                sample_weights=sample_weights,
                weighted_loss_fn=weighted_loss_fn,
                weight_decay=self.weight_decay
            )
            self.mlp_estimators.append(mlp)
            self.member_training_losses.append(training_loss)

        self.mlp_estimator = self.mlp_estimators[0]
        if self.ensemble_size == 1:
            self.training_loss = self.member_training_losses[0]
        else:
            self.training_loss = np.mean(
                self.member_training_losses, axis=0
            ).tolist()

    # Predicts standardized consensus from discrepancy_condition under the fitted MLP
    def _predict_standardized(self, discrepancy_condition):
        # Apply the fitted discrepancy standardization before the MLP forward pass
        discrepancy_standardized = (
            discrepancy_condition.to(self.device) - self.discrepancy_center
        ) / self.discrepancy_std
        discrepancy_features = polynomial_features(
            discrepancy_standardized, self.input_degree
        )
        with torch.no_grad():
            member_predictions = [
                to_numpy(mlp(discrepancy_features))
                for mlp in self.mlp_estimators
            ]
        if self.ensemble_size == 1:
            return member_predictions[0]
        return np.mean(member_predictions, axis=0)


###################
# Oracle baseline #
###################

# OracleRegression is not yet implemented; empty skeleton inheriting Regression's interface
class OracleRegression(Regression):
    pass
