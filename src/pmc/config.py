##############################
# Import necessary libraries #
##############################

# Standard libraries
import hashlib
import itertools
import json
from numbers import Integral, Real

# Third-party libraries
import numpy as np
import pandas as pd

# PMC library
from .utils import dedup_configs, json_to_config_value

##########################
# Configuration building #
##########################

# Config keys used only by neural-network methods
_NN_KEYS = (
    'hidden_features',
    'optimizer',
    'weight_decay',
    'base_learning_rate',
    'learning_rate_schedule',
    'batch_size',
    'n_training_steps',
    'n_transforms',
    'input_degree',
    'activation',
    'discrepancy_centering',
    'ensemble_size',
    'regression_loss',
    'huber_delta',
)

# Config keys used only by Gaussian-process regression methods
_GP_KEYS = (
    'gp_device',
    'gp_kernel',
    'gp_inducing_count',
    'gp_maximum_updates',
    'gp_natural_gradient_learning_rate',
    'gp_minimum_epochs',
    'gp_patience',
    'gp_relative_tolerance',
    'gp_learn_inducing_locations',
    'gp_torch_num_threads',
    'gp_confidence_level',
    'gp_local_neighborhood_exponent',
    'gp_minimum_local_count',
    'gp_maximum_local_count',
    'gp_base_learning_rate',
    'gp_batch_size',
    'gp_full_batch',
    'gp_fixed_hyperparameters',
    'gp_fixed_mean',
    'gp_fixed_lengthscale',
    'gp_fixed_outputscale',
    'gp_fixed_noise',
    'gp_dtype',
)

# Scalar-per-node DGP-parameter keys by parametric form (univariate_Gaussian's
# mu/sigma, Beta's a/b)
_SCALAR_KEYS_BY_FORM = {
    'univariate_Gaussian': ('mu', 'sigma'),
    'Beta': ('a', 'b'),
}

# Canonical column order for scalar DGP-parameter keys across all forms
_SCALAR_KEYS_CANONICAL_ORDER = tuple(
    dict.fromkeys(key for keys in _SCALAR_KEYS_BY_FORM.values() for key in keys)
)

# Use the compact result schema for high-dimensional Gaussian configurations
_COMPACT_RESULT_DIMENSION = 50

# Replace expanded semantic metadata with a fingerprint at ultra-high dimension
_ULTRA_COMPACT_RESULT_DIMENSION = 512

# Convert semantic configuration values to deterministic JSON-compatible values
def _canonical_semantic_value(value):
    if isinstance(value, np.ndarray):
        return _canonical_semantic_value(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_semantic_value(value.item())
    if isinstance(value, dict):
        return {
            key: _canonical_semantic_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_semantic_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        float_value = float(value)
        return int(float_value) if float_value.is_integer() else float_value
    return value

# Hash the full semantic DGP and consensus-weight specification
def _semantic_fingerprint(parametric_form, dgp_parameters, weights):
    semantic = {
        'DGP_parameters': dgp_parameters,
        'parametric_form': parametric_form,
        'w': weights,
    }
    payload = json.dumps(
        _canonical_semantic_value(semantic),
        allow_nan=False,
        ensure_ascii=True,
        separators=(',', ':'),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

# Expand a weight specification into weight-matrix configs
def _expand_weights(config, p):
    if 'w' not in config:
        if config.get('algorithm') == 'PMC':
            raise ValueError("w must be set for PMC configurations.")
        return [config]
    w = config['w']
    # Uniform weights: split evenly across all nodes
    if w == 'uniform':
        if p >= _ULTRA_COMPACT_RESULT_DIMENSION:
            config['w'] = [
                1.0 / config['num_nodes']
                for _ in range(config['num_nodes'])
            ]
        else:
            config['w'] = [
                np.eye(p) / config['num_nodes']
                for _ in range(config['num_nodes'])
            ]
        return [config]
    # Optimal weights: compute precision-weighted contributions
    if w == 'optimal':
        # Optimal weights are only implemented for the univariate Gaussian case
        if config['parametric_form'] != 'univariate_Gaussian':
            raise NotImplementedError(
                f"Optimal weights not implemented for parametric form: {config['parametric_form']}"
            )
        # Extract the DGP parameters
        dgp = config['DGP_parameters']
        # Compute the precision-weighted contribution of each node
        variances = np.array(dgp['sigma'])**2
        precisions = 1.0 / variances
        config['w'] = [
            np.array([[float(precisions[k] / np.sum(precisions))]])
            for k in range(config['num_nodes'])
        ]
        return [config]
    # Grid over weight configurations summing to a fixed number of steps
    if isinstance(w, tuple) and w[0] == 'all':
        num_nodes = config['num_nodes']
        steps = w[1] if len(w) > 1 else 10
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps <= 0
        ):
            raise ValueError("w = ['all', steps] requires positive integer steps.")
        # Generate all integer partitions of steps across the nodes
        grid = [
            pt for pt in itertools.product(range(steps + 1), repeat=num_nodes)
            if sum(pt) == steps
        ]
        # Expand into one config per grid point
        expanded = []
        for pt in grid:
            c = config.copy()
            c['w'] = [np.eye(p) * (pt_k / steps) for pt_k in pt]
            expanded.append(c)
        return expanded
    return [config]

# Compute the DGP schema (max num_nodes/p, scalar keys, CNF usage) for a batch
def _compute_dgp_schema(configurations):
    max_num_nodes = max(c['num_nodes'] for c in configurations)
    max_p = 0
    scalar_keys_in_use = set()
    vector_keys_in_use = {'mu'}
    matrix_keys_in_use = set()
    for c in configurations:
        if c['parametric_form'] in ('multivariate_Gaussian', 'bivariate_Gaussian'):
            dgp_parameters = c['DGP_parameters']
            p = np.asarray(dgp_parameters['mu']).shape[1]
            max_p = max(max_p, p)
            if 'cov_diag' in dgp_parameters:
                vector_keys_in_use.add('cov_diag')
            if 'cov' in dgp_parameters:
                matrix_keys_in_use.add('cov')
        scalar_keys_in_use.update(_SCALAR_KEYS_BY_FORM.get(c['parametric_form'], ()))
    scalar_keys = tuple(key for key in _SCALAR_KEYS_CANONICAL_ORDER if key in scalar_keys_in_use)
    uses_cnf = any(c.get('method') == 'CNF' for c in configurations)
    return {
        'max_num_nodes': max_num_nodes,
        'max_p': max_p,
        'scalar_keys': scalar_keys,
        'vector_keys': tuple(
            key for key in ('mu', 'cov_diag')
            if key in vector_keys_in_use
        ),
        'matrix_keys': tuple(
            key for key in ('cov',)
            if key in matrix_keys_in_use
        ),
        'uses_cnf': uses_cnf,
        'compact': max_p >= _COMPACT_RESULT_DIMENSION,
        'ultra_compact': max_p >= _ULTRA_COMPACT_RESULT_DIMENSION,
    }

# Validate a reserved selector without adding it to the configuration grid
def _validate_selector(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value

# Build the list of per-simulation configs from JSON config files
def build_configs(config_files):
    # Load and merge all config files
    configuration_parameters = {}
    selectors = {}
    for cf in config_files:
        with open(cf) as f:
            ext = json.load(f)
        for k, v in ext.items():
            if k == 'validated_cell_overrides':
                continue
            if k in ('selected_n', 'selected_p'):
                selectors[k] = v
            else:
                configuration_parameters[k] = json_to_config_value(k, v)

    # Extract reserved keys
    seed = configuration_parameters.pop('seed', [None])[0]
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int)
    ):
        raise ValueError("seed must be an integer or null.")
    n_iterations = configuration_parameters.pop('n_iterations', None)
    if n_iterations is None:
        raise ValueError("n_iterations must be set in a config file.")
    n_iterations = n_iterations[0]
    if (
        isinstance(n_iterations, bool)
        or not isinstance(n_iterations, int)
        or n_iterations <= 0
    ):
        raise ValueError("n_iterations must be a positive integer.")
    iteration_start_values = configuration_parameters.pop('iteration_start', [0])
    if (
        len(iteration_start_values) != 1
        or isinstance(iteration_start_values[0], bool)
        or not isinstance(iteration_start_values[0], int)
        or iteration_start_values[0] < 0
    ):
        raise ValueError("iteration_start must be a nonnegative integer.")
    iteration_start = iteration_start_values[0]
    h_values = configuration_parameters.pop('h', ['identity'])
    selected_n = (
        _validate_selector(selectors['selected_n'], 'selected_n')
        if 'selected_n' in selectors else None
    )
    selected_p = (
        _validate_selector(selectors['selected_p'], 'selected_p')
        if 'selected_p' in selectors else None
    )
    configuration_parameters.setdefault('discrepancy_method', ['consecutive'])

    # Cartesian product over remaining parameters
    keys = list(configuration_parameters.keys())
    base_configurations = [
        dict(zip(keys, v))
        for v in itertools.product(*configuration_parameters.values())
    ]

    # Null out inapplicable parameters per method/algorithm and deduplicate
    base_configurations = dedup_configs(
        base_configurations,
        method_nulls={
            'Gaussian': [*_NN_KEYS, *_GP_KEYS, 'degree'],
            'linear': [*_NN_KEYS, *_GP_KEYS],
            'CNF': [
                'degree', 'weight_decay', 'activation', 'discrepancy_centering',
                'ensemble_size', 'regression_loss', 'huber_delta', *_GP_KEYS
            ],
            'MLP': ['degree', *_GP_KEYS],
            'GP': [*_NN_KEYS, 'degree'],
        },
        algorithm_nulls={
            'PMC': ['n_inference'],
            'Oracle': [
                'method', 'degree', 'discrepancy_method', *_NN_KEYS, *_GP_KEYS
            ],
        },
    )

    # Compute the DGP schema shared by every config in this batch, for CSV rectangularity
    dgp_schema = _compute_dgp_schema(base_configurations)

    # Expand weight vectors once to establish the stable within-repetition grid
    expanded_base_configurations = []
    for base_config in base_configurations:
        config = base_config.copy()
        dgp_params = config['DGP_parameters']
        if config['parametric_form'] in (
            'multivariate_Gaussian', 'bivariate_Gaussian'
        ):
            p = np.asarray(dgp_params['mu']).shape[1]
        else:
            p = 1
        expanded_base_configurations.extend(_expand_weights(config, p))

    # Assign continuation-safe seeds across the requested global iterations
    configurations = []
    n_base = len(expanded_base_configurations)
    for iteration in range(iteration_start, iteration_start + n_iterations):
        for i, base_config in enumerate(expanded_base_configurations):
            config = base_config.copy()
            config['seed'] = seed + iteration * n_base + i if seed is not None else None
            config['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            config['iteration_id'] = iteration
            config['dgp_schema'] = dgp_schema
            config['h_values'] = h_values
            configurations.append(config)

    # Select from the complete batch to preserve schema and seed indices
    if selected_n is not None:
        configurations = [c for c in configurations if c['n'] == selected_n]
    if selected_p is not None:
        configurations = [
            c for c in configurations
            if (
                len(c['DGP_parameters']['mu'][0])
                if c['parametric_form'] in (
                    'multivariate_Gaussian', 'bivariate_Gaussian'
                ) else 1
            ) == selected_p
        ]
    if not configurations:
        raise ValueError("Selected n/p excludes all configurations.")

    return configurations
