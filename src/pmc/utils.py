##############################
# Import necessary libraries #
##############################

# Standard libraries
import itertools
from numbers import Real

# Third-party libraries
import numpy as np
import ot
import torch
from scipy.stats import multivariate_normal

#############################
# Shared validation helpers #
#############################

# Validate that value is a positive integer, raising a labeled ValueError otherwise
def validate_positive_int(value, label):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"The {label} must be a positive integer, got {value}")


# Validate that x is list-like, convert it to a float array, and check for NaN/Inf values
def to_finite_float_array(x, name):
    # Type check
    if not isinstance(x, (list, tuple, np.ndarray)):
        raise TypeError(f"{name} must be list-like (list, tuple, or numpy array).")

    # Safe conversion to float array
    try:
        array = np.array(x, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must contain only numeric values.")

    # Check for invalid numbers (NaN or Infinity)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} cannot contain NaN or Inf.")

    return array


################################
# Shared configuration helpers #
################################

# Converts a JSON-decoded config value into the list-of-options form used for config expansion
def json_to_config_value(key, val, tuple_keys=('w', 'method', 'hidden_features')):
    # JSON has no tuple type; restore tuples for keys that expect them
    items = val if isinstance(val, list) else [val]
    if key in tuple_keys:
        return [tuple(v) if isinstance(v, list) else v for v in items]
    return items


# Nulls out cfg's keys that don't apply given cfg's value for field, per nulls_map
def _null_inapplicable_keys(cfg, field, nulls_map):
    if nulls_map and cfg.get(field) in nulls_map:
        for k in nulls_map[cfg[field]]:
            cfg[k] = None


# Deduplicates config dicts, nulling out keys that do not apply to each config's method or algorithm
def dedup_configs(configs, method_nulls=None, algorithm_nulls=None):
    seen = set()
    result = []
    for cfg in configs:
        _null_inapplicable_keys(cfg, 'method', method_nulls)
        _null_inapplicable_keys(cfg, 'algorithm', algorithm_nulls)
        key = tuple((k, str(cfg[k])) for k in sorted(cfg))
        if key not in seen:
            seen.add(key)
            result.append(cfg)
    return result


#############################
# Shared CSV output helpers #
#############################

# Appends a dataframe's rows to a CSV file, writing a header only when starting a new file
def append_to_csv(df, path, state):
    import os
    import tempfile

    import pandas as pd

    # Copy the current CSV and the new rows into a temporary file, then publish it atomically
    def _replace_with_appended_rows(header):
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(fd, "wb") as temporary_file:
                if os.path.exists(path):
                    with open(path, "rb") as existing_file:
                        existing_bytes = existing_file.read()
                    temporary_file.write(existing_bytes)
                    if existing_bytes and not existing_bytes.endswith(b"\n"):
                        temporary_file.write(b"\n")
                csv_text = df.to_csv(index=False, header=header)
                temporary_file.write(csv_text.encode("utf-8"))
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        except Exception:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise

    if df.empty:
        return

    new_columns = list(df.columns)
    if not state['initialized']:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                existing_columns = list(pd.read_csv(path, nrows=0).columns)
            except pd.errors.EmptyDataError:
                raise ValueError(f"Output is empty and cannot be resumed: {path}") from None
            if existing_columns == new_columns:
                state['mode'] = 'a'
                state['header'] = False
            else:
                missing = [c for c in existing_columns if c not in new_columns]
                extra = [c for c in new_columns if c not in existing_columns]
                raise ValueError(
                    f"Column mismatch appending to {path}: "
                    f"missing columns {missing}, extra columns {extra}"
                )
        elif os.path.exists(path):
            raise ValueError(f"Output is zero-byte and cannot be resumed: {path}")
        _replace_with_appended_rows(state['header'])
        state['initialized'] = True
        state['mode'] = 'a'
        state['header'] = False
        state['columns'] = new_columns
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Output disappeared during append: {path}")
        if state.get('columns') != new_columns:
            raise ValueError(
                f"Column mismatch appending to {path}: "
                f"expected columns {state.get('columns')}, got columns {new_columns}"
            )
        _replace_with_appended_rows(False)


##############################################
# Shared DGP-parameter serialization helpers #
##############################################

# Builds a 1-indexed, underscore-joined DGP-parameter CSV column name
def dgp_param_column(key, node, *dims):
    return '_'.join([key, str(node), *map(str, dims)])


#################################
# Shared input-type conversions #
#################################

# Converts x to a numpy array, whether given as a torch tensor or array-like
def to_numpy(x):
    return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


# Converts x to a torch tensor of the given dtype, whether given as a torch tensor or array-like
def to_tensor(x, dtype=torch.float32):
    return x if torch.is_tensor(x) else torch.tensor(x, dtype=dtype)


# Converts discrepancy_condition to a torch tensor and adds a batch dimension if it is 1-D
def ensure_batched_tensor(x):
    x = to_tensor(x)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x


# Returns the feature dimensionality of x, treating a 1D array as a single feature
def feature_dim(x):
    return x.shape[1] if x.ndim > 1 else 1


# Expand inputs into every monomial of total degree up to degree
def polynomial_features(x, degree=1):
    if degree == 1:
        return x
    columns = []
    for total_degree in range(1, degree + 1):
        for indices in itertools.combinations_with_replacement(
            range(x.shape[1]), total_degree
        ):
            column = x[:, indices[0]]
            for index in indices[1:]:
                column = column * x[:, index]
            columns.append(column)
    return (
        torch.stack(columns, dim=1)
        if torch.is_tensor(x)
        else np.column_stack(columns)
    )


#############################
# Shared training utilities #
#############################

# Runs a minibatch optimization loop and returns per-step losses
def train_torch_model(model, consensus, discrepancy, loss_fn, n_training_steps, optimizer_name,
                       learning_rate, batch_size=None, use_cosine_schedule=False,
                       sample_weights=None, weighted_loss_fn=None, weight_decay=0.0):
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, Real)
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be numeric and nonnegative.")
    if optimizer_name == 'Adam':
        if weight_decay == 0:
            opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            opt = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
    elif optimizer_name == 'AdamW':
        opt = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError("Unsupported optimizer. Please use 'Adam' or 'AdamW'.")

    scheduler = None
    if use_cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_training_steps
        )

    n = consensus.shape[0]
    effective_batch = min(batch_size, n) if batch_size is not None else n
    training_loss = []
    for _ in range(n_training_steps):
        if effective_batch < n:
            # Sample with replacement so cost is independent of n
            idx = torch.randint(0, n, (effective_batch,), device=consensus.device)
            consensus_batch = consensus[idx]
            discrepancy_batch = discrepancy[idx]
            weights_batch = sample_weights[idx] if sample_weights is not None else None
        else:
            consensus_batch = consensus
            discrepancy_batch = discrepancy
            weights_batch = sample_weights
        opt.zero_grad()
        if weights_batch is not None and weighted_loss_fn is not None:
            loss = weighted_loss_fn(
                model, consensus_batch, discrepancy_batch, weights_batch
            )
        else:
            loss = loss_fn(model, consensus_batch, discrepancy_batch)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        training_loss.append(loss.item())

    return training_loss


# Computes normalized observation-weighted mean squared error
def weighted_mse_loss(predictions, targets, sample_weights):
    row_mse = torch.mean((predictions - targets) ** 2, dim=1)
    return torch.sum(sample_weights * row_mse) / torch.sum(sample_weights)


# Computes mean squared error for a regression model and minibatch
def regression_mse_loss(model, targets, inputs):
    return torch.nn.functional.mse_loss(model(inputs), targets)


# Computes weighted MSE for a regression model and minibatch
def weighted_regression_mse_loss(model, targets, inputs, sample_weights):
    return weighted_mse_loss(model(inputs), targets, sample_weights)


# Computes normalized observation-weighted Huber loss
def weighted_huber_loss(predictions, targets, sample_weights, delta=1.0):
    element_loss = torch.nn.functional.huber_loss(
        predictions, targets, reduction='none', delta=delta
    )
    row_loss = torch.mean(element_loss, dim=1)
    return torch.sum(sample_weights * row_loss) / torch.sum(sample_weights)


# Computes Huber loss for a regression model and minibatch
def regression_huber_loss(model, targets, inputs, delta=1.0):
    return torch.nn.functional.huber_loss(
        model(inputs), targets, delta=delta
    )


# Computes weighted Huber loss for a regression model and minibatch
def weighted_regression_huber_loss(
    model, targets, inputs, sample_weights, delta=1.0
):
    return weighted_huber_loss(
        model(inputs), targets, sample_weights, delta=delta
    )


_ACTIVATION_MODULES = {
    'relu': torch.nn.ReLU,
    'silu': torch.nn.SiLU,
    'tanh': torch.nn.Tanh,
}


# Validates that activation names a supported neural-network activation
def validate_activation(activation):
    if not isinstance(activation, str) or activation not in _ACTIVATION_MODULES:
        supported = ', '.join(_ACTIVATION_MODULES)
        raise ValueError(
            f"activation must be one of {supported}, got {activation!r}."
        )


# Builds a feedforward network mapping input_dim to output_dim
class MLP(torch.nn.Module):
    # Initializes the network layers with the selected activation
    def __init__(self, input_dim, output_dim, hidden_features, activation='relu'):
        super().__init__()
        validate_activation(activation)
        activation_module = _ACTIVATION_MODULES[activation]
        layers = []
        in_dim = input_dim
        for h in hidden_features:
            layers.extend([torch.nn.Linear(in_dim, h), activation_module()])
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    # Applies the network to x
    def forward(self, x):
        return self.net(x)


# Fits C ~ D via least squares, with degree > 1 adding per-dimension powers of D
def fit_linear_regression(consensus, discrepancy, degree=1):
    consensus_np = to_numpy(consensus)
    discrepancy_np = to_numpy(discrepancy)
    discrepancy_poly = polynomial_features(discrepancy_np, degree)
    discrepancy_aug = np.hstack(
        [discrepancy_poly, np.ones((discrepancy_np.shape[0], 1))]
    )
    coeffs, _, _, _ = np.linalg.lstsq(discrepancy_aug, consensus_np, rcond=None)
    weights = coeffs[:-1]
    bias = coeffs[-1]
    residuals = consensus_np - (discrepancy_poly @ weights + bias)
    residual_cov = np.cov(residuals, rowvar=False).reshape(
        consensus_np.shape[1], consensus_np.shape[1]
    )
    return weights, bias, residual_cov


#################################
# Shared distributional metrics #
#################################

# Computes the analytical KL divergence KL(N(mu_hat, cov_hat) || N(mu_true, cov_true))
def kl_gaussian(mu_hat, cov_hat, mu_true, cov_true):
    d = len(mu_hat)
    cov_true_inv = np.linalg.inv(cov_true)
    diff = np.array(mu_true) - np.array(mu_hat)
    _, logdet_true = np.linalg.slogdet(cov_true)
    _, logdet_hat = np.linalg.slogdet(cov_hat)
    return float(
        0.5 * (
            np.trace(cov_true_inv @ cov_hat) + diff @ cov_true_inv @ diff
            - d + logdet_true - logdet_hat
        )
    )


# Computes the analytical W2 (Bures metric) distance between two Gaussians
def w2_gaussian(mu_hat, cov_hat, mu_true, cov_true):
    from scipy.linalg import sqrtm
    diff = np.array(mu_hat) - np.array(mu_true)
    sqrt_hat = np.real(sqrtm(cov_hat))
    bures_mat = np.real(sqrtm(sqrt_hat @ cov_true @ sqrt_hat))
    bures_sq = np.real(np.trace(cov_hat + cov_true - 2 * bures_mat))
    return float(np.sqrt(max(float(np.dot(diff, diff)) + bures_sq, 0.0)))


# Computes Wasserstein-1 and Wasserstein-2 distances between two 1D empirical samples
def wasserstein_1d(true_samples, inferred_samples):
    true_sorted = np.sort(true_samples)
    inferred_sorted = np.sort(inferred_samples.flatten())
    w1 = np.mean(np.abs(true_sorted - inferred_sorted))
    w2 = np.sqrt(np.mean((true_sorted - inferred_sorted)**2))
    return w1, w2


# Computes analytic W2 and empirical-OT W1 between a fitted and a true multivariate Gaussian
def multivariate_wasserstein(mu_hat, cov_hat, mu_true, cov_true, samples_hat, samples_true):
    w2 = w2_gaussian(mu_hat, cov_hat, mu_true, cov_true)
    n = len(samples_hat)
    aw = np.ones(n) / n
    m1 = ot.dist(samples_hat, samples_true, metric='euclidean')
    w1 = float(ot.emd2(aw, aw, m1))
    return w1, w2


####################
# Filename helpers #
####################

# Function to convert the parameters of the DGP to a string for the filename
def parameters_to_filename(parameters):
    return '_'.join(
        f"{key}_{value}" for key, value in parameters.items()
        if not isinstance(value, np.ndarray)
    )


# Function to convert the weights to a string for the filename
def weights_to_filename(w):
    w_str = []
    for i, w_i in enumerate(w):
        val = (
            float(w_i[0, 0])
            if isinstance(w_i, np.ndarray) and w_i.ndim == 2
            else w_i.flatten()[0]
        )
        w_str.append(f"w{i+1}_{val}")
    return '_'.join(w_str)


###########################
# Shared Gaussian helpers #
###########################

# Builds the symmetric Gaussian x-range used for 1D density/marginal curves
def gaussian_x_range(mu, sigma, n_std=4, n_points=100):
    return np.linspace(mu - n_std * sigma, mu + n_std * sigma, n_points)


# Returns the mean and standard deviation of the 1D Gaussian marginal for dimension i
def gaussian_marginal_1d(mu, cov, i):
    return mu[i], np.sqrt(cov[i, i])


# Returns the mean vector and covariance slice of the 2D Gaussian marginal for dims i, j
def gaussian_slice_2d(mu, cov, i, j):
    return mu[[j, i]], cov[np.ix_([j, i], [j, i])]


# Builds a bivariate normal density grid over mu_2d/cov_2d for contour plotting
def bivariate_normal_density_grid(mu_2d, cov_2d, n_std=3, step=0.05):
    x_std, y_std = np.sqrt(cov_2d[0, 0]), np.sqrt(cov_2d[1, 1])
    x, y = np.mgrid[mu_2d[0]-n_std*x_std:mu_2d[0]+n_std*x_std:step,
                    mu_2d[1]-n_std*y_std:mu_2d[1]+n_std*y_std:step]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mu_2d, cov_2d)
    return x, y, rv.pdf(pos)
