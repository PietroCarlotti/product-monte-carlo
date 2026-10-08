##############################
# Import necessary libraries #
##############################

# Third-party libraries
import numpy as np
from scipy.stats import beta, multivariate_normal, norm

# PMC library
from .utils import kl_gaussian, multivariate_wasserstein, to_numpy, wasserstein_1d

######################
# Evaluation metrics #
######################

# Compute MSE between inferred and true product-distribution means
def mse(dgp, consensus_hat, h=None, true_mean=None, h_name=None):
    # Flatten the point estimate of E[h(consensus)]
    consensus_hat = np.array(consensus_hat).flatten()
    # Flatten the true E[h(consensus)] under the product distribution
    if true_mean is None:
        true_mean = (
            dgp.mean_product(h=h)
            if h_name is None
            else dgp.mean_product(h=h, h_name=h_name)
        )
    true_mean = np.array(true_mean).flatten()
    return float(np.sum((consensus_hat - true_mean) ** 2))

# Convergence check based on relative improvement in training loss
def check_convergence(loss_history, window_frac=0.1, threshold=0.01):
    loss = np.array(loss_history)
    n = len(loss)

    # Size of each of the two comparison windows
    window = max(1, int(n * window_frac))

    # Not enough history to compare two non-overlapping windows
    if n < 2 * window:
        return None, None

    # Split the tail of the loss history into an earlier and a later window
    pre = loss[-2 * window:-window]
    tail = loss[-window:]
    mean_pre = np.mean(pre)

    # Avoid dividing by zero when normalizing the improvement below
    if mean_pre == 0:
        return None, None

    # Relative drop in mean loss from the earlier window to the later one
    relative_improvement = float((mean_pre - np.mean(tail)) / abs(mean_pre))
    return bool(relative_improvement < threshold), relative_improvement

# Extracts (mu_hat, cov_hat) of an already-fitted Gaussian estimator, if one is applicable
def _try_extract_fitted_gaussian(estimator, dgp, discrepancy_condition):
    has_fitted_gaussian = (
        getattr(estimator, 'linear_weights', None) is not None
        or getattr(estimator, 'mu_product', None) is not None
    )
    if estimator is None or dgp.parametric_form not in (
        'univariate_Gaussian', 'bivariate_Gaussian', 'multivariate_Gaussian'
    ) or not has_fitted_gaussian:
        return None, None
    return _extract_fitted_gaussian_params(estimator, discrepancy_condition)

# Extracts (mu_hat, cov_hat) of the fitted conditional Gaussian, without any new estimation
def _extract_fitted_gaussian_params(estimator, discrepancy_condition=None):
    if hasattr(estimator, 'mu_product') and hasattr(estimator, 'cov_product'):
        # GaussianOracleSampling: parameters already in original space
        return np.atleast_1d(estimator.mu_product), np.atleast_2d(estimator.cov_product)

    # Gaussian PMC conditional fitted on standardized consensus, must back-transform
    if discrepancy_condition is None:
        d_np = np.zeros(1)
    else:
        d_np = to_numpy(discrepancy_condition).flatten()

    # Reuse the estimator's own conditional-Gaussian computation, in standardized space
    mu_hat_std, cov_hat_std = estimator._condition_gaussian(d_np)

    # Back-transform: consensus_original = consensus_std * std + mean
    std = np.array(estimator.consensus_transformed_std)
    mean = np.array(estimator.consensus_transformed_mean)
    mu_hat = mu_hat_std * std + mean
    cov_hat = np.diag(std) @ cov_hat_std @ np.diag(std)
    return mu_hat, cov_hat

# Extract true Gaussian product parameters as a vector and covariance matrix
def _true_gaussian_params(dgp):
    if dgp.parametric_form == 'univariate_Gaussian':
        mu_true = np.array([dgp.mu_product])
        cov_true = np.array([[dgp.sigma_product**2]])
    elif hasattr(dgp, 'cov_product_diag'):
        mu_true = np.atleast_1d(dgp.mu_product)
        cov_true = np.diag(np.atleast_1d(dgp.cov_product_diag))
    else:
        mu_true = np.atleast_1d(dgp.mu_product)
        cov_true = np.atleast_2d(dgp.cov_product)
    return mu_true, cov_true

# Kullback-Leibler divergence estimation
def kl(dgp, consensus_inference, log_p_consensus_inference, estimator=None,
       discrepancy_condition=None):
    # Use the exact closed-form Gaussian KL if the estimator has a fitted Gaussian
    mu_hat, cov_hat = _try_extract_fitted_gaussian(estimator, dgp, discrepancy_condition)
    if mu_hat is not None:
        mu_true, cov_true = _true_gaussian_params(dgp)
        return kl_gaussian(mu_hat, cov_hat, mu_true, cov_true)

    # Compute the log-probability of the inferred samples under the true product distribution
    if dgp.parametric_form == "univariate_Gaussian":
        # Univariate Gaussian product distribution
        log_p_true = norm.logpdf(
            consensus_inference,
            loc=dgp.mu_product,
            scale=dgp.sigma_product
        ).flatten()

    elif dgp.parametric_form == "bivariate_Gaussian":
        # Bivariate Gaussian product distribution
        log_p_true = multivariate_normal.logpdf(
            consensus_inference,
            mean=dgp.mu_product,
            cov=dgp.cov_product
        )

    elif dgp.parametric_form == "multivariate_Gaussian":
        # Multivariate Gaussian product distribution
        log_p_true = multivariate_normal.logpdf(
            consensus_inference,
            mean=dgp.mu_product,
            cov=dgp.cov_product
        )

    elif dgp.parametric_form == "Beta":
        # Beta product distribution
        log_p_true = beta.logpdf(
            consensus_inference,
            a=dgp.a_product,
            b=dgp.b_product
        ).flatten()

    else:
        raise ValueError(f"Unsupported DGP parametric form: {dgp.parametric_form}")

    # Estimate KL divergence
    kl_true_estimate = np.mean(log_p_consensus_inference - log_p_true)

    return kl_true_estimate

# Wasserstein distance estimation
def wasserstein(dgp, consensus_inference, n_subsample=None, estimator=None,
                 discrepancy_condition=None):
    # Use the exact closed-form/OT-based Wasserstein if the estimator has a fitted Gaussian
    mu_hat, cov_hat = _try_extract_fitted_gaussian(estimator, dgp, discrepancy_condition)
    if mu_hat is not None:
        if dgp.parametric_form == 'univariate_Gaussian':
            # Exact closed-form W1/W2 between two univariate Gaussians
            mu_h = float(np.array(mu_hat).flatten()[0])
            sig_h = float(np.sqrt(np.array(cov_hat).flatten()[0]))
            w2 = float(np.sqrt((mu_h - dgp.mu_product)**2 + (sig_h - dgp.sigma_product)**2))
            a = mu_h - dgp.mu_product
            b = sig_h - dgp.sigma_product
            if abs(b) < 1e-15:
                # Equal standard deviations: W1 reduces to the mean difference
                w1 = float(abs(a))
            else:
                # Folded-normal identity for the exact W1 between two Gaussians
                ratio = a / b
                w1 = float(abs(b) * (2 * norm.pdf(ratio) + ratio * (2 * norm.cdf(ratio) - 1)))
        else:
            # Approximate multivariate W1 via optimal transport; W2 stays closed-form
            n = n_subsample or 2000
            mu_true, cov_true = _true_gaussian_params(dgp)
            # Estimated-side samples: fresh draws from the fitted Gaussian (mu_hat, cov_hat)
            fitted_gaussian = multivariate_normal(
                mean=np.array(mu_hat).flatten(),
                cov=np.array(cov_hat),
                allow_singular=True,
            )
            samples_hat = np.asarray(fitted_gaussian.rvs(size=n)).reshape(
                -1, np.size(mu_hat)
            )
            # True-side samples: fresh draws from the true product distribution
            samples_true = dgp.sample_product(n)
            w1, w2 = multivariate_wasserstein(
                mu_hat, cov_hat, mu_true, cov_true, samples_hat, samples_true
            )
        return w1, w2

    if dgp.parametric_form == "univariate_Gaussian":
        # Sample from the true product distribution
        consensus_true = dgp.sample_product(consensus_inference.shape[0]).flatten()
        w1, w2 = wasserstein_1d(consensus_true, consensus_inference)

    elif dgp.parametric_form == "bivariate_Gaussian":
        # Estimate the mean/covariance of the inferred samples for the closed-form W2
        consensus_np = to_numpy(consensus_inference)
        mu_hat = np.mean(consensus_np, axis=0)
        cov_hat = np.cov(consensus_np, rowvar=False).reshape(mu_hat.shape[0], mu_hat.shape[0])
        n = n_subsample or 2000
        # True-side samples: fresh draws from the true product distribution
        samples_true = dgp.sample_product(n)
        # Estimated-side samples: subsample of the actual empirical inference draws
        samples_hat = consensus_np[
            np.random.choice(len(consensus_np), n, replace=len(consensus_np) < n)
        ]
        w1, w2 = multivariate_wasserstein(
            mu_hat, cov_hat, dgp.mu_product, dgp.cov_product, samples_hat, samples_true
        )

    elif dgp.parametric_form == "multivariate_Gaussian":
        # Estimate the mean/covariance of the inferred samples for the closed-form W2
        consensus_np = to_numpy(consensus_inference)
        mu_hat = np.mean(consensus_np, axis=0)
        cov_hat = np.cov(consensus_np, rowvar=False).reshape(mu_hat.shape[0], mu_hat.shape[0])
        n = n_subsample or 2000
        # True-side samples: fresh draws from the true product distribution
        samples_true = dgp.sample_product(n)
        # Estimated-side samples: subsample of the actual empirical inference draws
        samples_hat = consensus_np[
            np.random.choice(len(consensus_np), n, replace=len(consensus_np) < n)
        ]
        w1, w2 = multivariate_wasserstein(
            mu_hat, cov_hat, dgp.mu_product, dgp.cov_product, samples_hat, samples_true
        )

    elif dgp.parametric_form == "Beta":
        # Sample from the true product distribution
        consensus_true = dgp.sample_product(consensus_inference.shape[0]).flatten()
        w1, w2 = wasserstein_1d(consensus_true, consensus_inference)

    else:
        raise ValueError(f"Unsupported DGP parametric form: {dgp.parametric_form}")

    return w1, w2
