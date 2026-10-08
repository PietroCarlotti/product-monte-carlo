##############################
# Import necessary libraries #
##############################
import numpy as np

from .utils import to_finite_float_array, validate_positive_int

#################################
# Data Generating Process (DGP) #
#################################

# General class for Data Generating Process (DGP)
class DGP:
    # Validate and store the number of nodes shared by all subclasses
    def __init__(self, num_nodes):
        # Validate num_nodes
        validate_positive_int(num_nodes, "number of nodes")

        # Number of nodes (K in the paper)
        self.num_nodes = num_nodes

        # To be defined in subclasses
        self.parametric_form = None

    # Draw one sample per node from each node's own distribution
    def sample(self):
        raise NotImplementedError("Subclasses must implement sample()")

    # Draw n samples from the product distribution
    def sample_product(self, n):
        raise NotImplementedError("Subclasses must implement sample_product()")

    # Compute named supported moments analytically and otherwise estimate E[h(X)] by Monte Carlo
    def mean_product(self, h=None, n_mc=100000, h_name=None):
        if h_name == 'square':
            return self._closed_form_square_mean_product()
        if h_name == 'normalized_sum':
            mean = np.asarray(self._closed_form_mean_product())
            return np.array([np.sum(mean) / np.sqrt(mean.size)])
        if h is None or h_name == 'identity':
            return self._closed_form_mean_product()
        return np.mean(h(self.sample_product(n_mc)), axis=0)

    # Compute the closed-form E[X] under the product distribution
    def _closed_form_mean_product(self):
        raise NotImplementedError("Subclasses must implement _closed_form_mean_product()")

    # Compute the closed-form componentwise E[X^2] under the product distribution
    def _closed_form_square_mean_product(self):
        raise NotImplementedError(
            "Subclasses must implement _closed_form_square_mean_product()"
        )

# Shared parent for all Gaussian-family DGPs (univariate, bivariate, multivariate)
class GaussianDGP(DGP):
    # Validate per-node covariance matrices for shape, symmetry and positive-definiteness
    @staticmethod
    def _validate_covariance_matrices(cov_array):
        num_nodes = cov_array.shape[0]
        for i in range(num_nodes):
            matrix = cov_array[i]

            # Check for symmetry
            if not np.allclose(matrix, matrix.T):
                raise ValueError(f"Covariance matrix for node {i} is not symmetric.")

            # Check for positive definiteness
            eigenvalues = np.linalg.eigvalsh(matrix)
            if not np.all(eigenvalues > 0):
                raise ValueError(f"Covariance matrix for node {i} is not positive definite.")

    # Combine per-node Gaussians into the resulting product-Gaussian parameters
    @staticmethod
    def _combine_gaussians(mu_array, cov_array):
        # Compute parameters for the resulting Gaussian distribution
        # obtained by multiplying num_nodes Gaussian distributions
        lambda_array = np.linalg.inv(cov_array)
        lambda_product = np.sum(lambda_array, axis=0)
        cov_product = np.linalg.inv(lambda_product)

        weighted_mu_array = np.einsum('ijk,ik->ij', lambda_array, mu_array)
        mu_product = np.linalg.solve(lambda_product, np.sum(weighted_mu_array, axis=0))

        return mu_product, cov_product

    # Validate shape, consistency, symmetry and positive-definiteness of
    # multi-dimensional Gaussian DGP inputs; shared by BivariateGaussianDGP
    # and MultivariateGaussianDGP
    @staticmethod
    def _validate_gaussian_dgp_inputs(mu, cov):
        # Validate mu and cov, converting them to finite float arrays
        mu_array = to_finite_float_array(mu, "mu")
        # cov: per-node covariance matrices (Sigma in the paper)
        cov_array = to_finite_float_array(cov, "cov")

        # Shape validation
        if mu_array.ndim != 2:
            raise ValueError(f"mu must be 2D (num_nodes x p), got {mu_array.ndim}D")
        if cov_array.ndim != 3:
            raise ValueError(f"cov must be 3D (num_nodes x p x p), got {cov_array.ndim}D")

        k_mu, p_mu = mu_array.shape
        k_cov, p_sig1, p_sig2 = cov_array.shape

        # Consistency checks
        if k_mu != k_cov:
            raise ValueError(f"Number of nodes mismatch: mu has {k_mu} nodes, cov has {k_cov}")

        if p_mu != p_sig1 or p_mu != p_sig2:
            raise ValueError(f"Dimension p mismatch: mu is {p_mu}-dim, cov is {p_sig1}x{p_sig2}")

        # Number of nodes is determined by mu's shape
        num_nodes = k_mu

        # Dimensionality is determined by mu's shape
        p = p_mu

        GaussianDGP._validate_covariance_matrices(cov_array)

        return num_nodes, p, mu_array, cov_array

    # Finish constructing a multi-dimensional Gaussian DGP: initialize the
    # parent DGP class and assign the validated and combined parameters;
    # shared by BivariateGaussianDGP and MultivariateGaussianDGP
    def _finish_gaussian_init(self, num_nodes, p, mu_array, cov_array, parametric_form):
        # Initialize parent DGP class
        super().__init__(num_nodes)

        self.parametric_form = parametric_form

        # Compute parameters for the resulting Gaussian distribution
        # obtained by multiplying num_nodes Gaussian distributions
        mu_product, cov_product = self._combine_gaussians(mu_array, cov_array)

        # Assign validated parameters
        self.p = p
        self.mu = mu_array
        self.cov = cov_array
        self.mu_product = mu_product
        self.cov_product = cov_product

    # Draw n samples from each of the num_nodes Gaussian nodes
    def sample(self, n):
        # Validate n
        validate_positive_int(n, "sample size")

        # Generate samples for each node
        samples = np.zeros((n, self.num_nodes, self.p))
        for i in range(self.num_nodes):
            samples[:, i, :] = np.random.multivariate_normal(
                mean=self.mu[i],
                cov=self.cov[i],
                size=n
            )
        return samples

    # Draw n samples from the resulting product-Gaussian distribution
    def sample_product(self, n):
        return np.random.multivariate_normal(mean=self.mu_product, cov=self.cov_product, size=n)

    # Compute the closed-form E[X] under the product-Gaussian distribution
    def _closed_form_mean_product(self):
        return np.array(self.mu_product)

    # Compute the closed-form componentwise E[X^2] under the product-Gaussian distribution
    def _closed_form_square_mean_product(self):
        return np.array(self.mu_product) ** 2 + np.diag(self.cov_product)

# DGP where each node is a univariate Gaussian distribution
class UnivariateGaussianDGP(GaussianDGP):
    # Validate node parameters and compute the resulting product-Gaussian parameters
    def __init__(self, mu, sigma):
        # Validate mu and sigma, converting them to finite float arrays
        mu_array = to_finite_float_array(mu, "mu")
        sigma_array = to_finite_float_array(sigma, "sigma")

        # Initialize parent DGP class
        super().__init__(len(mu_array))

        self.parametric_form = "univariate_Gaussian"

        # Univariate-Gaussian-specific dimensionality & positivity Checks
        if mu_array.ndim != 1 or sigma_array.ndim != 1:
            raise ValueError("mu and sigma must be 1D vectors.")

        if mu_array.shape != sigma_array.shape:
            raise ValueError(
                f"Mismatch: mu length {len(mu_array)} != sigma length "
                f"{len(sigma_array)}"
            )

        if np.any(sigma_array <= 0):
            raise ValueError("Standard deviation (sigma) must be strictly positive.")

        # Compute parameters for the resulting Gaussian distribution
        # obtained by multiplying num_nodes univariate Gaussian distributions
        tau_array = 1.0 / sigma_array**2
        tau = np.sum(tau_array)

        sigma_product = np.sqrt(1.0 / tau)
        mu_product = np.sum(mu_array * tau_array) / tau

        # Assign validated parameters
        self.mu = mu_array
        self.sigma = sigma_array
        self.mu_product = mu_product
        self.sigma_product = sigma_product

    # Draw n samples from each of the num_nodes univariate Gaussian nodes
    def sample(self, n):
        # Validate n
        validate_positive_int(n, "sample size")

        # Sample from each univariate Gaussian node
        samples = np.random.normal(
            loc=self.mu,
            scale=self.sigma,
            size=(n, self.num_nodes)
        )
        # Add the p-dimension (where p=1)
        return samples[:, :, np.newaxis]

    # Draw n samples from the resulting product-Gaussian distribution
    def sample_product(self, n):
        return np.random.normal(loc=self.mu_product, scale=self.sigma_product, size=(n, 1))

    # Compute the closed-form E[X] under the product-Gaussian distribution
    def _closed_form_mean_product(self):
        return np.array([self.mu_product])

    # Compute the closed-form E[X^2] under the product-Gaussian distribution
    def _closed_form_square_mean_product(self):
        return np.array([self.mu_product ** 2 + self.sigma_product ** 2])

# DGP where each node is a bivariate Gaussian distribution
class BivariateGaussianDGP(GaussianDGP):
    # Validate node parameters and compute the resulting product-Gaussian parameters
    def __init__(self, mu, cov):
        num_nodes, p, mu_array, cov_array = self._validate_gaussian_dgp_inputs(mu, cov)

        # Bivariate-Gaussian-specific dimensionality check
        if p != 2:
            raise ValueError(
                f"BivariateGaussianDGP requires 2D inputs, got {p}D — use "
                "MultivariateGaussianDGP for other dimensions"
            )

        self._finish_gaussian_init(num_nodes, p, mu_array, cov_array, "bivariate_Gaussian")

# DGP where each node is a multivariate Gaussian distribution
class MultivariateGaussianDGP(GaussianDGP):
    # Validate node parameters and compute the resulting product-Gaussian parameters
    def __init__(self, mu, cov=None, cov_diag=None):
        if (cov is None) == (cov_diag is None):
            raise ValueError("Exactly one of cov and cov_diag must be provided.")

        if cov_diag is not None:
            mu_array = to_finite_float_array(mu, "mu")
            cov_diag_array = to_finite_float_array(cov_diag, "cov_diag")
            if mu_array.ndim != 2:
                raise ValueError(
                    f"mu must be 2D (num_nodes x p), got {mu_array.ndim}D"
                )
            if cov_diag_array.ndim != 2:
                raise ValueError(
                    "cov_diag must be 2D (num_nodes x p), got "
                    f"{cov_diag_array.ndim}D"
                )
            if mu_array.shape != cov_diag_array.shape:
                raise ValueError(
                    f"mu shape {mu_array.shape} does not match cov_diag shape "
                    f"{cov_diag_array.shape}"
                )
            if np.any(cov_diag_array <= 0):
                raise ValueError("Diagonal covariance entries must be strictly positive.")

            num_nodes, p = mu_array.shape
            DGP.__init__(self, num_nodes)
            precision_diag = 1.0 / cov_diag_array
            precision_product_diag = np.sum(precision_diag, axis=0)
            cov_product_diag = 1.0 / precision_product_diag
            mu_product = (
                np.sum(precision_diag * mu_array, axis=0)
                * cov_product_diag
            )

            self.parametric_form = "multivariate_Gaussian"
            self.p = p
            self.mu = mu_array
            self.cov = None
            self.cov_diag = cov_diag_array
            self.mu_product = mu_product
            self.cov_product = None
            self.cov_product_diag = cov_product_diag
            return

        num_nodes, p, mu_array, cov_array = self._validate_gaussian_dgp_inputs(mu, cov)

        self._finish_gaussian_init(num_nodes, p, mu_array, cov_array, "multivariate_Gaussian")

    # Draw n samples from each multivariate Gaussian node
    def sample(self, n):
        if not hasattr(self, "cov_diag"):
            return super().sample(n)
        validate_positive_int(n, "sample size")
        return np.random.normal(
            loc=self.mu,
            scale=np.sqrt(self.cov_diag),
            size=(n, self.num_nodes, self.p),
        )

    # Draw n samples from the multivariate product-Gaussian distribution
    def sample_product(self, n):
        if not hasattr(self, "cov_product_diag"):
            return super().sample_product(n)
        validate_positive_int(n, "sample size")
        return np.random.normal(
            loc=self.mu_product,
            scale=np.sqrt(self.cov_product_diag),
            size=(n, self.p),
        )

    # Compute the componentwise second moment of the product Gaussian
    def _closed_form_square_mean_product(self):
        if not hasattr(self, "cov_product_diag"):
            return super()._closed_form_square_mean_product()
        return np.array(self.mu_product) ** 2 + self.cov_product_diag

# DGP where each node is a Beta distribution
class BetaDGP(DGP):
    # Validate node parameters and compute the resulting product-Beta parameters
    def __init__(self, a, b):
        # Validate a and b, converting them to finite float arrays
        a_array = to_finite_float_array(a, "a")
        b_array = to_finite_float_array(b, "b")

        # Shape validation
        if a_array.ndim != 1 or b_array.ndim != 1:
            raise ValueError("a and b must be 1D vectors.")

        if a_array.shape != b_array.shape:
            raise ValueError(f"Mismatch: a length {len(a_array)} != b length {len(b_array)}")

        # Number of nodes is determined by the length of a and b
        num_nodes = len(a_array)

        # Initialize parent DGP class
        super().__init__(num_nodes)

        self.parametric_form = "Beta"

        # Beta-specific positivity checks
        if np.any(a_array <= 0):
            raise ValueError("Shape parameter a must be strictly positive.")

        if np.any(b_array <= 0):
            raise ValueError("Shape parameter b must be strictly positive.")

        # Compute parameters for the resulting Beta distribution obtained
        # by multiplying num_nodes Beta distributions
        a_product = np.sum(a_array) - (num_nodes - 1)
        b_product = np.sum(b_array) - (num_nodes - 1)

        if a_product <= 0 or b_product <= 0:
            raise ValueError(
                "Resulting shape parameters for the product Beta "
                "distribution must be strictly positive."
            )

        # Assign validated parameters
        self.a = a_array
        self.b = b_array
        self.a_product = a_product
        self.b_product = b_product

    # Draw n samples from each of the num_nodes Beta nodes
    def sample(self, n):
        # Validate n
        validate_positive_int(n, "sample size")

        # Sample from each Beta node
        samples = np.random.beta(
            a=self.a,
            b=self.b,
            size=(n, self.num_nodes)
        )

        # Add the p-dimension (where p=1)
        return samples[:, :, np.newaxis]

    # Draw n samples from the resulting product-Beta distribution
    def sample_product(self, n):
        return np.random.beta(a=self.a_product, b=self.b_product, size=(n, 1))

    # Compute the closed-form E[X] under the product-Beta distribution
    def _closed_form_mean_product(self):
        return np.array([self.a_product / (self.a_product + self.b_product)])

    # Compute the closed-form E[X^2] under the product-Beta distribution
    def _closed_form_square_mean_product(self):
        total = self.a_product + self.b_product
        second_moment = self.a_product * (self.a_product + 1) / (
            total * (total + 1)
        )
        return np.array([second_moment])
