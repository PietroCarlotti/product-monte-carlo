##############################
# Import necessary libraries #
##############################

import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

########################
# Exact product oracle #
########################

DEFAULT_MAX_EXPANDED_COMPONENTS = 100_000

@dataclass(frozen=True)
class ProductGaussianMixture:
    weights: np.ndarray
    means: np.ndarray
    standard_deviations: np.ndarray
    log_normalizer: float
    normalizer: float | None

    # Evaluate the normalized product-mixture density.
    def density(self, values):
        values = np.asarray(values, dtype=float)
        scaled = (
            values[..., np.newaxis] - self.means
        ) / self.standard_deviations
        component_densities = np.exp(-0.5 * scaled**2)
        component_densities /= np.sqrt(2.0 * np.pi) * self.standard_deviations
        return np.sum(self.weights * component_densities, axis=-1)

    # Draw exact samples from the normalized product mixture.
    def sample(self, sample_count, seed):
        if (
            isinstance(sample_count, (bool, np.bool_))
            or not isinstance(sample_count, (int, np.integer))
            or sample_count <= 0
        ):
            raise ValueError("sample_count must be a positive integer.")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed,
            (int, np.integer),
        ):
            raise ValueError("seed must be an integer.")
        if seed < 0:
            raise ValueError("seed must be a nonnegative integer.")

        rng = np.random.default_rng(seed)
        component_indices = rng.choice(
            len(self.weights),
            size=sample_count,
            p=self.weights,
        )
        return rng.normal(
            self.means[component_indices],
            self.standard_deviations[component_indices],
        )

    # Compute the exact product-mixture mean.
    def mean(self):
        return float(np.sum(self.weights * self.means))

    # Compute the exact product-mixture second moment.
    def second_moment(self):
        component_moments = self.standard_deviations**2 + self.means**2
        return float(np.sum(self.weights * component_moments))


# Validate and convert one component-density specification.
def _parse_node(node, node_index):
    required_keys = {"weights", "means", "standard_deviations"}
    if set(node) != required_keys:
        raise ValueError(
            f"Node {node_index} must contain exactly {sorted(required_keys)}."
        )

    weights = np.asarray(node["weights"], dtype=float)
    means = np.asarray(node["means"], dtype=float)
    standard_deviations = np.asarray(
        node["standard_deviations"],
        dtype=float,
    )
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError(f"Node {node_index} weights must be a nonempty vector.")
    if means.shape != weights.shape or standard_deviations.shape != weights.shape:
        raise ValueError(f"Node {node_index} mixture parameters must have equal lengths.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError(f"Node {node_index} weights must be finite and positive.")
    if not np.all(np.isfinite(means)):
        raise ValueError(f"Node {node_index} means must be finite.")
    if (
        not np.all(np.isfinite(standard_deviations))
        or np.any(standard_deviations <= 0.0)
    ):
        raise ValueError(
            f"Node {node_index} standard deviations must be finite and positive."
        )
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError(f"Node {node_index} weights must sum to one.")

    return weights, means, standard_deviations


# Expand node mixtures unless their Cartesian product exceeds the safety cap.
def build_exact_product(
    nodes,
    max_expanded_components=DEFAULT_MAX_EXPANDED_COMPONENTS,
):
    if len(nodes) == 0:
        raise ValueError("At least one node is required.")
    if (
        isinstance(max_expanded_components, (bool, np.bool_))
        or not isinstance(max_expanded_components, (int, np.integer))
        or max_expanded_components <= 0
    ):
        raise ValueError("max_expanded_components must be a positive integer.")

    parsed_nodes = [_parse_node(node, index) for index, node in enumerate(nodes)]
    expanded_component_count = math.prod(len(node[0]) for node in parsed_nodes)
    if expanded_component_count > max_expanded_components:
        raise ValueError(
            "Exact mixture expansion would create "
            f"{expanded_component_count} components, exceeding the "
            f"max_expanded_components cap of {max_expanded_components}."
        )
    combinations = product(*[range(len(node[0])) for node in parsed_nodes])
    log_weights = []
    product_means = []
    product_standard_deviations = []

    for combination in combinations:
        weights = np.array(
            [parsed_nodes[node][0][component] for node, component in enumerate(combination)]
        )
        means = np.array(
            [parsed_nodes[node][1][component] for node, component in enumerate(combination)]
        )
        standard_deviations = np.array(
            [parsed_nodes[node][2][component] for node, component in enumerate(combination)]
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            precisions = 1.0 / standard_deviations**2
        if np.any(~np.isfinite(precisions)) or np.any(precisions <= 0.0):
            raise ValueError("Derived component precisions must be finite and positive.")
        precision = np.sum(precisions)
        if not np.isfinite(precision) or precision <= 0.0:
            raise ValueError("Derived product precision must be finite and positive.")
        product_mean = np.sum(precisions * means) / precision
        product_standard_deviation = np.sqrt(1.0 / precision)
        quadratic_term = np.sum(precisions * (means - product_mean) ** 2)
        if (
            not np.isfinite(product_mean)
            or not np.isfinite(product_standard_deviation)
            or product_standard_deviation <= 0.0
            or not np.isfinite(quadratic_term)
            or quadratic_term < 0.0
        ):
            raise ValueError("Derived product component parameters are invalid.")
        log_integral = (
            -0.5 * (len(nodes) - 1) * np.log(2.0 * np.pi)
            - np.sum(np.log(standard_deviations))
            - 0.5 * np.log(precision)
            - 0.5 * quadratic_term
        )
        if not np.isfinite(log_integral):
            raise ValueError("Derived Gaussian overlap log-integral is invalid.")

        log_weights.append(np.sum(np.log(weights)) + log_integral)
        product_means.append(product_mean)
        product_standard_deviations.append(product_standard_deviation)

    log_weights = np.asarray(log_weights)
    log_normalizer = float(logsumexp(log_weights))
    if not math.isfinite(log_normalizer):
        raise ValueError("The exact product log-normalizer must be finite.")
    log_smallest = math.log(np.nextafter(0.0, 1.0))
    log_largest = math.log(np.finfo(float).max)
    normalizer = (
        math.exp(log_normalizer)
        if log_smallest <= log_normalizer <= log_largest
        else None
    )
    normalized_weights = np.exp(log_weights - log_normalizer)
    if (
        np.any(~np.isfinite(normalized_weights))
        or np.any(normalized_weights <= 0.0)
        or not np.isclose(
            np.sum(normalized_weights),
            1.0,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("The exact product weights are invalid.")
    product_means = np.asarray(product_means)
    product_standard_deviations = np.asarray(product_standard_deviations)
    component_second_moments = (
        product_standard_deviations**2 + product_means**2
    )
    if (
        np.any(~np.isfinite(product_means))
        or np.any(~np.isfinite(product_standard_deviations))
        or np.any(~np.isfinite(component_second_moments))
    ):
        raise ValueError("The exact product moments must be finite.")
    return ProductGaussianMixture(
        weights=normalized_weights,
        means=product_means,
        standard_deviations=product_standard_deviations,
        log_normalizer=log_normalizer,
        normalizer=normalizer,
    )


# Load and validate an exact-oracle JSON configuration.
def load_config(config_path):
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    required_keys = {
        "num_nodes",
        "oracle_sample_count",
        "component_sample_count",
        "seed",
        "pmc",
        "nodes",
    }
    if set(config) != required_keys:
        raise ValueError(f"Configuration must contain exactly {sorted(required_keys)}.")
    if config["num_nodes"] != len(config["nodes"]):
        raise ValueError("num_nodes must match the number of node specifications.")
    if not isinstance(config["oracle_sample_count"], int) or config[
        "oracle_sample_count"
    ] <= 0:
        raise ValueError("oracle_sample_count must be a positive integer.")
    if not isinstance(config["component_sample_count"], int) or config[
        "component_sample_count"
    ] <= 0:
        raise ValueError("component_sample_count must be a positive integer.")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer.")

    build_exact_product(config["nodes"])
    return config
