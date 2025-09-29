"""Combinatorial testing module for generating test suites with t-way coverage."""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any, Set
from itertools import combinations, product
import logging
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class CombinatorialTester:
    """
    Generates test suites with t-way interaction coverage.

    Implements the Input Parameter Model (IPM) approach from the paper
    for systematic test case generation.
    """

    def __init__(
        self,
        t_way: int = 3,
        discretization_method: str = "uniform",
        n_bins: int = 5,
        coverage: float = 1.0,
        cache_enabled: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize combinatorial tester.

        Args:
            t_way: T-way interaction strength
            discretization_method: Method for discretizing continuous features
            n_bins: Number of bins for discretization
            coverage: Coverage level (1.0 for 100%)
            cache_enabled: Whether to cache computations
            random_state: Random seed for reproducibility
        """
        self.t_way = t_way
        self.discretization_method = discretization_method
        self.n_bins = n_bins
        self.coverage = coverage
        self.cache_enabled = cache_enabled
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self.discretizers: Dict[int, KBinsDiscretizer] = {}
        self.feature_domains: List[List[Any]] = []
        self.interaction_cache: Dict[Tuple, Set] = {} if cache_enabled else None

    def generate_test_suite(
        self,
        feature_ranges: List[Union[Tuple[float, float], List[Any]]],
        categorical_features: Optional[List[int]] = None,
        constraints: Optional[List[Callable]] = None,
        existing_tests: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate test suite with t-way coverage.

        Args:
            feature_ranges: Range specifications for each feature
            categorical_features: Indices of categorical features
            constraints: Optional constraints on valid combinations
            existing_tests: Existing test cases to extend

        Returns:
            Array of test instances
        """
        n_features = len(feature_ranges)
        categorical_features = categorical_features or []

        logger.info(f"Generating {self.t_way}-way test suite for {n_features} features")

        # Discretize continuous features
        self._prepare_feature_domains(feature_ranges, categorical_features)

        # Generate all t-way combinations
        all_combinations = self._generate_t_way_combinations(n_features)

        # Generate covering array
        test_suite = self._generate_covering_array(
            all_combinations,
            n_features,
            constraints,
            existing_tests
        )

        # Convert back to continuous values
        continuous_suite = self._convert_to_continuous(
            test_suite,
            feature_ranges,
            categorical_features
        )

        logger.info(f"Generated {len(continuous_suite)} test cases")

        return continuous_suite

    def _prepare_feature_domains(
        self,
        feature_ranges: List[Union[Tuple, List]],
        categorical_features: List[int]
    ) -> None:
        """Prepare discretized domains for each feature."""
        self.feature_domains = []

        for i, range_spec in enumerate(feature_ranges):
            if i in categorical_features:
                # Categorical feature
                if isinstance(range_spec, tuple):
                    # Convert range to list of values
                    domain = list(range(int(range_spec[0]), int(range_spec[1]) + 1))
                else:
                    domain = list(range_spec)
            else:
                # Continuous feature - discretize
                if isinstance(range_spec, tuple):
                    domain = self._discretize_continuous_range(
                        range_spec[0],
                        range_spec[1],
                        i
                    )
                else:
                    # Already discrete values
                    domain = list(range_spec)

            self.feature_domains.append(domain)

    def _discretize_continuous_range(
        self,
        min_val: float,
        max_val: float,
        feature_idx: int
    ) -> List[float]:
        """Discretize a continuous range into bins."""
        if self.discretization_method == "uniform":
            # Uniform random sampling
            return list(np.linspace(min_val, max_val, self.n_bins))

        elif self.discretization_method == "equal_width":
            # Equal width bins
            return list(np.linspace(min_val, max_val, self.n_bins))

        elif self.discretization_method == "equal_freq":
            # Equal frequency bins (percentiles)
            percentiles = np.linspace(0, 100, self.n_bins)
            # Generate sample data for percentile calculation
            sample = np.random.uniform(min_val, max_val, 10000)
            return list(np.percentile(sample, percentiles))

        elif self.discretization_method == "entropy":
            # Entropy-based discretization
            return self._entropy_based_discretization(min_val, max_val)

        else:
            raise ValueError(f"Unknown discretization method: {self.discretization_method}")

    def _entropy_based_discretization(
        self,
        min_val: float,
        max_val: float
    ) -> List[float]:
        """Discretize using entropy-based method."""
        # Generate sample data
        sample = np.random.uniform(min_val, max_val, 10000)

        best_bins = None
        best_entropy = -np.inf

        for n_bins in range(2, self.n_bins + 1):
            bins = np.linspace(min_val, max_val, n_bins)
            hist, _ = np.histogram(sample, bins=bins)

            if hist.sum() > 0:
                probs = hist / hist.sum()
                ent = entropy(probs)

                if ent > best_entropy:
                    best_entropy = ent
                    best_bins = bins

        return list(best_bins) if best_bins is not None else list(np.linspace(min_val, max_val, self.n_bins))

    def _generate_t_way_combinations(self, n_features: int) -> List[Tuple]:
        """Generate all t-way feature combinations."""
        all_combinations = []

        # Get all t-way feature index combinations
        for feature_indices in combinations(range(n_features), self.t_way):
            # Get all value combinations for these features
            value_combinations = list(product(*[
                self.feature_domains[i] for i in feature_indices
            ]))

            for values in value_combinations:
                combination = (feature_indices, values)
                all_combinations.append(combination)

        logger.debug(f"Generated {len(all_combinations)} {self.t_way}-way combinations")

        return all_combinations

    def _generate_covering_array(
        self,
        all_combinations: List[Tuple],
        n_features: int,
        constraints: Optional[List[Callable]],
        existing_tests: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Generate covering array using greedy algorithm.

        This implements the IPOG-style algorithm for covering array construction.
        """
        test_suite = []
        covered_combinations = set()

        # Add existing tests if provided
        if existing_tests is not None:
            for test in existing_tests:
                test_suite.append(test)
                self._update_covered_combinations(
                    test,
                    all_combinations,
                    covered_combinations
                )

        # Calculate target coverage
        target_coverage = int(len(all_combinations) * self.coverage)

        # Greedy construction
        while len(covered_combinations) < target_coverage:
            best_test = None
            best_coverage_count = 0

            # Generate candidate tests
            for _ in range(100):  # Generate multiple candidates
                candidate = self._generate_candidate_test(n_features)

                # Check constraints
                if constraints and not self._satisfies_constraints(candidate, constraints):
                    continue

                # Count newly covered combinations
                new_coverage = self._count_new_coverage(
                    candidate,
                    all_combinations,
                    covered_combinations
                )

                if new_coverage > best_coverage_count:
                    best_coverage_count = new_coverage
                    best_test = candidate

            if best_test is not None:
                test_suite.append(best_test)
                self._update_covered_combinations(
                    best_test,
                    all_combinations,
                    covered_combinations
                )

                if len(test_suite) % 10 == 0:
                    current_coverage = len(covered_combinations) / len(all_combinations)
                    logger.debug(f"Test suite size: {len(test_suite)}, "
                               f"Coverage: {current_coverage:.2%}")
            else:
                # No valid test found, use random
                test_suite.append(self._generate_candidate_test(n_features))

            # Early termination if we've tried too many times
            if len(test_suite) > 10 * n_features:
                logger.warning("Covering array generation taking too long, terminating early")
                break

        return np.array(test_suite)

    def _generate_candidate_test(self, n_features: int) -> np.ndarray:
        """Generate a candidate test case."""
        test = np.zeros(n_features)

        for i in range(n_features):
            test[i] = np.random.choice(self.feature_domains[i])

        return test

    def _satisfies_constraints(
        self,
        test: np.ndarray,
        constraints: List[Callable]
    ) -> bool:
        """Check if test satisfies all constraints."""
        for constraint in constraints:
            if not constraint(test):
                return False
        return True

    def _count_new_coverage(
        self,
        test: np.ndarray,
        all_combinations: List[Tuple],
        covered_combinations: Set[Tuple]
    ) -> int:
        """Count how many new combinations this test covers."""
        count = 0

        for feature_indices, values in all_combinations:
            # Check if this combination is covered by the test
            test_values = tuple(test[list(feature_indices)])

            # Check if values match (accounting for floating point)
            if self._values_match(test_values, values):
                combo_key = (feature_indices, values)
                if combo_key not in covered_combinations:
                    count += 1

        return count

    def _update_covered_combinations(
        self,
        test: np.ndarray,
        all_combinations: List[Tuple],
        covered_combinations: Set[Tuple]
    ) -> None:
        """Update the set of covered combinations."""
        for feature_indices, values in all_combinations:
            test_values = tuple(test[list(feature_indices)])

            if self._values_match(test_values, values):
                combo_key = (feature_indices, values)
                covered_combinations.add(combo_key)

    def _values_match(self, test_values: Tuple, target_values: Tuple) -> bool:
        """Check if test values match target values."""
        if len(test_values) != len(target_values):
            return False

        for tv, tg in zip(test_values, target_values):
            # Use approximate matching for floats
            if isinstance(tv, float) and isinstance(tg, float):
                if not np.isclose(tv, tg, rtol=1e-5, atol=1e-8):
                    return False
            else:
                if tv != tg:
                    return False

        return True

    def _convert_to_continuous(
        self,
        test_suite: np.ndarray,
        feature_ranges: List[Union[Tuple, List]],
        categorical_features: List[int]
    ) -> np.ndarray:
        """Convert discretized test suite back to continuous values."""
        continuous_suite = np.zeros_like(test_suite)

        for i in range(test_suite.shape[1]):
            if i in categorical_features:
                # Keep categorical as-is
                continuous_suite[:, i] = test_suite[:, i]
            else:
                # Add noise to discretized values for continuous features
                if isinstance(feature_ranges[i], tuple):
                    min_val, max_val = feature_ranges[i]
                    bin_width = (max_val - min_val) / (self.n_bins - 1)

                    # Add uniform noise within bin
                    noise = np.random.uniform(-bin_width/2, bin_width/2, len(test_suite))
                    continuous_suite[:, i] = np.clip(
                        test_suite[:, i] + noise,
                        min_val,
                        max_val
                    )
                else:
                    continuous_suite[:, i] = test_suite[:, i]

        return continuous_suite

    def analyze_coverage(
        self,
        test_suite: np.ndarray,
        feature_ranges: List[Union[Tuple, List]],
        categorical_features: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the t-way coverage of a test suite.

        Args:
            test_suite: Test cases to analyze
            feature_ranges: Feature range specifications
            categorical_features: Indices of categorical features

        Returns:
            Dictionary with coverage analysis
        """
        n_features = test_suite.shape[1]
        categorical_features = categorical_features or []

        # Prepare domains
        self._prepare_feature_domains(feature_ranges, categorical_features)

        # Generate all combinations
        all_combinations = self._generate_t_way_combinations(n_features)

        # Count covered combinations
        covered_combinations = set()

        for test in test_suite:
            self._update_covered_combinations(
                test,
                all_combinations,
                covered_combinations
            )

        coverage_ratio = len(covered_combinations) / len(all_combinations)

        analysis = {
            'total_combinations': len(all_combinations),
            'covered_combinations': len(covered_combinations),
            'coverage_ratio': coverage_ratio,
            'test_suite_size': len(test_suite),
            't_way': self.t_way,
            'missing_combinations': len(all_combinations) - len(covered_combinations)
        }

        # Find most/least covered features
        feature_coverage = np.zeros(n_features)

        for (feature_indices, _), _ in covered_combinations:
            for idx in feature_indices:
                feature_coverage[idx] += 1

        analysis['feature_coverage'] = {
            f'feature_{i}': int(count)
            for i, count in enumerate(feature_coverage)
        }

        return analysis

    def extend_test_suite(
        self,
        existing_suite: np.ndarray,
        feature_ranges: List[Union[Tuple, List]],
        target_coverage: float = 1.0,
        categorical_features: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Extend existing test suite to achieve target coverage.

        Args:
            existing_suite: Current test suite
            feature_ranges: Feature range specifications
            target_coverage: Target coverage level
            categorical_features: Indices of categorical features

        Returns:
            Extended test suite
        """
        self.coverage = target_coverage

        return self.generate_test_suite(
            feature_ranges=feature_ranges,
            categorical_features=categorical_features,
            existing_tests=existing_suite
        )

    def minimize_test_suite(
        self,
        test_suite: np.ndarray,
        feature_ranges: List[Union[Tuple, List]],
        min_coverage: float = 0.95,
        categorical_features: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Minimize test suite while maintaining coverage.

        Args:
            test_suite: Original test suite
            feature_ranges: Feature range specifications
            min_coverage: Minimum acceptable coverage
            categorical_features: Indices of categorical features

        Returns:
            Minimized test suite
        """
        n_features = test_suite.shape[1]
        categorical_features = categorical_features or []

        # Prepare domains
        self._prepare_feature_domains(feature_ranges, categorical_features)

        # Generate all combinations
        all_combinations = self._generate_t_way_combinations(n_features)

        # Greedy minimization
        minimized_suite = []
        covered_combinations = set()

        # Sort tests by their coverage contribution
        test_contributions = []

        for i, test in enumerate(test_suite):
            contribution = self._count_new_coverage(
                test,
                all_combinations,
                set()
            )
            test_contributions.append((contribution, i))

        test_contributions.sort(reverse=True)

        # Add tests greedily until target coverage
        target_combinations = int(len(all_combinations) * min_coverage)

        for _, idx in test_contributions:
            test = test_suite[idx]
            new_coverage = self._count_new_coverage(
                test,
                all_combinations,
                covered_combinations
            )

            if new_coverage > 0:
                minimized_suite.append(test)
                self._update_covered_combinations(
                    test,
                    all_combinations,
                    covered_combinations
                )

                if len(covered_combinations) >= target_combinations:
                    break

        minimized_array = np.array(minimized_suite)

        logger.info(f"Minimized test suite from {len(test_suite)} to "
                   f"{len(minimized_array)} tests")

        return minimized_array


__all__ = ['CombinatorialTester']