"""Active learning module for intelligent query selection."""

import numpy as np
from typing import Any, List, Optional, Union, Dict, Callable, Tuple
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.stats import entropy
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class ActiveLearner:
    """
    Implements active learning strategies for query selection.

    Supports uncertainty-based, diversity-based, and combined strategies
    as described in the paper.
    """

    def __init__(
        self,
        strategy: str = "combined",
        uncertainty_weight: float = 0.5,
        batch_size: int = 32,
        top_k_ratio: float = 0.1,
        distance_metric: str = "euclidean",
        cache_distances: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize active learner.

        Args:
            strategy: Selection strategy ('uncertainty', 'diversity', 'combined')
            uncertainty_weight: Weight for uncertainty in combined strategy
            batch_size: Number of queries to select per iteration
            top_k_ratio: Ratio of candidate pool to consider
            distance_metric: Distance metric for diversity calculation
            cache_distances: Whether to cache distance computations
            random_state: Random seed for reproducibility
        """
        self.strategy = strategy
        self.uncertainty_weight = uncertainty_weight
        self.batch_size = batch_size
        self.top_k_ratio = top_k_ratio
        self.distance_metric = distance_metric
        self.cache_distances = cache_distances
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self.distance_cache: Dict[Tuple, np.ndarray] = {} if cache_distances else None
        self.selection_history: List[Dict[str, Any]] = []

    def select_queries(
        self,
        model: BaseEstimator,
        candidate_pool: np.ndarray,
        existing_data: Optional[np.ndarray] = None,
        custom_scorer: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Select queries from candidate pool.

        Args:
            model: Current surrogate model
            candidate_pool: Pool of candidate instances
            existing_data: Existing training data
            custom_scorer: Optional custom scoring function

        Returns:
            Selected query instances
        """
        n_candidates = len(candidate_pool)

        if n_candidates <= self.batch_size:
            return candidate_pool

        logger.debug(f"Selecting {self.batch_size} queries from {n_candidates} candidates")

        # Compute scores based on strategy
        if self.strategy == "uncertainty":
            scores = self._compute_uncertainty_scores(model, candidate_pool, custom_scorer)
        elif self.strategy == "diversity":
            scores = self._compute_diversity_scores(candidate_pool, existing_data)
        elif self.strategy == "combined":
            uncertainty_scores = self._compute_uncertainty_scores(model, candidate_pool, custom_scorer)
            diversity_scores = self._compute_diversity_scores(candidate_pool, existing_data)

            # Normalize scores
            uncertainty_scores = self._normalize_scores(uncertainty_scores)
            diversity_scores = self._normalize_scores(diversity_scores)

            # Combine scores
            scores = (self.uncertainty_weight * uncertainty_scores +
                     (1 - self.uncertainty_weight) * diversity_scores)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Select top candidates
        selected_indices = self._select_batch(scores, candidate_pool)

        # Log selection stats
        self.selection_history.append({
            'strategy': self.strategy,
            'n_candidates': n_candidates,
            'n_selected': len(selected_indices),
            'mean_score': float(np.mean(scores)),
            'max_score': float(np.max(scores)),
            'min_score': float(np.min(scores))
        })

        return candidate_pool[selected_indices]

    def _compute_uncertainty_scores(
        self,
        model: BaseEstimator,
        candidate_pool: np.ndarray,
        custom_scorer: Optional[Callable] = None
    ) -> np.ndarray:
        """Compute uncertainty scores for candidates."""
        if custom_scorer is not None:
            return custom_scorer(model, candidate_pool)

        if not hasattr(model, 'predict_proba'):
            logger.warning("Model doesn't support predict_proba, using random scores")
            return np.random.rand(len(candidate_pool))

        # Get prediction probabilities
        try:
            probas = model.predict_proba(candidate_pool)
        except Exception as e:
            logger.warning(f"Error computing probabilities: {e}")
            return np.random.rand(len(candidate_pool))

        # Compute uncertainty metrics
        scores = self._compute_entropy(probas)

        # Alternative: least confidence
        # scores = 1 - np.max(probas, axis=1)

        # Alternative: margin sampling
        # sorted_probas = np.sort(probas, axis=1)
        # scores = 1 - (sorted_probas[:, -1] - sorted_probas[:, -2])

        return scores

    def _compute_entropy(self, probas: np.ndarray) -> np.ndarray:
        """Compute entropy-based uncertainty."""
        eps = 1e-10
        probas = np.clip(probas, eps, 1 - eps)

        entropies = np.array([entropy(proba) for proba in probas])

        return entropies

    def _compute_diversity_scores(
        self,
        candidate_pool: np.ndarray,
        existing_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute diversity scores for candidates."""
        scores = np.zeros(len(candidate_pool))

        if existing_data is not None and len(existing_data) > 0:
            # Compute distance to existing data
            distances = self._compute_distances(candidate_pool, existing_data)

            # Use minimum distance to existing data as diversity score
            scores = np.min(distances, axis=1)
        else:
            # Compute pairwise distances within candidate pool
            distances = self._compute_distances(candidate_pool, candidate_pool)

            # Use average distance to other candidates
            np.fill_diagonal(distances, np.inf)
            scores = np.mean(np.where(distances == np.inf, 0, distances), axis=1)

        return scores

    def _compute_distances(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise distances with caching."""
        # Create cache key
        cache_key = (id(X), id(Y), self.distance_metric)

        if self.cache_distances and cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        # Compute distances
        if self.distance_metric == "euclidean":
            distances = euclidean_distances(X, Y)
        elif self.distance_metric == "cosine":
            distances = cosine_distances(X, Y)
        elif self.distance_metric == "manhattan":
            distances = cdist(X, Y, metric='cityblock')
        else:
            distances = cdist(X, Y, metric=self.distance_metric)

        # Cache result
        if self.cache_distances:
            self.distance_cache[cache_key] = distances

        return distances

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_score = np.min(scores)
        max_score = np.max(scores)

        if max_score - min_score < 1e-10:
            return np.ones_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def _select_batch(
        self,
        scores: np.ndarray,
        candidate_pool: np.ndarray
    ) -> np.ndarray:
        """Select batch of queries based on scores."""
        # Filter to top-k candidates
        n_top_k = max(int(len(scores) * self.top_k_ratio), self.batch_size)
        top_k_indices = np.argsort(scores)[-n_top_k:]

        if len(top_k_indices) <= self.batch_size:
            return top_k_indices

        # Select diverse batch from top-k
        selected_indices = []
        remaining_indices = list(top_k_indices)

        # Select instance with highest score
        best_idx = remaining_indices[np.argmax(scores[remaining_indices])]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        # Iteratively select diverse instances
        while len(selected_indices) < self.batch_size and remaining_indices:
            # Compute distances to selected instances
            selected_pool = candidate_pool[selected_indices]
            remaining_pool = candidate_pool[remaining_indices]

            distances = self._compute_distances(remaining_pool, selected_pool)
            min_distances = np.min(distances, axis=1)

            # Weight by original scores
            remaining_scores = scores[remaining_indices]
            combined_scores = remaining_scores * min_distances

            # Select instance with highest combined score
            best_idx = remaining_indices[np.argmax(combined_scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return np.array(selected_indices)

    def suggest_next_batch_size(
        self,
        current_performance: float,
        target_performance: float,
        queries_used: int,
        query_budget: int
    ) -> int:
        """
        Suggest batch size for next iteration based on performance.

        Args:
            current_performance: Current model performance
            target_performance: Target performance
            queries_used: Queries used so far
            query_budget: Total query budget

        Returns:
            Suggested batch size
        """
        remaining_budget = query_budget - queries_used

        if remaining_budget <= 0:
            return 0

        # Calculate performance gap
        performance_gap = target_performance - current_performance

        if performance_gap <= 0:
            return 0

        # Adaptive batch size based on performance gap
        if performance_gap > 0.2:
            # Large gap - use larger batches
            suggested_size = min(self.batch_size * 2, remaining_budget)
        elif performance_gap > 0.1:
            # Medium gap - use normal batches
            suggested_size = min(self.batch_size, remaining_budget)
        else:
            # Small gap - use smaller batches
            suggested_size = min(self.batch_size // 2, remaining_budget)

        return max(1, suggested_size)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about query selection."""
        if not self.selection_history:
            return {}

        stats = {
            'total_selections': len(self.selection_history),
            'total_queries_selected': sum(h['n_selected'] for h in self.selection_history),
            'avg_candidates_per_selection': np.mean([h['n_candidates'] for h in self.selection_history]),
            'avg_score': np.mean([h['mean_score'] for h in self.selection_history]),
            'strategy_used': self.strategy
        }

        if self.cache_distances and self.distance_cache:
            stats['distance_cache_size'] = len(self.distance_cache)

        return stats

    def reset(self) -> None:
        """Reset the active learner state."""
        self.selection_history.clear()

        if self.cache_distances:
            self.distance_cache.clear()


class UncertaintySampler(ActiveLearner):
    """Specialized sampler for uncertainty-based selection."""

    def __init__(self, method: str = "entropy", **kwargs):
        """
        Initialize uncertainty sampler.

        Args:
            method: Uncertainty method ('entropy', 'least_confidence', 'margin')
            **kwargs: Additional arguments for ActiveLearner
        """
        super().__init__(strategy="uncertainty", **kwargs)
        self.method = method

    def _compute_uncertainty_scores(
        self,
        model: BaseEstimator,
        candidate_pool: np.ndarray,
        custom_scorer: Optional[Callable] = None
    ) -> np.ndarray:
        """Compute uncertainty scores based on specified method."""
        if custom_scorer is not None:
            return custom_scorer(model, candidate_pool)

        if not hasattr(model, 'predict_proba'):
            return np.random.rand(len(candidate_pool))

        probas = model.predict_proba(candidate_pool)

        if self.method == "entropy":
            return self._compute_entropy(probas)
        elif self.method == "least_confidence":
            return 1 - np.max(probas, axis=1)
        elif self.method == "margin":
            sorted_probas = np.sort(probas, axis=1)
            return 1 - (sorted_probas[:, -1] - sorted_probas[:, -2])
        else:
            raise ValueError(f"Unknown uncertainty method: {self.method}")


class DiversitySampler(ActiveLearner):
    """Specialized sampler for diversity-based selection."""

    def __init__(self, method: str = "max_min", **kwargs):
        """
        Initialize diversity sampler.

        Args:
            method: Diversity method ('max_min', 'k_means', 'density')
            **kwargs: Additional arguments for ActiveLearner
        """
        super().__init__(strategy="diversity", **kwargs)
        self.method = method

    def _select_batch(
        self,
        scores: np.ndarray,
        candidate_pool: np.ndarray
    ) -> np.ndarray:
        """Select diverse batch based on specified method."""
        if self.method == "max_min":
            return self._max_min_selection(scores, candidate_pool)
        elif self.method == "k_means":
            return self._k_means_selection(candidate_pool)
        elif self.method == "density":
            return self._density_based_selection(scores, candidate_pool)
        else:
            return super()._select_batch(scores, candidate_pool)

    def _max_min_selection(
        self,
        scores: np.ndarray,
        candidate_pool: np.ndarray
    ) -> np.ndarray:
        """Select using max-min diversity criterion."""
        selected_indices = []

        # Start with highest scoring instance
        best_idx = np.argmax(scores)
        selected_indices.append(best_idx)

        # Iteratively select most distant instances
        for _ in range(self.batch_size - 1):
            remaining_mask = np.ones(len(candidate_pool), dtype=bool)
            remaining_mask[selected_indices] = False

            if not np.any(remaining_mask):
                break

            selected_pool = candidate_pool[selected_indices]
            remaining_pool = candidate_pool[remaining_mask]

            distances = self._compute_distances(remaining_pool, selected_pool)
            min_distances = np.min(distances, axis=1)

            remaining_indices = np.where(remaining_mask)[0]
            best_idx = remaining_indices[np.argmax(min_distances)]
            selected_indices.append(best_idx)

        return np.array(selected_indices)

    def _k_means_selection(self, candidate_pool: np.ndarray) -> np.ndarray:
        """Select using k-means clustering."""
        from sklearn.cluster import KMeans

        # Cluster candidates
        kmeans = KMeans(n_clusters=min(self.batch_size, len(candidate_pool)),
                       random_state=self.random_state)
        labels = kmeans.fit_predict(candidate_pool)

        # Select closest instance to each cluster center
        selected_indices = []

        for i in range(kmeans.n_clusters):
            cluster_mask = labels == i
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) > 0:
                cluster_points = candidate_pool[cluster_mask]
                center = kmeans.cluster_centers_[i]

                distances = np.linalg.norm(cluster_points - center, axis=1)
                best_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(best_idx)

        return np.array(selected_indices)

    def _density_based_selection(
        self,
        scores: np.ndarray,
        candidate_pool: np.ndarray
    ) -> np.ndarray:
        """Select based on density in feature space."""
        # Compute density for each point
        distances = self._compute_distances(candidate_pool, candidate_pool)
        np.fill_diagonal(distances, np.inf)

        # Use k-nearest neighbors for density estimation
        k = min(10, len(candidate_pool) - 1)
        knn_distances = np.sort(distances, axis=1)[:, :k]
        densities = 1.0 / (np.mean(knn_distances, axis=1) + 1e-10)

        # Combine with original scores
        combined_scores = scores * (1.0 / densities)  # Prefer low-density regions

        # Select top instances
        selected_indices = np.argsort(combined_scores)[-self.batch_size:]

        return selected_indices


__all__ = [
    'ActiveLearner',
    'UncertaintySampler',
    'DiversitySampler'
]