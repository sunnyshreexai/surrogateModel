"""Counterfactual generation module for creating opposite-class examples."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from sklearn.base import BaseEstimator
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


class CounterfactualGenerator:
    """
    Generates counterfactual explanations for model predictions.

    Implements multiple methods for generating opposite-class examples
    as described in the paper.
    """

    def __init__(
        self,
        method: str = "dice",
        diversity: float = 0.5,
        n_counterfactuals: int = 1,
        max_iterations: int = 100,
        distance_weight: float = 0.5,
        feature_weights: Optional[np.ndarray] = None,
        immutable_features: Optional[List[int]] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize counterfactual generator.

        Args:
            method: Generation method ('dice', 'genetic', 'gradient', 'prototype')
            diversity: Diversity parameter for multiple counterfactuals
            n_counterfactuals: Number of counterfactuals to generate per instance
            max_iterations: Maximum optimization iterations
            distance_weight: Weight for distance in objective function
            feature_weights: Importance weights for features
            immutable_features: Indices of features that cannot be changed
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.diversity = diversity
        self.n_counterfactuals = n_counterfactuals
        self.max_iterations = max_iterations
        self.distance_weight = distance_weight
        self.feature_weights = feature_weights
        self.immutable_features = immutable_features or []
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self.generation_history: List[Dict[str, Any]] = []

    def generate(
        self,
        instance: np.ndarray,
        target_class: int,
        model: BaseEstimator,
        feature_ranges: List[Union[Tuple[float, float], List[Any]]],
        categorical_features: Optional[List[int]] = None
    ) -> Optional[np.ndarray]:
        """
        Generate counterfactual for given instance.

        Args:
            instance: Original instance
            target_class: Target class for counterfactual
            model: Surrogate model
            feature_ranges: Valid ranges for features
            categorical_features: Indices of categorical features

        Returns:
            Counterfactual instance or None if generation failed
        """
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model doesn't support predict_proba")
            return None

        # Check current prediction
        current_pred = model.predict(instance.reshape(1, -1))[0]

        if current_pred == target_class:
            logger.debug("Instance already belongs to target class")
            return instance

        logger.debug(f"Generating counterfactual using {self.method} method")

        # Generate based on method
        if self.method == "dice":
            counterfactual = self._generate_dice(
                instance, target_class, model, feature_ranges, categorical_features
            )
        elif self.method == "genetic":
            counterfactual = self._generate_genetic(
                instance, target_class, model, feature_ranges, categorical_features
            )
        elif self.method == "gradient":
            counterfactual = self._generate_gradient(
                instance, target_class, model, feature_ranges, categorical_features
            )
        elif self.method == "prototype":
            counterfactual = self._generate_prototype(
                instance, target_class, model, feature_ranges
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Validate counterfactual
        if counterfactual is not None:
            cf_pred = model.predict(counterfactual.reshape(1, -1))[0]

            if cf_pred == target_class:
                # Log successful generation
                self.generation_history.append({
                    'success': True,
                    'method': self.method,
                    'distance': float(euclidean(instance, counterfactual)),
                    'n_features_changed': int(np.sum(instance != counterfactual))
                })
                return counterfactual
            else:
                logger.debug("Generated counterfactual doesn't achieve target class")

        self.generation_history.append({'success': False, 'method': self.method})
        return None

    def _generate_dice(
        self,
        instance: np.ndarray,
        target_class: int,
        model: BaseEstimator,
        feature_ranges: List[Union[Tuple, List]],
        categorical_features: Optional[List[int]]
    ) -> Optional[np.ndarray]:
        """Generate counterfactual using DiCE-inspired method."""
        n_features = len(instance)
        categorical_features = categorical_features or []

        # Initialize counterfactual
        counterfactual = instance.copy()

        # Iterative optimization
        for iteration in range(self.max_iterations):
            # Get current prediction probability
            proba = model.predict_proba(counterfactual.reshape(1, -1))[0]

            if np.argmax(proba) == target_class and proba[target_class] > 0.5:
                return counterfactual

            # Compute gradient estimate
            gradients = self._estimate_gradients(
                counterfactual, target_class, model, epsilon=0.01
            )

            # Update features
            for i in range(n_features):
                if i in self.immutable_features:
                    continue

                if i in categorical_features:
                    # Categorical feature - try different values
                    best_value = counterfactual[i]
                    best_score = proba[target_class]

                    for value in feature_ranges[i]:
                        if value == counterfactual[i]:
                            continue

                        test_cf = counterfactual.copy()
                        test_cf[i] = value

                        test_proba = model.predict_proba(test_cf.reshape(1, -1))[0]

                        if test_proba[target_class] > best_score:
                            best_score = test_proba[target_class]
                            best_value = value

                    counterfactual[i] = best_value
                else:
                    # Continuous feature - gradient-based update
                    step_size = 0.1 * (feature_ranges[i][1] - feature_ranges[i][0])

                    if gradients[i] > 0:
                        counterfactual[i] = min(
                            counterfactual[i] + step_size,
                            feature_ranges[i][1]
                        )
                    else:
                        counterfactual[i] = max(
                            counterfactual[i] - step_size,
                            feature_ranges[i][0]
                        )

            # Add diversity if generating multiple counterfactuals
            if self.n_counterfactuals > 1 and iteration % 10 == 0:
                noise_scale = self.diversity * 0.1
                noise = np.random.randn(n_features) * noise_scale

                for i in range(n_features):
                    if i not in self.immutable_features and i not in categorical_features:
                        counterfactual[i] = np.clip(
                            counterfactual[i] + noise[i],
                            feature_ranges[i][0],
                            feature_ranges[i][1]
                        )

        return counterfactual if model.predict(counterfactual.reshape(1, -1))[0] == target_class else None

    def _generate_genetic(
        self,
        instance: np.ndarray,
        target_class: int,
        model: BaseEstimator,
        feature_ranges: List[Union[Tuple, List]],
        categorical_features: Optional[List[int]]
    ) -> Optional[np.ndarray]:
        """Generate counterfactual using genetic algorithm."""
        n_features = len(instance)
        categorical_features = categorical_features or []

        # Define objective function
        def objective(x):
            # Prediction loss
            proba = model.predict_proba(x.reshape(1, -1))[0]
            pred_loss = 1.0 - proba[target_class]

            # Distance loss
            if self.feature_weights is not None:
                distance = np.sqrt(np.sum(self.feature_weights * (x - instance) ** 2))
            else:
                distance = euclidean(x, instance)

            # Combined loss
            return self.distance_weight * distance + (1 - self.distance_weight) * pred_loss

        # Set bounds
        bounds = []
        for i in range(n_features):
            if i in self.immutable_features:
                bounds.append((instance[i], instance[i]))
            elif isinstance(feature_ranges[i], tuple):
                bounds.append(feature_ranges[i])
            else:
                bounds.append((min(feature_ranges[i]), max(feature_ranges[i])))

        # Run differential evolution
        result = differential_evolution(
            objective,
            bounds,
            maxiter=self.max_iterations,
            seed=self.random_state,
            polish=True,
            atol=1e-3
        )

        counterfactual = result.x

        # Handle categorical features
        for i in categorical_features:
            if i not in self.immutable_features:
                # Round to nearest valid value
                valid_values = feature_ranges[i]
                distances = [abs(counterfactual[i] - v) for v in valid_values]
                counterfactual[i] = valid_values[np.argmin(distances)]

        return counterfactual if model.predict(counterfactual.reshape(1, -1))[0] == target_class else None

    def _generate_gradient(
        self,
        instance: np.ndarray,
        target_class: int,
        model: BaseEstimator,
        feature_ranges: List[Union[Tuple, List]],
        categorical_features: Optional[List[int]]
    ) -> Optional[np.ndarray]:
        """Generate counterfactual using gradient-based optimization."""
        n_features = len(instance)
        categorical_features = categorical_features or []

        # Define objective function
        def objective(x):
            proba = model.predict_proba(x.reshape(1, -1))[0]
            pred_loss = -np.log(proba[target_class] + 1e-10)

            if self.feature_weights is not None:
                distance = np.sqrt(np.sum(self.feature_weights * (x - instance) ** 2))
            else:
                distance = euclidean(x, instance)

            return self.distance_weight * distance + (1 - self.distance_weight) * pred_loss

        # Define gradient function
        def gradient(x):
            grad = np.zeros(n_features)
            epsilon = 1e-5

            base_obj = objective(x)

            for i in range(n_features):
                if i in self.immutable_features:
                    continue

                x_plus = x.copy()
                x_plus[i] += epsilon

                grad[i] = (objective(x_plus) - base_obj) / epsilon

            return grad

        # Set bounds
        bounds = []
        for i in range(n_features):
            if i in self.immutable_features:
                bounds.append((instance[i], instance[i]))
            elif isinstance(feature_ranges[i], tuple):
                bounds.append(feature_ranges[i])
            else:
                bounds.append((min(feature_ranges[i]), max(feature_ranges[i])))

        # Run optimization
        result = minimize(
            objective,
            instance,
            method='L-BFGS-B',
            jac=gradient,
            bounds=bounds,
            options={'maxiter': self.max_iterations}
        )

        counterfactual = result.x

        # Handle categorical features
        for i in categorical_features:
            if i not in self.immutable_features:
                valid_values = feature_ranges[i]
                distances = [abs(counterfactual[i] - v) for v in valid_values]
                counterfactual[i] = valid_values[np.argmin(distances)]

        return counterfactual if model.predict(counterfactual.reshape(1, -1))[0] == target_class else None

    def _generate_prototype(
        self,
        instance: np.ndarray,
        target_class: int,
        model: BaseEstimator,
        feature_ranges: List[Union[Tuple, List]]
    ) -> Optional[np.ndarray]:
        """Generate counterfactual using prototype-based method."""
        n_features = len(instance)

        # Generate random prototypes from target class
        n_prototypes = 100
        prototypes = []

        for _ in range(n_prototypes):
            prototype = np.zeros(n_features)

            for i in range(n_features):
                if isinstance(feature_ranges[i], tuple):
                    prototype[i] = np.random.uniform(
                        feature_ranges[i][0],
                        feature_ranges[i][1]
                    )
                else:
                    prototype[i] = np.random.choice(feature_ranges[i])

            # Check if prototype belongs to target class
            if model.predict(prototype.reshape(1, -1))[0] == target_class:
                prototypes.append(prototype)

        if not prototypes:
            logger.warning("No valid prototypes found")
            return None

        # Find nearest prototype
        prototypes = np.array(prototypes)
        distances = np.array([euclidean(instance, p) for p in prototypes])
        nearest_idx = np.argmin(distances)

        # Interpolate between instance and nearest prototype
        prototype = prototypes[nearest_idx]
        alpha = 0.0

        counterfactual = instance.copy()

        for step in range(self.max_iterations):
            alpha = min(1.0, alpha + 0.1)

            for i in range(n_features):
                if i not in self.immutable_features:
                    counterfactual[i] = (1 - alpha) * instance[i] + alpha * prototype[i]

            if model.predict(counterfactual.reshape(1, -1))[0] == target_class:
                return counterfactual

        return None

    def _estimate_gradients(
        self,
        instance: np.ndarray,
        target_class: int,
        model: BaseEstimator,
        epsilon: float = 0.01
    ) -> np.ndarray:
        """Estimate gradients using finite differences."""
        n_features = len(instance)
        gradients = np.zeros(n_features)

        base_proba = model.predict_proba(instance.reshape(1, -1))[0][target_class]

        for i in range(n_features):
            if i in self.immutable_features:
                continue

            # Forward difference
            instance_plus = instance.copy()
            instance_plus[i] += epsilon

            proba_plus = model.predict_proba(instance_plus.reshape(1, -1))[0][target_class]

            gradients[i] = (proba_plus - base_proba) / epsilon

        return gradients

    def generate_batch(
        self,
        instances: np.ndarray,
        target_classes: np.ndarray,
        model: BaseEstimator,
        feature_ranges: List[Union[Tuple, List]],
        categorical_features: Optional[List[int]] = None,
        parallel: bool = False
    ) -> List[Optional[np.ndarray]]:
        """
        Generate counterfactuals for multiple instances.

        Args:
            instances: Array of instances
            target_classes: Target classes for each instance
            model: Surrogate model
            feature_ranges: Valid ranges for features
            categorical_features: Indices of categorical features
            parallel: Whether to generate in parallel

        Returns:
            List of counterfactuals (None for failed generations)
        """
        counterfactuals = []

        if parallel and len(instances) > 1:
            from multiprocessing import Pool

            with Pool() as pool:
                args = [
                    (inst, target, model, feature_ranges, categorical_features)
                    for inst, target in zip(instances, target_classes)
                ]

                results = pool.starmap(self.generate, args)
                counterfactuals = results
        else:
            for instance, target_class in zip(instances, target_classes):
                cf = self.generate(
                    instance,
                    target_class,
                    model,
                    feature_ranges,
                    categorical_features
                )
                counterfactuals.append(cf)

        # Log batch statistics
        n_successful = sum(1 for cf in counterfactuals if cf is not None)
        logger.info(f"Generated {n_successful}/{len(instances)} counterfactuals successfully")

        return counterfactuals

    def generate_diverse_set(
        self,
        instance: np.ndarray,
        target_class: int,
        model: BaseEstimator,
        feature_ranges: List[Union[Tuple, List]],
        n_counterfactuals: int = None,
        categorical_features: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Generate diverse set of counterfactuals for single instance.

        Args:
            instance: Original instance
            target_class: Target class
            model: Surrogate model
            feature_ranges: Valid ranges for features
            n_counterfactuals: Number of counterfactuals to generate
            categorical_features: Indices of categorical features

        Returns:
            List of diverse counterfactuals
        """
        n_counterfactuals = n_counterfactuals or self.n_counterfactuals
        counterfactuals = []

        # Generate initial counterfactual
        cf = self.generate(
            instance,
            target_class,
            model,
            feature_ranges,
            categorical_features
        )

        if cf is not None:
            counterfactuals.append(cf)

        # Generate additional diverse counterfactuals
        attempts = 0
        max_attempts = n_counterfactuals * 10

        while len(counterfactuals) < n_counterfactuals and attempts < max_attempts:
            attempts += 1

            # Add noise for diversity
            noisy_instance = instance.copy()

            for i in range(len(instance)):
                if i not in self.immutable_features:
                    if categorical_features and i in categorical_features:
                        if np.random.random() < self.diversity:
                            noisy_instance[i] = np.random.choice(feature_ranges[i])
                    else:
                        noise = np.random.randn() * self.diversity * (feature_ranges[i][1] - feature_ranges[i][0])
                        noisy_instance[i] = np.clip(
                            noisy_instance[i] + noise,
                            feature_ranges[i][0],
                            feature_ranges[i][1]
                        )

            # Generate counterfactual from noisy instance
            cf = self.generate(
                noisy_instance,
                target_class,
                model,
                feature_ranges,
                categorical_features
            )

            if cf is not None:
                # Check diversity from existing counterfactuals
                is_diverse = True

                for existing_cf in counterfactuals:
                    distance = euclidean(cf, existing_cf)

                    if distance < 0.1:  # Minimum diversity threshold
                        is_diverse = False
                        break

                if is_diverse:
                    counterfactuals.append(cf)

        logger.info(f"Generated {len(counterfactuals)} diverse counterfactuals")

        return counterfactuals

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        if not self.generation_history:
            return {}

        successful = [h for h in self.generation_history if h.get('success', False)]

        stats = {
            'total_attempts': len(self.generation_history),
            'successful_generations': len(successful),
            'success_rate': len(successful) / len(self.generation_history) if self.generation_history else 0,
            'method_used': self.method
        }

        if successful:
            stats['avg_distance'] = np.mean([h['distance'] for h in successful])
            stats['avg_features_changed'] = np.mean([h['n_features_changed'] for h in successful])

        return stats


__all__ = ['CounterfactualGenerator']