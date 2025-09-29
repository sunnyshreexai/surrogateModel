"""Core SurrogateModel implementation with two-phase approach."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
import warnings
from pathlib import Path
import pickle
import json

from .config import SurrogateModelConfig, load_config
from .combinatorial import CombinatorialTester
from .active_learning import ActiveLearner
from .counterfactual import CounterfactualGenerator
from .utils import (
    set_seed,
    get_device,
    create_surrogate_model,
    evaluate_agreement,
    compute_feature_importance
)

logger = logging.getLogger(__name__)


class SurrogateModel:
    """
    Main class for surrogate model construction using combinatorial testing and active learning.

    This implements the two-phase approach from the paper:
    1. Phase 1: Combinatorial testing to capture feature interactions
    2. Phase 2: Active learning with counterfactuals to refine the model
    """

    def __init__(
        self,
        config: Optional[Union[SurrogateModelConfig, Dict, str]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize SurrogateModel.

        Args:
            config: Configuration object, dict, path, or preset name
            seed: Random seed for reproducibility
        """
        self.config = load_config(config) if config else SurrogateModelConfig()
        self.config.validate()

        if seed is not None:
            set_seed(seed)

        self.surrogate_model: Optional[BaseEstimator] = None
        self.combinatorial_tester: Optional[CombinatorialTester] = None
        self.active_learner: Optional[ActiveLearner] = None
        self.counterfactual_generator: Optional[CounterfactualGenerator] = None

        self.training_data: Optional[np.ndarray] = None
        self.training_labels: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.n_features: Optional[int] = None
        self.n_classes: Optional[int] = None

        self.results: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.cache: Dict[str, Any] = {}

        if self.config.verbose >= 1:
            self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging based on verbosity level."""
        level = logging.INFO if self.config.verbose == 1 else logging.DEBUG
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def construct(
        self,
        black_box_model: Callable,
        feature_ranges: Union[List[Tuple], np.ndarray],
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Construct surrogate model for the given black-box model.

        Args:
            black_box_model: Black-box model to approximate (callable that returns predictions)
            feature_ranges: List of (min, max) tuples for continuous features or list of values for categorical
            feature_names: Optional names for features
            categorical_features: Indices of categorical features
            validation_data: Optional (X_val, y_val) for validation

        Returns:
            Dictionary containing results and metrics
        """
        start_time = time.time()

        logger.info("Starting surrogate model construction")

        # Initialize components
        self.n_features = len(feature_ranges)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.n_features)]

        # Phase 1: Combinatorial Testing
        logger.info("Phase 1: Combinatorial testing for initial training set")

        self.combinatorial_tester = CombinatorialTester(
            t_way=self.config.t_way,
            discretization_method=self.config.discretization_method,
            n_bins=self.config.discretization_bins,
            coverage=self.config.initial_coverage
        )

        # Generate initial training set
        initial_instances = self.combinatorial_tester.generate_test_suite(
            feature_ranges=feature_ranges,
            categorical_features=categorical_features
        )

        # Query black-box model
        initial_labels = self._query_black_box(black_box_model, initial_instances)

        self.training_data = initial_instances
        self.training_labels = initial_labels
        self.n_classes = len(np.unique(initial_labels))

        logger.info(f"Generated {len(initial_instances)} initial instances")

        # Train initial surrogate model
        self._train_surrogate_model()

        initial_accuracy = self._evaluate_on_validation(validation_data, black_box_model)
        logger.info(f"Initial surrogate accuracy: {initial_accuracy:.4f}")

        # Phase 2: Active Learning with Counterfactuals
        logger.info("Phase 2: Active learning refinement")

        self.active_learner = ActiveLearner(
            strategy=self.config.selection_strategy,
            uncertainty_weight=self.config.uncertainty_weight,
            batch_size=self.config.batch_size,
            top_k_ratio=self.config.top_k_ratio
        )

        self.counterfactual_generator = CounterfactualGenerator(
            method=self.config.cf_method,
            diversity=self.config.cf_diversity,
            n_counterfactuals=self.config.counterfactuals_per_point
        )

        # Active learning loop
        query_budget = int(len(initial_instances) * self.config.query_budget_ratio)

        for iteration in range(self.config.max_iterations):
            if len(self.training_data) >= len(initial_instances) + query_budget:
                logger.info("Query budget exhausted")
                break

            # Check convergence
            current_accuracy = self._evaluate_on_validation(validation_data, black_box_model)

            if current_accuracy >= self.config.convergence_threshold:
                logger.info(f"Converged at iteration {iteration} with accuracy {current_accuracy:.4f}")
                break

            if iteration > 0:
                improvement = current_accuracy - self.history[-1]['accuracy']
                if improvement < self.config.min_improvement:
                    logger.info(f"Insufficient improvement: {improvement:.4f}")
                    break

            # Select queries using active learning
            candidate_pool = self._generate_candidate_pool(feature_ranges, categorical_features)

            queries = self.active_learner.select_queries(
                model=self.surrogate_model,
                candidate_pool=candidate_pool,
                existing_data=self.training_data
            )

            # Check class balance and generate counterfactuals if needed
            queries = self._balance_with_counterfactuals(
                queries, black_box_model, feature_ranges
            )

            # Query black-box model
            query_labels = self._query_black_box(black_box_model, queries)

            # Update training set
            self.training_data = np.vstack([self.training_data, queries])
            self.training_labels = np.hstack([self.training_labels, query_labels])

            # Retrain surrogate model
            self._train_surrogate_model()

            # Log iteration results
            iteration_results = {
                'iteration': iteration,
                'queries_added': len(queries),
                'total_samples': len(self.training_data),
                'accuracy': current_accuracy,
                'time': time.time() - start_time
            }

            self.history.append(iteration_results)

            if self.config.save_intermediate:
                self._save_checkpoint(iteration)

            logger.info(f"Iteration {iteration}: Added {len(queries)} queries, "
                       f"Accuracy: {current_accuracy:.4f}")

        # Final evaluation
        final_accuracy = self._evaluate_on_validation(validation_data, black_box_model)

        # Compile results
        self.results = {
            'success': final_accuracy >= self.config.convergence_threshold,
            'final_accuracy': final_accuracy,
            'initial_accuracy': initial_accuracy,
            'improvement': final_accuracy - initial_accuracy,
            'total_queries': len(self.training_data),
            'iterations': len(self.history),
            'time_taken': time.time() - start_time,
            'config': self.config.to_dict(),
            'history': self.history,
            'feature_importance': self._compute_feature_importance()
        }

        logger.info(f"Surrogate model construction complete. "
                   f"Final accuracy: {final_accuracy:.4f}")

        return self.results

    def _train_surrogate_model(self) -> None:
        """Train the surrogate model on current training data."""
        if self.config.surrogate_model_type == "auto":
            self.surrogate_model = self._select_best_model()
        else:
            self.surrogate_model = create_surrogate_model(
                model_type=self.config.surrogate_model_type,
                complexity=self.config.model_complexity,
                n_features=self.n_features,
                n_classes=self.n_classes
            )

        # Split for validation if needed
        if self.config.validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                self.training_data,
                self.training_labels,
                test_size=self.config.validation_split,
                stratify=self.training_labels,
                random_state=42
            )

            # Train with early stopping if applicable
            if hasattr(self.surrogate_model, 'partial_fit'):
                self._train_with_early_stopping(X_train, y_train, X_val, y_val)
            else:
                self.surrogate_model.fit(X_train, y_train)
        else:
            self.surrogate_model.fit(self.training_data, self.training_labels)

    def _select_best_model(self) -> BaseEstimator:
        """Automatically select the best model type."""
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }

        best_score = -1
        best_model = None

        X_train, X_val, y_train, y_val = train_test_split(
            self.training_data,
            self.training_labels,
            test_size=0.2,
            stratify=self.training_labels,
            random_state=42
        )

        for name, model in models.items():
            model.fit(X_train, y_train)
            score = accuracy_score(y_val, model.predict(X_val))

            if score > best_score:
                best_score = score
                best_model = model

            logger.debug(f"Model {name} validation score: {score:.4f}")

        logger.info(f"Selected model with score: {best_score:.4f}")
        return best_model

    def _train_with_early_stopping(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """Train model with early stopping."""
        best_score = -1
        patience_counter = 0

        for epoch in range(100):
            self.surrogate_model.partial_fit(X_train, y_train, classes=np.unique(y_train))

            val_score = accuracy_score(y_val, self.surrogate_model.predict(X_val))

            if val_score > best_score:
                best_score = val_score
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                logger.debug(f"Early stopping at epoch {epoch}")
                break

    def _query_black_box(
        self,
        black_box_model: Callable,
        instances: np.ndarray
    ) -> np.ndarray:
        """Query the black-box model for labels."""
        if self.config.parallel_workers > 1:
            # Parallel querying
            from multiprocessing import Pool

            with Pool(self.config.parallel_workers) as pool:
                labels = pool.map(black_box_model, instances)
            return np.array(labels)
        else:
            return np.array([black_box_model(x) for x in instances])

    def _evaluate_on_validation(
        self,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]],
        black_box_model: Callable
    ) -> float:
        """Evaluate surrogate model agreement with black-box on validation data."""
        if validation_data is None:
            # Generate validation set
            n_val = min(1000, len(self.training_data) // 2)
            val_indices = np.random.choice(len(self.training_data), n_val, replace=False)
            X_val = self.training_data[val_indices]
            y_val = self.training_labels[val_indices]
        else:
            X_val, y_val = validation_data

        surrogate_predictions = self.surrogate_model.predict(X_val)

        if y_val is None:
            y_val = self._query_black_box(black_box_model, X_val)

        return accuracy_score(y_val, surrogate_predictions)

    def _generate_candidate_pool(
        self,
        feature_ranges: List[Tuple],
        categorical_features: Optional[List[int]]
    ) -> np.ndarray:
        """Generate candidate pool for active learning."""
        n_candidates = min(10000, 100 * self.config.batch_size)
        candidates = []

        for _ in range(n_candidates):
            instance = np.zeros(self.n_features)

            for i, range_spec in enumerate(feature_ranges):
                if categorical_features and i in categorical_features:
                    instance[i] = np.random.choice(range_spec)
                else:
                    instance[i] = np.random.uniform(range_spec[0], range_spec[1])

            candidates.append(instance)

        return np.array(candidates)

    def _balance_with_counterfactuals(
        self,
        queries: np.ndarray,
        black_box_model: Callable,
        feature_ranges: List[Tuple]
    ) -> np.ndarray:
        """Balance queries with counterfactual examples if needed."""
        # Get current class distribution
        query_labels = self._query_black_box(black_box_model, queries)

        class_counts = np.bincount(query_labels)
        minority_class = np.argmin(class_counts)
        majority_class = np.argmax(class_counts)

        ratio = class_counts[minority_class] / (class_counts[majority_class] + 1e-10)

        if ratio < self.config.opposite_class_threshold:
            # Generate counterfactuals for majority class instances
            majority_indices = np.where(query_labels == majority_class)[0]

            counterfactuals = []
            for idx in majority_indices[:self.config.counterfactuals_per_point]:
                cf = self.counterfactual_generator.generate(
                    instance=queries[idx],
                    target_class=minority_class,
                    model=self.surrogate_model,
                    feature_ranges=feature_ranges
                )
                if cf is not None:
                    counterfactuals.append(cf)

            if counterfactuals:
                queries = np.vstack([queries, np.array(counterfactuals)])
                logger.debug(f"Added {len(counterfactuals)} counterfactuals for balance")

        return queries

    def _compute_feature_importance(self) -> Optional[Dict[str, float]]:
        """Compute feature importance scores."""
        if hasattr(self.surrogate_model, 'feature_importances_'):
            importances = self.surrogate_model.feature_importances_
        elif hasattr(self.surrogate_model, 'coef_'):
            importances = np.abs(self.surrogate_model.coef_).mean(axis=0)
        else:
            return None

        return dict(zip(self.feature_names, importances))

    def _save_checkpoint(self, iteration: int) -> None:
        """Save intermediate checkpoint."""
        if not self.config.output_dir:
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / f"checkpoint_iter_{iteration}.pkl"

        checkpoint = {
            'iteration': iteration,
            'model': self.surrogate_model,
            'training_data': self.training_data,
            'training_labels': self.training_labels,
            'history': self.history,
            'config': self.config.to_dict()
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        logger.debug(f"Saved checkpoint to {checkpoint_path}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the surrogate model."""
        if self.surrogate_model is None:
            raise ValueError("Surrogate model not trained yet")
        return self.surrogate_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the surrogate model."""
        if self.surrogate_model is None:
            raise ValueError("Surrogate model not trained yet")
        return self.surrogate_model.predict_proba(X)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        black_box_model: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Evaluate surrogate model performance.

        Args:
            X: Test instances
            y_true: True labels (if available)
            black_box_model: Black-box model for comparison

        Returns:
            Dictionary of evaluation metrics
        """
        if self.surrogate_model is None:
            raise ValueError("Surrogate model not trained yet")

        predictions = self.predict(X)

        metrics = {}

        if y_true is not None:
            metrics['accuracy'] = accuracy_score(y_true, predictions)
            metrics['f1_score'] = f1_score(y_true, predictions, average='weighted')

        if black_box_model is not None:
            bb_predictions = self._query_black_box(black_box_model, X)
            metrics['agreement'] = accuracy_score(bb_predictions, predictions)

        return metrics

    def save(self, path: str) -> None:
        """Save the surrogate model and configuration."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'model': self.surrogate_model,
            'config': self.config.to_dict(),
            'training_data': self.training_data,
            'training_labels': self.training_labels,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'results': self.results,
            'history': self.history
        }

        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)

        logger.info(f"Saved surrogate model to {save_path}")

    @classmethod
    def load(cls, path: str) -> 'SurrogateModel':
        """Load a saved surrogate model."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        instance = cls(config=save_dict['config'])
        instance.surrogate_model = save_dict['model']
        instance.training_data = save_dict['training_data']
        instance.training_labels = save_dict['training_labels']
        instance.feature_names = save_dict['feature_names']
        instance.n_features = save_dict['n_features']
        instance.n_classes = save_dict['n_classes']
        instance.results = save_dict.get('results', {})
        instance.history = save_dict.get('history', [])

        return instance

    def explain(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Explain a prediction using the surrogate model.

        Args:
            instance: Input instance to explain

        Returns:
            Dictionary containing explanation details
        """
        if self.surrogate_model is None:
            raise ValueError("Surrogate model not trained yet")

        prediction = self.predict(instance.reshape(1, -1))[0]
        proba = self.predict_proba(instance.reshape(1, -1))[0]

        explanation = {
            'prediction': int(prediction),
            'confidence': float(max(proba)),
            'probabilities': {f"class_{i}": float(p) for i, p in enumerate(proba)}
        }

        # Feature contributions (if available)
        if hasattr(self.surrogate_model, 'coef_'):
            contributions = instance * self.surrogate_model.coef_[prediction if self.n_classes == 2 else 0]
            explanation['feature_contributions'] = {
                name: float(contrib)
                for name, contrib in zip(self.feature_names, contributions)
            }

        return explanation


__all__ = ['SurrogateModel']