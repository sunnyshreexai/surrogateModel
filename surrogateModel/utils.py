"""Utility functions for SurrogateModel package."""

import numpy as np
import pickle
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import random
import torch
import warnings

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger.info(f"Random seed set to {seed}")


def get_device() -> str:
    """
    Get the available device (CPU/GPU).

    Returns:
        Device string ('cuda' or 'cpu')
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = 'cpu'
            logger.info("Using CPU")
    except ImportError:
        device = 'cpu'
        logger.info("PyTorch not available, using CPU")

    return device


def create_surrogate_model(
    model_type: str = "auto",
    complexity: str = "medium",
    n_features: Optional[int] = None,
    n_classes: Optional[int] = None,
    **kwargs
) -> BaseEstimator:
    """
    Create a surrogate model based on specifications.

    Args:
        model_type: Type of model ('logistic', 'random_forest', 'neural_network', 'auto')
        complexity: Model complexity ('low', 'medium', 'high')
        n_features: Number of features
        n_classes: Number of classes
        **kwargs: Additional model parameters

    Returns:
        Scikit-learn compatible model
    """
    if model_type == "logistic":
        model = LogisticRegression(
            max_iter=1000 if complexity == "low" else 5000,
            C=10.0 if complexity == "low" else 1.0,
            random_state=42,
            **kwargs
        )
    elif model_type == "random_forest":
        n_estimators = {"low": 10, "medium": 100, "high": 500}.get(complexity, 100)
        max_depth = {"low": 5, "medium": 10, "high": None}.get(complexity, 10)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
    elif model_type == "neural_network":
        hidden_layers = {
            "low": (50,),
            "medium": (100, 50),
            "high": (200, 100, 50)
        }.get(complexity, (100, 50))

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=1000,
            early_stopping=True,
            random_state=42,
            **kwargs
        )
    elif model_type == "auto":
        # Auto-select based on dataset characteristics
        if n_features and n_classes:
            if n_features < 10 and n_classes == 2:
                model = create_surrogate_model("logistic", complexity)
            elif n_features < 100:
                model = create_surrogate_model("random_forest", complexity)
            else:
                model = create_surrogate_model("neural_network", complexity)
        else:
            model = create_surrogate_model("random_forest", complexity)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.debug(f"Created {model_type} model with {complexity} complexity")

    return model


def evaluate_agreement(
    model1: BaseEstimator,
    model2: Union[BaseEstimator, Callable],
    X: np.ndarray,
    return_detailed: bool = False
) -> Union[float, Dict[str, Any]]:
    """
    Evaluate agreement between two models.

    Args:
        model1: First model
        model2: Second model (or callable black-box)
        X: Test instances
        return_detailed: Whether to return detailed metrics

    Returns:
        Agreement accuracy or detailed metrics dict
    """
    predictions1 = model1.predict(X)

    if hasattr(model2, 'predict'):
        predictions2 = model2.predict(X)
    else:
        # Assume model2 is a callable black-box
        predictions2 = np.array([model2(x) for x in X])

    agreement = accuracy_score(predictions1, predictions2)

    if return_detailed:
        metrics = {
            'agreement': agreement,
            'n_samples': len(X),
            'n_disagreements': np.sum(predictions1 != predictions2)
        }

        # Compute confusion matrix if both are classifiers
        if len(np.unique(predictions1)) < 20 and len(np.unique(predictions2)) < 20:
            metrics['confusion_matrix'] = confusion_matrix(predictions1, predictions2).tolist()

        return metrics

    return agreement


def compute_feature_importance(
    model: BaseEstimator,
    feature_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute feature importance scores.

    Args:
        model: Trained model
        feature_names: Names of features

    Returns:
        Dictionary mapping feature names to importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        logger.warning("Model doesn't have feature importance attributes")
        return {}

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance_dict = dict(zip(feature_names, importances))

    # Sort by importance
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    return importance_dict


def load_model(path: str) -> Any:
    """
    Load a saved model.

    Args:
        path: Path to saved model

    Returns:
        Loaded model object
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.suffix == '.pkl':
        with open(path, 'rb') as f:
            model = pickle.load(f)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            model = json.load(f)
    elif path.suffix in ['.pt', '.pth']:
        try:
            import torch
            model = torch.load(path, map_location='cpu')
        except ImportError:
            raise ImportError("PyTorch is required to load .pt/.pth files")
    else:
        # Try pickle by default
        with open(path, 'rb') as f:
            model = pickle.load(f)

    logger.info(f"Loaded model from {path}")

    return model


def save_model(model: Any, path: str) -> None:
    """
    Save a model to disk.

    Args:
        model: Model to save
        path: Save path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, 'save'):
        # Model has its own save method
        model.save(path)
    elif path.suffix == '.json':
        # Save as JSON if possible
        try:
            with open(path, 'w') as f:
                json.dump(model, f, indent=2, default=str)
        except TypeError:
            # Fall back to pickle
            with open(path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(model, f)
    else:
        # Default to pickle
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    logger.info(f"Saved model to {path}")


def normalize_data(
    X: np.ndarray,
    method: str = "minmax",
    feature_range: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize data.

    Args:
        X: Data to normalize
        method: Normalization method ('minmax', 'standard', 'robust')
        feature_range: Range for minmax normalization

    Returns:
        Normalized data and normalization parameters
    """
    params = {'method': method}

    if method == "minmax":
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)

        # Avoid division by zero
        X_range = X_max - X_min
        X_range[X_range == 0] = 1

        X_norm = (X - X_min) / X_range
        X_norm = X_norm * (feature_range[1] - feature_range[0]) + feature_range[0]

        params['min'] = X_min
        params['max'] = X_max
        params['feature_range'] = feature_range

    elif method == "standard":
        mean = X.mean(axis=0)
        std = X.std(axis=0)

        # Avoid division by zero
        std[std == 0] = 1

        X_norm = (X - mean) / std

        params['mean'] = mean
        params['std'] = std

    elif method == "robust":
        median = np.median(X, axis=0)
        mad = np.median(np.abs(X - median), axis=0)

        # Avoid division by zero
        mad[mad == 0] = 1

        X_norm = (X - median) / mad

        params['median'] = median
        params['mad'] = mad

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm, params


def denormalize_data(
    X_norm: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Denormalize data.

    Args:
        X_norm: Normalized data
        params: Normalization parameters

    Returns:
        Denormalized data
    """
    method = params['method']

    if method == "minmax":
        feature_range = params['feature_range']
        X_norm = (X_norm - feature_range[0]) / (feature_range[1] - feature_range[0])
        X = X_norm * (params['max'] - params['min']) + params['min']

    elif method == "standard":
        X = X_norm * params['std'] + params['mean']

    elif method == "robust":
        X = X_norm * params['mad'] + params['median']

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X


def generate_synthetic_data(
    n_samples: int,
    n_features: int,
    n_classes: int = 2,
    n_informative: Optional[int] = None,
    noise: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset for testing.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        n_informative: Number of informative features
        noise: Noise level
        random_state: Random seed

    Returns:
        X, y arrays
    """
    from sklearn.datasets import make_classification

    n_informative = n_informative or max(2, n_features // 2)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=min(n_features - n_informative, n_features // 4),
        n_classes=n_classes,
        n_clusters_per_class=max(1, n_classes // 2),
        flip_y=noise,
        random_state=random_state
    )

    logger.debug(f"Generated synthetic dataset: {X.shape}")

    return X, y


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        average: Averaging strategy for multi-class

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred)
    }

    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics['per_class'] = {
        'precision': precision_per_class.tolist(),
        'recall': recall_per_class.tolist(),
        'f1_score': f1_per_class.tolist(),
        'support': support.tolist()
    }

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # ROC-AUC if probabilities are available
    if y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score

            n_classes = y_proba.shape[1]

            if n_classes == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                # Multi-class
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average=average
                )
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")

    return metrics


def validate_feature_ranges(
    feature_ranges: List[Union[Tuple, List]],
    X: Optional[np.ndarray] = None
) -> List[Union[Tuple, List]]:
    """
    Validate and adjust feature ranges.

    Args:
        feature_ranges: Specified feature ranges
        X: Optional data to infer ranges from

    Returns:
        Validated feature ranges
    """
    validated_ranges = []

    for i, range_spec in enumerate(feature_ranges):
        if isinstance(range_spec, tuple):
            if len(range_spec) != 2:
                raise ValueError(f"Invalid range for feature {i}: {range_spec}")

            min_val, max_val = range_spec

            if min_val >= max_val:
                if X is not None and i < X.shape[1]:
                    # Use data range
                    min_val = X[:, i].min()
                    max_val = X[:, i].max()
                else:
                    # Default range
                    min_val, max_val = 0, 1

                logger.warning(f"Adjusted range for feature {i}: ({min_val}, {max_val})")

            validated_ranges.append((min_val, max_val))

        elif isinstance(range_spec, list):
            if len(range_spec) == 0:
                raise ValueError(f"Empty range for feature {i}")

            validated_ranges.append(range_spec)

        else:
            raise ValueError(f"Invalid range type for feature {i}: {type(range_spec)}")

    return validated_ranges


def create_black_box_wrapper(
    model: Any,
    preprocessing: Optional[Callable] = None,
    postprocessing: Optional[Callable] = None
) -> Callable:
    """
    Create a wrapper for black-box model.

    Args:
        model: The model to wrap
        preprocessing: Optional preprocessing function
        postprocessing: Optional postprocessing function

    Returns:
        Callable black-box function
    """
    def black_box(x: np.ndarray) -> Any:
        # Preprocess
        if preprocessing is not None:
            x = preprocessing(x)

        # Get prediction
        if hasattr(model, 'predict'):
            # Ensure correct shape
            if x.ndim == 1:
                x = x.reshape(1, -1)
            pred = model.predict(x)[0]
        elif callable(model):
            pred = model(x)
        else:
            raise ValueError("Model must be callable or have predict method")

        # Postprocess
        if postprocessing is not None:
            pred = postprocessing(pred)

        return pred

    return black_box


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Operation"):
        """Initialize timer with optional name."""
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        """Stop timing and log."""
        self.elapsed = time.time() - self.start_time
        logger.debug(f"{self.name} took {self.elapsed:.2f} seconds")


def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of optional dependencies.

    Returns:
        Dictionary of dependency availability
    """
    dependencies = {}

    # Check PyTorch
    try:
        import torch
        dependencies['pytorch'] = True
        dependencies['cuda'] = torch.cuda.is_available()
    except ImportError:
        dependencies['pytorch'] = False
        dependencies['cuda'] = False

    # Check TensorFlow
    try:
        import tensorflow
        dependencies['tensorflow'] = True
    except ImportError:
        dependencies['tensorflow'] = False

    # Check visualization
    try:
        import matplotlib
        dependencies['matplotlib'] = True
    except ImportError:
        dependencies['matplotlib'] = False

    # Check optimization libraries
    try:
        import optuna
        dependencies['optuna'] = True
    except ImportError:
        dependencies['optuna'] = False

    return dependencies


# Ensure compatibility
import time

__all__ = [
    'set_seed',
    'get_device',
    'create_surrogate_model',
    'evaluate_agreement',
    'compute_feature_importance',
    'load_model',
    'save_model',
    'normalize_data',
    'denormalize_data',
    'generate_synthetic_data',
    'compute_metrics',
    'validate_feature_ranges',
    'create_black_box_wrapper',
    'Timer',
    'check_dependencies'
]