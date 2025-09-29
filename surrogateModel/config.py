"""Configuration module for SurrogateModel with all parameters optional and configurable."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
import yaml
import json
from pathlib import Path


@dataclass
class SurrogateModelConfig:
    """
    Comprehensive configuration for surrogate model construction.
    All parameters have sensible defaults and are fully configurable.
    """

    # ============ Phase 1: Combinatorial Testing Parameters ============
    t_way: int = 3
    """T-way interaction strength for combinatorial testing (default: 3)"""

    discretization_method: str = "uniform"
    """Method for discretizing continuous features: 'uniform', 'equal_width', 'equal_freq', 'entropy' (default: 'uniform')"""

    discretization_bins: int = 5
    """Number of bins for discretization (default: 5)"""

    initial_coverage: float = 1.0
    """Coverage level for initial t-way testing (default: 1.0 for 100% coverage)"""

    # ============ Phase 2: Active Learning Parameters ============
    query_budget_ratio: float = 0.25
    """Ratio of dataset size to use as query budget (default: 0.25)"""

    selection_strategy: str = "combined"
    """Query selection strategy: 'uncertainty', 'diversity', 'combined' (default: 'combined')"""

    uncertainty_weight: float = 0.5
    """Weight for uncertainty in combined strategy (default: 0.5)"""

    batch_size: int = 32
    """Batch size for query selection (default: 32)"""

    top_k_ratio: float = 0.1
    """Ratio of candidate points to select as queries (default: 0.1)"""

    # ============ Counterfactual Generation Parameters ============
    opposite_class_threshold: float = 0.1
    """Minimum ratio of opposite-class to target-class points (default: 0.1)"""

    counterfactuals_per_point: int = 1
    """Number of counterfactuals to generate per data point (default: 1)"""

    cf_method: str = "dice"
    """Counterfactual generation method: 'dice', 'genetic', 'gradient' (default: 'dice')"""

    cf_diversity: float = 0.5
    """Diversity parameter for counterfactual generation (default: 0.5)"""

    # ============ Model Training Parameters ============
    surrogate_model_type: str = "auto"
    """Type of surrogate model: 'logistic', 'random_forest', 'neural_network', 'auto' (default: 'auto')"""

    model_complexity: str = "medium"
    """Complexity level: 'low', 'medium', 'high' (default: 'medium')"""

    validation_split: float = 0.15
    """Validation data split ratio (default: 0.15)"""

    early_stopping_patience: int = 5
    """Patience for early stopping (default: 5)"""

    # ============ Iteration Control Parameters ============
    max_iterations: int = 10
    """Maximum iterations for essential interaction isolation (default: 10)"""

    convergence_threshold: float = 0.95
    """Agreement accuracy threshold for convergence (default: 0.95)"""

    min_improvement: float = 0.01
    """Minimum improvement per iteration to continue (default: 0.01)"""

    # ============ Performance Optimization Parameters ============
    use_gpu: bool = True
    """Use GPU if available (default: True)"""

    parallel_workers: int = 4
    """Number of parallel workers for processing (default: 4)"""

    cache_interactions: bool = True
    """Cache computed interactions for reuse (default: True)"""

    memory_efficient: bool = False
    """Use memory-efficient mode at the cost of speed (default: False)"""

    checkpoint_interval: int = 100
    """Interval for saving checkpoints (default: 100)"""

    # ============ Logging and Output Parameters ============
    verbose: int = 1
    """Verbosity level: 0=silent, 1=progress, 2=detailed (default: 1)"""

    log_file: Optional[str] = None
    """Path to log file (default: None)"""

    save_intermediate: bool = False
    """Save intermediate results during construction (default: False)"""

    output_dir: Optional[str] = None
    """Directory for saving outputs (default: None)"""

    results_format: str = "json"
    """Format for saving results: 'json', 'pickle', 'csv' (default: 'json')"""

    # ============ Advanced Parameters ============
    feature_importance_threshold: float = 0.0
    """Minimum feature importance to consider (default: 0.0)"""

    interaction_pruning: bool = True
    """Enable pruning of non-essential interactions (default: True)"""

    adaptive_discretization: bool = False
    """Adapt discretization based on feature distribution (default: False)"""

    use_ensemble: bool = False
    """Use ensemble of surrogate models (default: False)"""

    ensemble_size: int = 3
    """Number of models in ensemble (default: 3)"""

    # ============ Experimental Features ============
    use_metalearning: bool = False
    """Use meta-learning for model selection (default: False)"""

    transfer_learning: bool = False
    """Enable transfer learning from similar models (default: False)"""

    automl_optimization: bool = False
    """Use AutoML for hyperparameter optimization (default: False)"""

    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []

        # Validate ranges
        if self.t_way < 1 or self.t_way > 6:
            errors.append(f"t_way must be between 1 and 6, got {self.t_way}")

        if not 0 < self.query_budget_ratio <= 1:
            errors.append(f"query_budget_ratio must be in (0, 1], got {self.query_budget_ratio}")

        if not 0 <= self.uncertainty_weight <= 1:
            errors.append(f"uncertainty_weight must be in [0, 1], got {self.uncertainty_weight}")

        if self.discretization_bins < 2:
            errors.append(f"discretization_bins must be at least 2, got {self.discretization_bins}")

        if self.max_iterations < 1:
            errors.append(f"max_iterations must be positive, got {self.max_iterations}")

        # Validate string options
        valid_discretization = ['uniform', 'equal_width', 'equal_freq', 'entropy']
        if self.discretization_method not in valid_discretization:
            errors.append(f"Invalid discretization_method: {self.discretization_method}")

        valid_strategies = ['uncertainty', 'diversity', 'combined']
        if self.selection_strategy not in valid_strategies:
            errors.append(f"Invalid selection_strategy: {self.selection_strategy}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_yaml(self, path: Optional[str] = None) -> str:
        """Export configuration to YAML format."""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=True)

        if path:
            Path(path).write_text(yaml_str)

        return yaml_str

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """Export configuration to JSON format."""
        json_str = json.dumps(self.to_dict(), indent=indent, default=str)

        if path:
            Path(path).write_text(json_str)

        return json_str

    @classmethod
    def from_yaml(cls, path: str) -> "SurrogateModelConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "SurrogateModelConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SurrogateModelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def update(self, **kwargs) -> "SurrogateModelConfig":
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Unknown configuration parameter: {key}")

        self.validate()
        return self

    def __str__(self) -> str:
        """String representation of configuration."""
        params = []
        for key, value in self.to_dict().items():
            params.append(f"  {key}: {value}")
        return "SurrogateModelConfig(\n" + "\n".join(params) + "\n)"

    def __repr__(self) -> str:
        """Detailed representation of configuration."""
        return f"SurrogateModelConfig({self.to_dict()})"


class ConfigPresets:
    """Preset configurations for common scenarios."""

    @staticmethod
    def fast() -> SurrogateModelConfig:
        """Fast configuration for quick prototyping."""
        return SurrogateModelConfig(
            t_way=2,
            discretization_bins=3,
            query_budget_ratio=0.15,
            max_iterations=5,
            model_complexity="low",
            cache_interactions=True,
            memory_efficient=True
        )

    @staticmethod
    def accurate() -> SurrogateModelConfig:
        """High-accuracy configuration for thorough analysis."""
        return SurrogateModelConfig(
            t_way=4,
            discretization_bins=7,
            query_budget_ratio=0.35,
            max_iterations=15,
            model_complexity="high",
            use_ensemble=True,
            ensemble_size=5
        )

    @staticmethod
    def balanced() -> SurrogateModelConfig:
        """Balanced configuration for general use."""
        return SurrogateModelConfig(
            t_way=3,
            discretization_bins=5,
            query_budget_ratio=0.25,
            max_iterations=10,
            model_complexity="medium"
        )

    @staticmethod
    def memory_efficient() -> SurrogateModelConfig:
        """Memory-efficient configuration for large datasets."""
        return SurrogateModelConfig(
            batch_size=16,
            memory_efficient=True,
            cache_interactions=False,
            parallel_workers=2,
            checkpoint_interval=50
        )

    @staticmethod
    def gpu_optimized() -> SurrogateModelConfig:
        """GPU-optimized configuration."""
        return SurrogateModelConfig(
            use_gpu=True,
            batch_size=128,
            parallel_workers=8,
            cache_interactions=True
        )


def load_config(source: Any) -> SurrogateModelConfig:
    """
    Load configuration from various sources.

    Args:
        source: Can be:
            - SurrogateModelConfig instance
            - Dictionary of parameters
            - Path to YAML/JSON file
            - String preset name

    Returns:
        SurrogateModelConfig instance
    """
    if isinstance(source, SurrogateModelConfig):
        return source

    if isinstance(source, dict):
        return SurrogateModelConfig.from_dict(source)

    if isinstance(source, str):
        # Check if it's a preset
        if source.lower() in ['fast', 'accurate', 'balanced', 'memory_efficient', 'gpu_optimized']:
            return getattr(ConfigPresets, source.lower())()

        # Check if it's a file path
        path = Path(source)
        if path.exists():
            if path.suffix in ['.yaml', '.yml']:
                return SurrogateModelConfig.from_yaml(source)
            elif path.suffix == '.json':
                return SurrogateModelConfig.from_json(source)

    raise ValueError(f"Cannot load configuration from: {source}")


__all__ = [
    "SurrogateModelConfig",
    "ConfigPresets",
    "load_config"
]