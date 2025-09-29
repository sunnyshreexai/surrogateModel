"""
SurrogateModel: ML Model Construction Using Combinatorial Testing and Active Learning

A production-ready framework for constructing surrogate models that approximate
black-box ML models through feature interaction capture and query optimization.
"""

__version__ = "1.0.0"
__author__ = "Sunny Shree, Krishna Khadka, Yu Lei"
__email__ = "sunny.shree@mavs.uta.edu"

# Core imports
from .config import SurrogateModelConfig
from .core import SurrogateModel
from .combinatorial import CombinatorialTester
from .active_learning import ActiveLearner
from .counterfactual import CounterfactualGenerator
from .utils import (
    create_surrogate_model,
    load_model,
    save_model,
    evaluate_agreement,
    set_seed,
    get_device
)

__all__ = [
    # Core classes
    "SurrogateModel",
    "SurrogateModelConfig",
    "CombinatorialTester",
    "ActiveLearner",
    "CounterfactualGenerator",

    # Utilities
    "create_surrogate_model",
    "load_model",
    "save_model",
    "evaluate_agreement",
    "set_seed",
    "get_device",

    # Version info
    "__version__",
]


def get_version():
    """Return the version string."""
    return __version__


def show_config():
    """Display default configuration."""
    from .config import SurrogateModelConfig

    config = SurrogateModelConfig()
    print("Default SurrogateModel Configuration:")
    print(config)


# Package metadata
__metadata__ = {
    "name": "SurrogateModel",
    "version": __version__,
    "description": "ML Model Construction Using Combinatorial Testing and Active Learning",
    "url": "https://github.com/sunnyshreexai/surrogateModel",
    "license": "MIT",
    "python_requires": ">=3.8",
}