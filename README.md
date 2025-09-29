# Constructing Surrogate Models in Machine Learning Using Combinatorial Testing and Active Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

SurrogateModel is a production-ready Python package for constructing surrogate models that approximate black-box machine learning models through systematic feature interaction capture and intelligent query optimization. It implements the two-phase approach combining combinatorial testing with active learning as described in the research paper.

## Key Features

- **Two-Phase Construction**: Systematic approach combining combinatorial testing and active learning
- **T-way Testing**: Capture feature interactions systematically using combinatorial testing
- **Active Learning**: Intelligent query selection using uncertainty and diversity strategies
- **Counterfactual Generation**: Generate opposite-class examples for improved class balance
- **Production Ready**: Comprehensive error handling, logging, and performance optimization
- **Framework Agnostic**: Works with any black-box model (PyTorch, TensorFlow, scikit-learn)
- **Performance Optimized**: GPU support, caching, and parallel processing capabilities

## Installation

### From PyPI (recommended)

```bash
pip install surrogatemodel
```

### From Source

```bash
git clone https://github.com/sunnyshreexai/surrogateModel.git
cd surrogateModel
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/sunnyshreexai/surrogateModel.git
cd surrogateModel
pip install -e ".[dev]"
```

## Quick Start

```python
from surrogatemodel import SurrogateModel, SurrogateModelConfig

# Define your black-box model
def black_box_model(x):
    # Your complex model logic here
    return prediction

# Define feature ranges
feature_ranges = [
    (0, 10),      # Continuous feature 1
    (0, 1),       # Continuous feature 2
    [0, 1, 2, 3]  # Categorical feature 3
]

# Initialize with default configuration
surrogate = SurrogateModel()

# Construct surrogate model
results = surrogate.construct(
    black_box_model=black_box_model,
    feature_ranges=feature_ranges,
    categorical_features=[2]  # Index of categorical feature
)

print(f"Final accuracy: {results['final_accuracy']:.4f}")
print(f"Queries used: {results['total_queries']}")
```

## Architecture

SurrogateModel employs a two-phase approach:

### Phase 1: Combinatorial Testing
- Generates initial training set using t-way testing
- Ensures systematic coverage of feature interactions
- Discretizes continuous features intelligently

### Phase 2: Active Learning with Counterfactuals
- Selects informative queries using uncertainty/diversity strategies
- Generates counterfactual examples for class balance
- Iteratively refines the surrogate model

```
┌─────────────────────────────────────────┐
│         Black-Box Model                  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│   Phase 1: Combinatorial Testing        │
│   • Generate t-way test suite           │
│   • Query black-box model               │
│   • Train initial surrogate             │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│   Phase 2: Active Learning              │
│   • Select queries (uncertainty/diversity)│
│   • Generate counterfactuals            │
│   • Retrain surrogate model             │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│       Optimized Surrogate Model         │
└─────────────────────────────────────────┘
```

## Configuration

All parameters are optional and configurable:

```python
from surrogatemodel import SurrogateModelConfig, ConfigPresets

# Use preset configuration
config = ConfigPresets.accurate()  # High accuracy configuration

# Or customize parameters
config = SurrogateModelConfig(
    # Combinatorial Testing
    t_way=3,                           # T-way interaction strength
    discretization_method="uniform",   # Discretization method
    discretization_bins=5,             # Number of bins

    # Active Learning
    query_budget_ratio=0.25,           # Query budget as ratio
    selection_strategy="combined",     # Query selection strategy
    uncertainty_weight=0.5,            # Weight for uncertainty

    # Counterfactual Generation
    cf_method="dice",                  # Counterfactual method
    counterfactuals_per_point=1,       # Number of counterfactuals

    # Model Training
    surrogate_model_type="auto",       # Auto-select model type
    validation_split=0.15,             # Validation split ratio

    # Performance
    use_gpu=True,                      # Use GPU if available
    parallel_workers=4,                # Parallel workers
    cache_interactions=True,           # Cache computations

    # Output
    verbose=1,                         # Logging verbosity
    save_intermediate=True             # Save checkpoints
)

surrogate = SurrogateModel(config)
```

### Configuration Presets

```python
# Fast configuration for quick prototyping
config = ConfigPresets.fast()

# Accurate configuration for thorough analysis
config = ConfigPresets.accurate()

# Balanced configuration for general use
config = ConfigPresets.balanced()

# Memory-efficient for large datasets
config = ConfigPresets.memory_efficient()

# GPU-optimized configuration
config = ConfigPresets.gpu_optimized()
```



## API Reference

### Core Classes

- `SurrogateModel`: Main class for surrogate model construction
- `SurrogateModelConfig`: Configuration management
- `CombinatorialTester`: T-way testing implementation
- `ActiveLearner`: Query selection strategies
- `CounterfactualGenerator`: Counterfactual generation



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@inproceedings{shree2024surrogate,
  title={Constructing Surrogate Models Using Combinatorial Testing and Active Learning},
  author={Shree, Sunny and Khadka, Krishna and Lei, Yu},
  booktitle={Proceedings of the International Conference on Software Testing},
  year={2024}
}
```

## Support

For issues, questions, or suggestions, please open an issue on [GitHub](https://github.com/sunnyshreexai/surrogateModel/issues).
