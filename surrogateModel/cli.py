"""Command-line interface for SurrogateModel."""

import click
import json
import yaml
import pickle
from pathlib import Path
from typing import Optional, Any
import logging
import numpy as np

from .core import SurrogateModel
from .config import SurrogateModelConfig, ConfigPresets
from .utils import generate_synthetic_data, set_seed, check_dependencies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """SurrogateModel: Construct surrogate models for black-box ML models."""
    pass


@cli.command()
@click.option('--model-path', '-m', required=True, help='Path to black-box model')
@click.option('--feature-ranges', '-f', required=True, help='JSON file with feature ranges')
@click.option('--config', '-c', help='Config file or preset name')
@click.option('--output', '-o', default='surrogate_model.pkl', help='Output path')
@click.option('--validation-data', '-v', help='Path to validation data')
@click.option('--t-way', type=int, help='T-way interaction strength')
@click.option('--query-budget', type=float, help='Query budget ratio')
@click.option('--verbose', '-v', count=True, help='Increase verbosity')
def construct(model_path, feature_ranges, config, output, validation_data, t_way, query_budget, verbose):
    """Construct a surrogate model for a black-box model."""

    # Load configuration
    if config:
        if Path(config).exists():
            if config.endswith('.yaml'):
                config_obj = SurrogateModelConfig.from_yaml(config)
            else:
                config_obj = SurrogateModelConfig.from_json(config)
        else:
            # Try as preset
            config_obj = getattr(ConfigPresets, config)() if hasattr(ConfigPresets, config) else SurrogateModelConfig()
    else:
        config_obj = SurrogateModelConfig()

    # Override with command-line options
    if t_way:
        config_obj.t_way = t_way
    if query_budget:
        config_obj.query_budget_ratio = query_budget
    if verbose:
        config_obj.verbose = verbose

    # Load black-box model
    with open(model_path, 'rb') as f:
        black_box_model = pickle.load(f)

    # Load feature ranges
    with open(feature_ranges, 'r') as f:
        ranges_data = json.load(f)

    feature_ranges_list = ranges_data['feature_ranges']
    categorical_features = ranges_data.get('categorical_features', [])

    # Load validation data if provided
    val_data = None
    if validation_data:
        with open(validation_data, 'rb') as f:
            val_data = pickle.load(f)

    # Initialize and construct surrogate model
    surrogate = SurrogateModel(config_obj)

    results = surrogate.construct(
        black_box_model=black_box_model,
        feature_ranges=feature_ranges_list,
        categorical_features=categorical_features,
        validation_data=val_data
    )

    # Save surrogate model
    surrogate.save(output)

    # Display results
    click.echo(f"\nSurrogate Model Construction Complete:")
    click.echo(f"  Final accuracy: {results['final_accuracy']:.4f}")
    click.echo(f"  Improvement: {results['improvement']:.4f}")
    click.echo(f"  Total queries: {results['total_queries']}")
    click.echo(f"  Time taken: {results['time_taken']:.2f}s")
    click.echo(f"  Model saved to: {output}")


@cli.command()
@click.option('--model-path', '-m', required=True, help='Path to surrogate model')
@click.option('--data-path', '-d', required=True, help='Path to test data')
@click.option('--black-box', '-b', help='Path to black-box model for comparison')
@click.option('--output', '-o', help='Save results to file')
def evaluate(model_path, data_path, black_box, output):
    """Evaluate a surrogate model."""

    # Load surrogate model
    surrogate = SurrogateModel.load(model_path)

    # Load test data
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    X_test = test_data['X']
    y_test = test_data.get('y', None)

    # Load black-box model if provided
    bb_model = None
    if black_box:
        with open(black_box, 'rb') as f:
            bb_model = pickle.load(f)

    # Evaluate
    metrics = surrogate.evaluate(X_test, y_test, bb_model)

    # Display results
    click.echo("\nEvaluation Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            click.echo(f"  {key}: {value:.4f}")
        else:
            click.echo(f"  {key}: {value}")

    # Save results if requested
    if output:
        with open(output, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        click.echo(f"\nResults saved to: {output}")


@cli.command()
@click.option('--preset', '-p', type=click.Choice(['fast', 'accurate', 'balanced', 'memory_efficient', 'gpu_optimized']),
              help='Configuration preset to show')
@click.option('--output', '-o', help='Save configuration to file')
def config(preset, output):
    """Show or save configuration presets."""

    if preset:
        config_obj = getattr(ConfigPresets, preset)()
        config_dict = config_obj.to_dict()

        if output:
            if output.endswith('.yaml'):
                config_obj.to_yaml(output)
            else:
                config_obj.to_json(output)
            click.echo(f"Configuration saved to: {output}")
        else:
            click.echo(f"\n{preset.upper()} Configuration:")
            for key, value in config_dict.items():
                click.echo(f"  {key}: {value}")
    else:
        click.echo("\nAvailable presets:")
        click.echo("  fast: Quick construction with approximations")
        click.echo("  accurate: Thorough analysis with high precision")
        click.echo("  balanced: Good balance of speed and accuracy")
        click.echo("  memory_efficient: For large datasets")
        click.echo("  gpu_optimized: Optimized for GPU computation")


@cli.command()
@click.option('--n-samples', '-n', default=1000, help='Number of samples')
@click.option('--n-features', '-f', default=10, help='Number of features')
@click.option('--n-classes', '-c', default=2, help='Number of classes')
@click.option('--output', '-o', required=True, help='Output path for synthetic data')
def generate_data(n_samples, n_features, n_classes, output):
    """Generate synthetic dataset for testing."""

    set_seed(42)

    # Generate data
    X, y = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes
    )

    # Create feature ranges
    feature_ranges = [(float(X[:, i].min()), float(X[:, i].max())) for i in range(n_features)]

    # Save data
    data = {
        'X': X,
        'y': y,
        'feature_ranges': feature_ranges,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes
    }

    with open(output, 'wb') as f:
        pickle.dump(data, f)

    click.echo(f"\nGenerated synthetic dataset:")
    click.echo(f"  Samples: {n_samples}")
    click.echo(f"  Features: {n_features}")
    click.echo(f"  Classes: {n_classes}")
    click.echo(f"  Saved to: {output}")


@cli.command()
@click.option('--model-path', '-m', required=True, help='Path to surrogate model')
@click.option('--instance', '-i', required=True, help='JSON file with instance to explain')
def explain(model_path, instance):
    """Explain a prediction using the surrogate model."""

    # Load surrogate model
    surrogate = SurrogateModel.load(model_path)

    # Load instance
    with open(instance, 'r') as f:
        instance_data = json.load(f)

    instance_array = np.array(instance_data['values'])

    # Get explanation
    explanation = surrogate.explain(instance_array)

    # Display explanation
    click.echo("\nPrediction Explanation:")
    click.echo(f"  Predicted class: {explanation['prediction']}")
    click.echo(f"  Confidence: {explanation['confidence']:.2%}")

    if 'probabilities' in explanation:
        click.echo("\n  Class probabilities:")
        for cls, prob in explanation['probabilities'].items():
            click.echo(f"    {cls}: {prob:.4f}")

    if 'feature_contributions' in explanation:
        click.echo("\n  Feature contributions:")
        sorted_contrib = sorted(
            explanation['feature_contributions'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        for feat, contrib in sorted_contrib[:10]:
            click.echo(f"    {feat}: {contrib:+.4f}")


@cli.command()
def check_env():
    """Check environment and dependencies."""

    click.echo("Checking environment...\n")

    # Check Python version
    import sys
    click.echo(f"Python version: {sys.version}")

    # Check dependencies
    deps = check_dependencies()

    click.echo("\nDependency status:")
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        click.echo(f"  {status} {dep}")

    # Check GPU
    if deps.get('cuda'):
        import torch
        click.echo(f"\nGPU available: {torch.cuda.get_device_name()}")
        click.echo(f"CUDA version: {torch.version.cuda}")


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()