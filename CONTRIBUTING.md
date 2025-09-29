# Contributing to SurrogateModel

We welcome contributions to the SurrogateModel project! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully

## How to Contribute

### Reporting Issues

1. Check existing issues to avoid duplicates
2. Use issue templates when available
3. Provide detailed information:
   - Python version
   - Package versions
   - Minimal reproducible example
   - Error messages and stack traces

### Submitting Pull Requests

1. **Fork the repository** and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the existing code style
   - Add/update tests as needed
   - Update documentation

3. **Test your changes**:
   ```bash
   pytest tests/
   ```

4. **Format your code**:
   ```bash
   black surrogatemodel/
   isort surrogatemodel/
   flake8 surrogatemodel/
   ```

5. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add new feature X"
   ```

6. **Push and create PR**:
   - Push to your fork
   - Create pull request with description
   - Link related issues

### Commit Message Convention

We follow conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sunnyshreexai/surrogateModel.git
   cd surrogateModel
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests**:
   ```bash
   pytest tests/ --cov=surrogatemodel
   ```

## Code Style Guidelines

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to all public functions/classes

## Testing Guidelines

- Write tests for new features
- Maintain test coverage above 80%
- Use pytest fixtures for common setup
- Test edge cases and error conditions

## Documentation

- Update docstrings for API changes
- Update README for new features
- Add examples for complex functionality
- Keep documentation concise and clear

## Areas for Contribution

### Priority Areas
- Performance optimizations
- Additional counterfactual methods
- New active learning strategies
- Visualization tools
- Integration with more ML frameworks

### Good First Issues
Look for issues labeled `good first issue` for beginner-friendly tasks.

## Review Process

1. All PRs require at least one review
2. CI tests must pass
3. Code coverage should not decrease
4. Documentation must be updated

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion
- Contact maintainers

Thank you for contributing to SurrogateModel!