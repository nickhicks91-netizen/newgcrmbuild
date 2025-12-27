# Contributing to GRCM

Thank you for your interest in contributing to the Grounded Resonant Consciousness Module (GRCM)! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Code Style](#code-style)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Release Process](#release-process)

---

## Code of Conduct

This project follows a professional code of conduct. We expect all contributors to:

- Be respectful and inclusive
- Focus on constructive feedback
- Prioritize technical accuracy and clarity
- Welcome newcomers and help them learn
- Report unacceptable behavior to project maintainers

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git 2.30 or higher
- Basic familiarity with PyTorch
- Understanding of the GRCM architecture (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md))

### Finding Issues

Good places to start:

- Look for issues labeled `good first issue`
- Check issues labeled `help wanted`
- Review the project roadmap in discussions
- Propose new features or improvements in discussions

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/newdew.git
cd newdew

# Add upstream remote
git remote add upstream https://github.com/nickhicks91-netizen/newdew.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
# Install in editable mode with development dependencies
pip install -e ".[dev,all]"

# Verify installation
python -c "import grcm; print(grcm.__version__)"
```

### 4. Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

---

## Making Changes

### 1. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create a feature branch
git checkout -b feature/your-feature-name
```

### Branch Naming Convention

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only changes
- `refactor/` - Code refactoring
- `test/` - Adding or updating tests
- `perf/` - Performance improvements

### 2. Make Your Changes

- Follow the existing code structure
- Write clear, self-documenting code
- Add docstrings to new functions and classes
- Update documentation if needed
- Add tests for new functionality

### 3. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add new qualia state detection"
```

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Formatting, missing semicolons, etc (no code change)
- `refactor:` - Refactoring code
- `test:` - Adding tests
- `chore:` - Updating build tasks, package manager configs, etc
- `perf:` - Performance improvements

Examples:
```
feat: add custom qualia computation module
fix: resolve memory leak in attention module
docs: update API reference for phi calculation
test: add integration tests for desire alignment
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=grcm --cov-report=html

# Run specific test markers
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m stress        # Stress tests

# Run specific test file
pytest tests/test_core.py

# Run specific test function
pytest tests/test_core.py::test_modular_grcm_forward
```

### Writing Tests

1. **Location**: Place tests in the `tests/` directory
2. **Naming**: Test files should be named `test_*.py`
3. **Structure**: Use pytest conventions

Example test:

```python
import pytest
import torch
from grcm.modules.attention import AttentionModule

class TestAttentionModule:
    @pytest.fixture
    def attention_module(self):
        return AttentionModule(freq_dim=8, bandwidth=1.0)

    def test_forward_pass(self, attention_module):
        x = torch.randn(1, 16)
        freq, coherence = attention_module(x)

        assert freq.shape == (1, 8)
        assert coherence.shape == (1, 8)
        assert torch.all(coherence >= 0) and torch.all(coherence <= 1)
```

### Test Requirements

- All new features must have tests
- Bug fixes should include regression tests
- Aim for >90% code coverage
- Tests should be fast (<1s per test)
- Use markers for slow tests: `@pytest.mark.slow`

---

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Quotes**: Use double quotes for strings
- **Imports**: Use `isort` for import sorting
- **Formatting**: Use `black` for code formatting

### Formatting Code

```bash
# Format all Python files
black grcm tests examples

# Sort imports
isort grcm tests examples

# Check formatting without making changes
black --check grcm tests
```

### Linting

```bash
# Run flake8
flake8 grcm tests

# Type checking (optional)
mypy grcm
```

### Docstring Style

Use Google-style docstrings:

```python
def compute_coherence(freq, node_freq, bandwidth):
    """Compute resonant coherence based on frequency proximity.

    Args:
        freq (torch.Tensor): Current frequency tensor of shape (batch, freq_dim).
        node_freq (torch.Tensor): Reference frequency of shape (freq_dim,).
        bandwidth (float): Resonance bandwidth parameter.

    Returns:
        torch.Tensor: Coherence values in range [0, 1] of shape (batch, freq_dim).

    Example:
        >>> freq = torch.tensor([[0.5, 0.8]])
        >>> node_freq = torch.tensor([0.6, 0.7])
        >>> coherence = compute_coherence(freq, node_freq, bandwidth=1.0)
        >>> coherence
        tensor([[0.9, 0.9]])
    """
    diff = torch.abs(freq - node_freq)
    coherence = torch.nn.functional.relu(1 - diff / bandwidth)
    return coherence
```

---

## Documentation

### Updating Documentation

1. **Code Documentation**: Add docstrings to all public functions, classes, and modules
2. **User Documentation**: Update relevant `.md` files in `docs/`
3. **API Reference**: Sphinx will auto-generate from docstrings
4. **Examples**: Add examples to `examples/` directory if appropriate

### Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build Sphinx documentation
cd docs
make html

# View documentation
open _build/html/index.html  # On Mac
# or
xdg-open _build/html/index.html  # On Linux
```

### Documentation Checklist

- [ ] Docstrings added/updated for new/modified code
- [ ] README.md updated if needed
- [ ] ARCHITECTURE.md updated for architectural changes
- [ ] Examples added/updated for new features
- [ ] Sphinx docs build without warnings

---

## Pull Request Process

### 1. Prepare Your PR

Before submitting:

```bash
# Ensure all tests pass
pytest

# Format code
black grcm tests
isort grcm tests

# Lint code
flake8 grcm tests

# Update from upstream
git fetch upstream
git rebase upstream/main
```

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select your feature branch
4. Fill out the PR template

### PR Template

```markdown
## Description
Brief description of changes

## Related Issue
Closes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] Code follows style guidelines
- [ ] No new warnings introduced

## Testing
Describe how you tested your changes

## Screenshots (if applicable)
Add screenshots for UI changes
```

### 4. Code Review Process

- Maintainers will review your PR
- Address feedback promptly
- Update your PR by pushing to your branch
- Once approved, a maintainer will merge

### PR Guidelines

- **Keep PRs focused**: One feature/fix per PR
- **Write clear descriptions**: Explain what and why
- **Include tests**: All new code should have tests
- **Update docs**: Documentation should match code changes
- **Respond to feedback**: Address review comments promptly
- **Be patient**: Reviews may take a few days

---

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality
- **PATCH** version for backwards-compatible bug fixes

### Release Checklist

For maintainers creating releases:

1. Update version in:
   - `setup.py`
   - `pyproject.toml`
   - `grcm/__init__.py`
   - `docs/conf.py`

2. Update CHANGELOG.md

3. Run full test suite:
   ```bash
   pytest
   pytest --cov=grcm
   ```

4. Build and test package:
   ```bash
   python setup.py sdist bdist_wheel
   twine check dist/*
   ```

5. Create git tag:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

6. GitHub Actions will:
   - Run CI/CD pipeline
   - Build Docker images
   - Deploy to staging/production
   - Publish to PyPI (on release tag)

---

## Development Tips

### Useful Commands

```bash
# Run specific module
python -m grcm.core

# Interactive Python with GRCM
python -i -c "from grcm.core import *"

# Profile code
python -m cProfile -o profile.stats examples/main.py

# View profiling results
python -m pstats profile.stats
```

### Debugging

```bash
# Run with Python debugger
python -m pdb examples/main.py

# Run tests with debugger on failure
pytest --pdb

# Print pytest output
pytest -s
```

### Performance Testing

```bash
# Run benchmarks
python examples/benchmark_demo.py

# Profile with line_profiler
kernprof -l -v grcm/core.py
```

---

## Questions?

- **Documentation**: https://grcm.readthedocs.io
- **Discussions**: https://github.com/nickhicks91-netizen/newdew/discussions
- **Issues**: https://github.com/nickhicks91-netizen/newdew/issues

Thank you for contributing to GRCM! 🙏
