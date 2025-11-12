# Build and Distribution Guide

This guide explains how to build and distribute the `mlflow-eval-tools` package to other teams.

## Prerequisites

- Python 3.12+
- Poetry installed (`pip install poetry`)
- Git repository access

## Building the Package

### 1. Update Version (if needed)

Edit `pyproject.toml` and `src/mlflow_eval_tools/__init__.py`:

```toml
# pyproject.toml
[tool.poetry]
version = "0.2.0"  # Update version
```

```python
# src/mlflow_eval_tools/__init__.py
__version__ = "0.2.0"  # Update version
```

### 2. Install Dependencies

```bash
poetry install
```

### 3. Run Tests

```bash
poetry run pytest
```

### 4. Build Distribution Files

```bash
poetry build
```

This creates two files in the `dist/` directory:

- `mlflow_eval_tools-0.1.0-py3-none-any.whl` (wheel - recommended)
- `mlflow_eval_tools-0.1.0.tar.gz` (source distribution)

## Distribution Options

### Option 1: Direct Wheel Distribution (Easiest for Teams)

Share the wheel file directly with other teams:

```bash
# Build the wheel
poetry build

# Share dist/mlflow_eval_tools-0.1.0-py3-none-any.whl with teams

# Teams install with:
pip install mlflow_eval_tools-0.1.0-py3-none-any.whl
```

**Advantages:**
- Simple, no external dependencies
- Works in air-gapped environments
- Fast installation
- No need for build tools

### Option 2: Internal PyPI Server

Set up an internal PyPI server (e.g., with `devpi` or `pypiserver`):

```bash
# Build
poetry build

# Upload to internal PyPI
poetry config repositories.internal http://pypi.internal.company.com
poetry publish -r internal
```

**Teams install with:**

```bash
pip install mlflow-eval-tools --index-url http://pypi.internal.company.com/simple
```

### Option 3: Git Repository

Teams can install directly from the Git repository:

```bash
pip install git+https://github.com/sdeery14/llm-system-lifecycle.git
```

Or from a specific branch/tag:

```bash
pip install git+https://github.com/sdeery14/llm-system-lifecycle.git@v0.1.0
```

### Option 4: Network Share

Place the wheel on a network share:

```bash
# Copy to network share
cp dist/mlflow_eval_tools-0.1.0-py3-none-any.whl //share/python-packages/

# Teams install with:
pip install //share/python-packages/mlflow_eval_tools-0.1.0-py3-none-any.whl
```

### Option 5: Public PyPI (if open-source)

```bash
# Configure PyPI token
poetry config pypi-token.pypi your-token-here

# Publish
poetry publish
```

**Teams install with:**

```bash
pip install mlflow-eval-tools
```

## Installation Instructions for Teams

### For Wheel Distribution

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the package
pip install mlflow_eval_tools-0.1.0-py3-none-any.whl

# Verify installation
mlflow-eval-tools --version
mlflow-eval-tools info
```

### For Requirements File

Teams can add to `requirements.txt`:

```txt
# From wheel file
mlflow_eval_tools @ file:///path/to/mlflow_eval_tools-0.1.0-py3-none-any.whl

# Or from Git
mlflow-eval-tools @ git+https://github.com/sdeery14/llm-system-lifecycle.git@v0.1.0

# Or from PyPI
mlflow-eval-tools==0.1.0
```

Then:

```bash
pip install -r requirements.txt
```

### For Poetry Projects

Add to `pyproject.toml`:

```toml
[tool.poetry.dependencies]
mlflow-eval-tools = {path = "path/to/mlflow_eval_tools-0.1.0-py3-none-any.whl"}

# Or from Git
mlflow-eval-tools = {git = "https://github.com/sdeery14/llm-system-lifecycle.git", tag = "v0.1.0"}

# Or from PyPI
mlflow-eval-tools = "^0.1.0"
```

Then:

```bash
poetry install
```

## Package Contents

The distributed package includes:

```
mlflow_eval_tools/
├── __init__.py           # Package initialization
├── __main__.py           # CLI entry point
└── cli.py                # CLI implementation

app_agents/
├── dataset_builder.py    # Dataset builder agent
└── agent_analysis.py     # Agent analysis tool

Documentation:
├── README.md
├── PACKAGE_README.md
├── LICENSE
└── docs/
    ├── QUICK_START_TEAMS.md
    └── [other docs]
```

## Verifying the Build

### Check Package Structure

```bash
# List wheel contents
unzip -l dist/mlflow_eval_tools-0.1.0-py3-none-any.whl

# Or with Python
python -m zipfile -l dist/mlflow_eval_tools-0.1.0-py3-none-any.whl
```

### Test Installation in Clean Environment

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate

# Install
pip install dist/mlflow_eval_tools-0.1.0-py3-none-any.whl

# Test CLI
mlflow-eval-tools --version
mlflow-eval-tools info

# Cleanup
deactivate
rm -rf test_env
```

## Versioning Strategy

Follow semantic versioning (SemVer):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.2.0): New features, backwards compatible
- **PATCH** (0.1.1): Bug fixes

### Version Bump Checklist

1. Update version in `pyproject.toml`
2. Update version in `src/mlflow_eval_tools/__init__.py`
3. Update version in `src/mlflow_eval_tools/cli.py` (version_option)
4. Update CHANGELOG.md (create if doesn't exist)
5. Commit changes: `git commit -m "Bump version to 0.2.0"`
6. Tag release: `git tag -a v0.2.0 -m "Release v0.2.0"`
7. Push: `git push origin main --tags`
8. Build and distribute

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/build.yml
name: Build Package

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install Poetry
        run: pip install poetry
      
      - name: Build package
        run: poetry build
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: dist
          path: dist/
```

## Troubleshooting

### "Module not found" errors after installation

Ensure `packages` in `pyproject.toml` includes all necessary modules:

```toml
packages = [
    { include = "mlflow_eval_tools", from = "src" },
    { include = "app_agents", from = "src" },
]
```

### CLI command not found

Check that `[tool.poetry.scripts]` is configured:

```toml
[tool.poetry.scripts]
mlflow-eval-tools = "mlflow_eval_tools.cli:cli"
```

### Missing dependencies

Verify all dependencies are in `pyproject.toml`:

```toml
[tool.poetry.dependencies]
click = "^8.1.7"
mlflow = "^3.4.0"
# ... etc
```

## Support Documentation for Teams

When distributing, include:

1. **Installation Instructions**
   - Copy from QUICK_START_TEAMS.md

2. **Environment Setup**
   - Provide .env.example
   - List required environment variables

3. **Quick Start Guide**
   - Simple examples
   - Common workflows

4. **Troubleshooting**
   - Common issues and solutions

5. **Contact Information**
   - Support channel
   - Issue tracker

## Example Distribution Email

```
Subject: New Tool: mlflow-eval-tools for Agent Evaluation

Hi Team,

We've packaged our evaluation tools for OpenAI Agents SDK projects.

Installation:
  pip install mlflow_eval_tools-0.1.0-py3-none-any.whl

Quick Start:
  1. Set OPENAI_API_KEY in .env
  2. Run: mlflow-eval-tools dataset-builder
  3. Run: mlflow-eval-tools agent-analysis <run_id> <dataset_name>

Documentation:
  - See attached QUICK_START_TEAMS.md
  - Full docs: PACKAGE_README.md

Support:
  - Slack: #ml-evaluation
  - Issues: https://github.com/sdeery14/llm-system-lifecycle/issues

Best regards,
ML Platform Team
```

## Maintenance

### Regular Updates

1. Update dependencies periodically:
   ```bash
   poetry update
   ```

2. Check for security vulnerabilities:
   ```bash
   poetry show --outdated
   pip-audit  # if installed
   ```

3. Test with latest OpenAI Agents SDK versions

### Deprecation Process

If deprecating features:

1. Add deprecation warnings (use `warnings` module)
2. Update documentation
3. Maintain backwards compatibility for 1-2 versions
4. Remove in major version bump

## Summary

Choose the distribution method that best fits your organization:

- **Quick start**: Direct wheel distribution
- **Long-term**: Internal PyPI server
- **Open-source**: Public PyPI

Always include:
- Wheel file (.whl)
- Documentation
- Example .env file
- Quick start guide

For questions, contact the ML Platform team or create an issue on GitHub.
