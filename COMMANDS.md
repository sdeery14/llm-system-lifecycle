# Quick Commands Reference

Quick reference for common tasks with `mlflow-eval-tools`.

## Installation

```bash
# From wheel (recommended)
pip install mlflow_eval_tools-0.1.0-py3-none-any.whl

# From source
git clone https://github.com/sdeery14/llm-system-lifecycle.git
cd llm-system-lifecycle
poetry install

# Verify installation
mlflow-eval-tools --version
python test_installation.py
```

## Environment Setup

```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-your-key-here
MAX_DATASET_INSTANCES=100
EOF

# Or export directly
export OPENAI_API_KEY=sk-your-key-here
```

## Dataset Builder Commands

```bash
# Interactive mode (recommended for first-time users)
mlflow-eval-tools dataset-builder

# Specify agent upfront
mlflow-eval-tools dataset-builder \
  --agent-file src/dev_agents/my_agent.py \
  --agent-class MyAgent

# Use previously logged agent
mlflow-eval-tools dataset-builder --use-previous abc123runid

# List available agents
mlflow-eval-tools dataset-builder --list-agents

# Custom configuration
mlflow-eval-tools dataset-builder \
  --agent-file src/dev_agents/my_agent.py \
  --agent-class MyAgent \
  --max-size 200 \
  --batch-size 50 \
  --model gpt-4o \
  --worker-model gpt-4o-mini
```

## Agent Analysis Commands

```bash
# Basic evaluation
mlflow-eval-tools agent-analysis abc123runid my_dataset_v1

# Custom experiment names
mlflow-eval-tools agent-analysis abc123runid my_dataset \
  --dataset-experiment my-datasets \
  --eval-experiment my-evaluations

# Skip saving artifacts
mlflow-eval-tools agent-analysis abc123runid my_dataset --no-artifacts
```

## Package Info

```bash
# Show package information
mlflow-eval-tools info

# Show CLI help
mlflow-eval-tools --help
mlflow-eval-tools dataset-builder --help
mlflow-eval-tools agent-analysis --help
```

## MLflow Commands

```bash
# Start MLflow UI
mlflow ui

# Open in browser: http://localhost:5000

# Start on different port
mlflow ui --port 8080

# Use specific backend store
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Building & Distribution

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Clean previous builds
rm -rf dist/

# Build package
poetry build

# Verify build
ls -lh dist/

# Inspect wheel contents
unzip -l dist/mlflow_eval_tools-0.1.0-py3-none-any.whl

# Test in clean environment
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate
pip install dist/mlflow_eval_tools-0.1.0-py3-none-any.whl
python test_installation.py
deactivate
rm -rf test_env
```

## Git & Versioning

```bash
# Commit changes
git add .
git commit -m "Bump version to 0.2.0"

# Tag release
git tag -a v0.2.0 -m "Release version 0.2.0"

# Push with tags
git push origin main --tags

# List tags
git tag -l
```

## Development Workflow

```bash
# 1. Make changes to code
# 2. Run tests
poetry run pytest

# 3. Update version in:
#    - pyproject.toml
#    - src/mlflow_eval_tools/__init__.py
#    - src/mlflow_eval_tools/cli.py

# 4. Build
poetry build

# 5. Test installation
python test_installation.py

# 6. Commit and tag
git commit -am "Release v0.2.0"
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin main --tags

# 7. Distribute wheel file
```

## Testing

```bash
# Run all tests
poetry run pytest

# Run specific test
poetry run pytest tests/test_customer_service_agent_smoke.py

# Run with coverage
poetry run pytest --cov=src

# Run installation test
python test_installation.py

# Manual CLI testing
mlflow-eval-tools info
mlflow-eval-tools dataset-builder --help
mlflow-eval-tools agent-analysis --help
```

## Troubleshooting

```bash
# Check if package is installed
pip list | grep mlflow-eval-tools

# Check where CLI is installed
which mlflow-eval-tools  # Unix
where mlflow-eval-tools  # Windows

# Reinstall package
pip uninstall mlflow-eval-tools -y
pip install dist/mlflow_eval_tools-0.1.0-py3-none-any.whl

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Check imports
python -c "import mlflow_eval_tools; print(mlflow_eval_tools.__version__)"
python -c "from mlflow_eval_tools import cli; print('CLI imported OK')"

# Check environment
python -c "import os; print('OPENAI_API_KEY:', 'set' if os.getenv('OPENAI_API_KEY') else 'not set')"

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

## Poetry Commands

```bash
# Install dependencies
poetry install

# Add new dependency
poetry add click

# Add dev dependency
poetry add --group dev pytest

# Update dependencies
poetry update

# Show installed packages
poetry show

# Show outdated packages
poetry show --outdated

# Export requirements
poetry export -f requirements.txt -o requirements.txt

# Run command in virtual env
poetry run python script.py
poetry run pytest

# Activate shell
poetry shell
```

## Publishing (if using PyPI)

```bash
# Configure PyPI credentials
poetry config pypi-token.pypi your-token-here

# Test publish to Test PyPI
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry publish -r testpypi

# Publish to PyPI
poetry publish

# Or build and publish in one step
poetry publish --build
```

## Example Complete Workflow

```bash
# As a team member installing and using the package

# 1. Install
pip install mlflow_eval_tools-0.1.0-py3-none-any.whl

# 2. Set up environment
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Verify installation
mlflow-eval-tools info

# 4. Create dataset
mlflow-eval-tools dataset-builder \
  --agent-file src/my_agent.py \
  --agent-class MyAgent

# 5. Note the run ID from output (e.g., abc123)

# 6. Run evaluation
mlflow-eval-tools agent-analysis abc123 my_dataset_v1

# 7. View results
mlflow ui
# Open http://localhost:5000
```

## Useful Python Snippets

```python
# Check package version
import mlflow_eval_tools
print(mlflow_eval_tools.__version__)

# Import CLI programmatically
from mlflow_eval_tools.cli import cli
cli()

# Run CLI from Python
import sys
from mlflow_eval_tools.cli import cli
sys.argv = ['mlflow-eval-tools', 'info']
cli()

# Import dataset builder
from app_agents.dataset_builder import DatasetBuilderAgent

# Import agent analysis
from app_agents.agent_analysis import main as analysis_main
```

## File Locations

```
# Package source
src/mlflow_eval_tools/

# Built distributions
dist/

# MLflow tracking
mlruns/

# Checkpoints (if enabled)
data/checkpoints/

# Documentation
PACKAGE_README.md
docs/QUICK_START_TEAMS.md
BUILD_GUIDE.md
```

## Getting Help

```bash
# Command help
mlflow-eval-tools --help
mlflow-eval-tools dataset-builder --help
mlflow-eval-tools agent-analysis --help

# Package info
mlflow-eval-tools info

# Read documentation
cat PACKAGE_README.md
cat docs/QUICK_START_TEAMS.md

# Check issues
# Visit: https://github.com/sdeery14/llm-system-lifecycle/issues
```

## Environment Variables

```bash
# Required
export OPENAI_API_KEY=sk-...

# Optional
export MAX_DATASET_INSTANCES=100
export MLFLOW_TRACKING_URI=http://localhost:5000

# Check current values
mlflow-eval-tools info
```

---

Keep this file handy for quick reference!
