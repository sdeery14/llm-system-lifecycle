# mlflow-eval-tools

> Evaluation tools for OpenAI Agents SDK projects

Build high-quality evaluation datasets and run LLM-judge analysis for agents built with the OpenAI Agents SDK.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

```bash
# Install
pip install mlflow_eval_tools-0.1.0-py3-none-any.whl

# Create dataset
mlflow-eval-tools dataset-builder

# Run evaluation
mlflow-eval-tools agent-analysis <agent_run_id> <dataset_name>

# View results
mlflow ui
```

## 📦 What's Included

- **Dataset Builder**: Interactive agent for creating evaluation datasets
- **Agent Analysis**: LLM-judge evaluation with 5 custom scorers
- **MLflow Integration**: Full tracking and versioning
- **Simple CLI**: Easy-to-use command-line interface

## 📚 Documentation

- **[Documentation Index](docs/README.md)** - Complete documentation guide
- **[Quick Start Guide](docs/QUICK_START_TEAMS.md)** - Get started quickly
- **[Package Reference](docs/PACKAGE_README.md)** - Complete package documentation
- **[Build Guide](docs/BUILD_GUIDE.md)** - Build and distribution guide
- **[Commands Reference](docs/COMMANDS.md)** - Quick command reference

## 💡 Features

### Dataset Builder
- ✅ Interactive conversation-based creation
- ✅ Automatic agent analysis
- ✅ Parallel batch generation
- ✅ Checkpointing & resumability
- ✅ Diversity validation

### Agent Analysis
- ✅ 5 custom scorers (exact match, content match, tool validation, efficiency, LLM-judge)
- ✅ Comprehensive reports
- ✅ Per-category breakdown
- ✅ Actionable recommendations

## 🔧 Installation

### For Teams (Recommended)

```bash
pip install mlflow_eval_tools-0.1.0-py3-none-any.whl
```

### From Source

```bash
git clone https://github.com/sdeery14/mlflow-eval-tools.git
cd mlflow-eval-tools
poetry install
```

## 📖 Usage

### Create an Evaluation Dataset

```bash
mlflow-eval-tools dataset-builder \
  --agent-file src/my_agents/support_agent.py \
  --agent-class SupportAgent
```

### Run Evaluation

```bash
mlflow-eval-tools agent-analysis abc123runid my_dataset_v1
```

### View Results

```bash
mlflow ui
# Open http://localhost:5000
```

## 🧪 Test Installation

```bash
python test_installation.py
```

## 📋 Requirements

- Python 3.12+
- OpenAI API key
- See [pyproject.toml](pyproject.toml) for full dependencies

## 🤝 Contributing

Contributions welcome! Please see the original project below for development setup.

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🔗 Links

- **Repository**: https://github.com/sdeery14/mlflow-eval-tools
- **Issues**: https://github.com/sdeery14/mlflow-eval-tools/issues
