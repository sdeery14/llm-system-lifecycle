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

- **[PACKAGE_README.md](PACKAGE_README.md)** - Complete package documentation
- **[QUICK_START_TEAMS.md](docs/QUICK_START_TEAMS.md)** - Quick start guide for teams
- **[BUILD_GUIDE.md](BUILD_GUIDE.md)** - Build and distribution guide
- **[PACKAGE_SUMMARY.md](PACKAGE_SUMMARY.md)** - Package overview

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
git clone https://github.com/sdeery14/llm-system-lifecycle.git
cd llm-system-lifecycle
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

- **Repository**: https://github.com/sdeery14/llm-system-lifecycle
- **Issues**: https://github.com/sdeery14/llm-system-lifecycle/issues

---

# Original Project: LLM System Lifecycle

Below is information about the original research project this package was extracted from.

## Project Overview

This project demonstrates a complete lifecycle for LLM-based agents, including:

- **Agent Development**: Building agents with the OpenAI Agents SDK
- **MLflow Integration**: Logging, versioning, and serving agents
- **Dataset Creation**: Automated evaluation dataset generation
- **Testing & Evaluation**: Comprehensive testing frameworks

## Key Components

### 1. Customer Service Agent (`src/dev_agents/customer_service_agent.py`)

A production-ready customer service agent built with the OpenAI Agents SDK that handles:
- Order status inquiries
- Account balance retrieval
- Refund processing
- Knowledge base search (using FAISS vector store)
- Customer contact information updates

### 2. Dataset Builder Agent (`src/app_agents/dataset_builder.py`)

An intelligent agent that helps you create high-quality evaluation datasets:
- Logs target agents in MLflow
- Collaboratively designs test categories with users
- Generates test cases in batches (max 20 at a time)
- Tracks progress to ensure exact variance as planned
- Stores datasets in MLflow for evaluation

**See [docs/dataset_builder.md](docs/dataset_builder.md) for detailed documentation.**

### 3. MLflow Integration Scripts

Scripts demonstrating the complete MLflow lifecycle:
- `scripts/mlflow_agent_lifecycle.py` - Log, register, promote, and serve agents
- `scripts/mlflow_dataset_example.py` - Create and manage evaluation datasets

## Quick Start

### Installation

```bash
# Install dependencies
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run the Customer Service Agent

```bash
python src/dev_agents/customer_service_agent.py
```

### Create an Evaluation Dataset

```bash
python scripts/dataset_builder_demo.py
```

Choose option 2 for interactive mode and follow the agent's guidance to create a dataset.

### Run MLflow Agent Lifecycle

```bash
python scripts/mlflow_agent_lifecycle.py
```

This demonstrates logging, registering, promoting, and serving an agent with MLflow.

## Configuration

Set these environment variables in your `.env` file:

```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional
MAX_DATASET_INSTANCES=100  # Maximum dataset size for dataset builder
```

## Project Structure

```
llm-system-lifecycle/
├── src/
│   ├── dev_agents/           # Development agents
│   │   └── customer_service_agent.py
│   └── app_agents/           # Application agents
│       └── dataset_builder.py
├── scripts/
│   ├── mlflow_agent_lifecycle.py
│   ├── mlflow_dataset_example.py
│   ├── dataset_builder_demo.py
│   ├── logged_agents/        # MLflow-compatible agent files
│   └── templates/            # Agent wrapper templates
├── tests/
│   └── test_customer_service_agent_smoke.py
├── docs/
│   ├── architecture.md
│   └── dataset_builder.md
└── pyproject.toml
```

## Documentation

- [Dataset Builder Guide](docs/dataset_builder.md) - Complete guide to the dataset builder agent
- [Architecture](docs/architecture.md) - System architecture overview

## Features

### Agent Development
- Production-ready agent examples
- Best practices for tool design
- State management patterns
- Vector store integration (FAISS)

### MLflow Integration
- Model logging and versioning
- Model registry management
- Production promotion workflows
- Dataset tracking and management

### Dataset Creation
- Automated test case generation
- Category-based organization
- Batch processing for large datasets
- Progress tracking and variance control
- MLflow integration for storage

### Testing
- Smoke tests for agents
- Evaluation dataset management
- Comprehensive test coverage

## Usage Examples

### Creating a Dataset for Your Agent

```python
import asyncio
from app_agents.dataset_builder import DatasetBuilderAgent
from agents import Runner

async def create_dataset():
    builder = DatasetBuilderAgent()
    agent = builder.get_agent()
    
    query = """
    I want to create an evaluation dataset for my agent.
    Agent file: src/dev_agents/customer_service_agent.py
    Class: CustomerServiceAgent
    Description: Handles customer service inquiries
    """
    
    result = await Runner.run(agent, query)
    print(result.final_output)

asyncio.run(create_dataset())
```

### Loading and Using Datasets

```python
import mlflow
from mlflow.genai.datasets import load_dataset

# Load a dataset
dataset = load_dataset("customer_service_eval_v1")

# Iterate over test cases
for test_case in dataset:
    print(test_case["inputs"])
    print(test_case["expectations"])
```

## Development

### Running Tests

```bash
poetry run pytest
```

### Adding New Agents

1. Create your agent in `src/dev_agents/` or `src/app_agents/`
2. Follow the pattern in `customer_service_agent.py`
3. Add tests in `tests/`
4. Use the dataset builder to create evaluation datasets

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

See LICENSE file for details.
