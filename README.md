# LLM System Lifecycle

The purpose of this project is to provide a systematic approach to building and improving upon LLM models.

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
