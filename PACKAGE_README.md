# mlflow-eval-tools

> Evaluation tools for OpenAI Agents SDK projects

Build high-quality evaluation datasets and run LLM-judge analysis for agents built with the [OpenAI Agents SDK](https://github.com/openai/openai-agents-sdk).

## Features

- **📊 Dataset Builder**: Interactive agent that helps you create comprehensive evaluation datasets
  - Logs target agents in MLflow for versioning
  - Collaboratively designs test categories
  - Generates diverse test cases using structured outputs
  - Supports parallel batch generation for scalability
  - Automatic checkpointing for resumable dataset creation
  - Stores datasets in MLflow for reuse

- **🔍 Agent Analysis**: LLM-judge evaluation framework with custom scorers
  - Exact match scoring
  - Content similarity evaluation
  - Tool usage validation
  - Tool call efficiency analysis
  - Response quality assessment (LLM-as-judge)
  - Comprehensive analysis reports logged to MLflow

- **🚀 Easy CLI**: Simple command-line interface for all operations
- **🔄 MLflow Integration**: Native integration with MLflow for tracking and versioning
- **📦 Portable**: Distribute to other teams as a Python package

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/sdeery14/llm-system-lifecycle.git
cd llm-system-lifecycle

# Install with poetry
poetry install

# Or with pip (in a virtual environment)
pip install -e .
```

### From PyPI (Once Published)

```bash
pip install mlflow-eval-tools
```

## Quick Start

### 1. Set Up Environment

Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=your-openai-api-key
MAX_DATASET_INSTANCES=100  # Optional: maximum dataset size
```

### 2. Build an Evaluation Dataset

Run the interactive dataset builder:

```bash
mlflow-eval-tools dataset-builder
```

Or specify your agent upfront:

```bash
mlflow-eval-tools dataset-builder \
  --agent-file src/dev_agents/customer_service_agent.py \
  --agent-class CustomerServiceAgent
```

The agent will guide you through:
1. Analyzing your target agent
2. Defining test categories
3. Generating test cases
4. Storing the dataset in MLflow

### 3. Run Agent Analysis

Evaluate your agent using the created dataset:

```bash
mlflow-eval-tools agent-analysis <AGENT_RUN_ID> <DATASET_NAME>
```

Example:

```bash
mlflow-eval-tools agent-analysis abc123runid customer_service_eval_v1
```

### 4. View Results

Open MLflow UI to see evaluation results:

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view:
- Agent evaluation metrics
- Detailed analysis reports
- Tool usage statistics
- Per-category performance breakdown

## CLI Reference

### `mlflow-eval-tools dataset-builder`

Interactive dataset builder for creating evaluation datasets.

**Options:**

- `--agent-file PATH`: Path to the agent file (relative to project root)
- `--agent-class NAME`: Name of the agent class (e.g., 'CustomerServiceAgent')
- `--use-previous RUN_ID`: Use a previously logged agent from MLflow
- `--list-agents`: List previously logged agents from MLflow
- `--max-size N`: Maximum dataset size (overrides env var)
- `--batch-size N`: Batch size for test case generation (default: 20)
- `--model NAME`: LLM model for main agent (default: gpt-4o)
- `--worker-model NAME`: LLM model for worker agents (default: gpt-4o-mini)

**Examples:**

```bash
# Interactive mode (recommended)
mlflow-eval-tools dataset-builder

# Start with a specific agent
mlflow-eval-tools dataset-builder \
  --agent-file src/dev_agents/my_agent.py \
  --agent-class MyAgent

# Use a previously logged agent
mlflow-eval-tools dataset-builder --use-previous abc123runid

# List available agents
mlflow-eval-tools dataset-builder --list-agents

# Custom configuration
mlflow-eval-tools dataset-builder \
  --agent-file src/dev_agents/my_agent.py \
  --agent-class MyAgent \
  --max-size 200 \
  --batch-size 10 \
  --model gpt-4o \
  --worker-model gpt-4o-mini
```

### `mlflow-eval-tools agent-analysis`

Run LLM-judge analysis on an agent using an evaluation dataset.

**Usage:**

```bash
mlflow-eval-tools agent-analysis AGENT_RUN_ID DATASET_NAME [OPTIONS]
```

**Arguments:**

- `AGENT_RUN_ID`: MLflow run ID of the agent to evaluate
- `DATASET_NAME`: Name of the evaluation dataset in MLflow

**Options:**

- `--dataset-experiment NAME`: Experiment containing the dataset (default: evaluation-datasets)
- `--eval-experiment NAME`: Experiment to log evaluation results (default: agent-evaluation)
- `--no-artifacts`: Skip saving analysis artifacts to MLflow

**Examples:**

```bash
# Basic evaluation
mlflow-eval-tools agent-analysis abc123runid customer_service_eval_v1

# Custom experiment names
mlflow-eval-tools agent-analysis abc123runid my_dataset \
  --dataset-experiment my-datasets \
  --eval-experiment my-evaluations

# Skip artifact saving
mlflow-eval-tools agent-analysis abc123runid my_dataset --no-artifacts
```

### `mlflow-eval-tools info`

Display package and environment information.

```bash
mlflow-eval-tools info
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY` (required): Your OpenAI API key
- `MAX_DATASET_INSTANCES` (optional): Maximum dataset size (default: 100)
- `MLFLOW_TRACKING_URI` (optional): MLflow tracking server URI (default: local ./mlruns)

### Configuration File

The dataset builder supports advanced configuration through Python:

```python
from mlflow_eval_tools import DatasetBuilderConfig

config = DatasetBuilderConfig(
    max_dataset_size=200,
    batch_size=20,
    model="gpt-4o",
    worker_model="gpt-4o-mini",
    enable_parallel_generation=True,
    enable_checkpointing=True,
    checkpoint_interval=50,
    enable_diversity_check=True,
    diversity_window=20,
    incremental_mlflow_logging=True
)
```

## Evaluation Scorers

The agent analysis tool includes several built-in scorers:

### 1. **Exact Match**
Checks if the agent's output exactly matches the expected answer.

### 2. **Contains Expected Content**
More lenient scorer that checks if key elements from the expected answer appear in the output (50% word overlap threshold).

### 3. **Uses Correct Tools**
Validates that the agent called the expected tools by analyzing MLflow traces.

### 4. **Tool Call Efficiency**
Evaluates whether the agent used an optimal number of tool calls (not too many, not too few).

### 5. **Response Quality (LLM-as-Judge)**
Uses GPT-4o-mini to evaluate response quality based on:
- Completeness
- Accuracy
- Clarity
- Conciseness
- Relevance

Provides a 0-100 quality score with detailed rationale.

## Example Workflow

### Step 1: Create Your Agent

```python
# src/my_agents/support_agent.py
from agents import Agent, function_tool

@function_tool
def get_order_status(order_id: str) -> str:
    """Get the status of an order."""
    # Implementation...
    return f"Order {order_id} is shipped"

class SupportAgent:
    def __init__(self):
        self.agent = Agent(
            name="SupportAgent",
            instructions="You are a helpful support agent.",
            tools=[get_order_status],
            model="gpt-4o"
        )
    
    def get_agent(self):
        return self.agent
```

### Step 2: Build Evaluation Dataset

```bash
mlflow-eval-tools dataset-builder \
  --agent-file src/my_agents/support_agent.py \
  --agent-class SupportAgent
```

Chat with the agent to define your dataset:

```
Agent: I've analyzed your SupportAgent. It has 1 tool: get_order_status.
       What categories should we include in the dataset?

You: Let's create 3 categories:
     1. Order status queries (30 cases)
     2. Edge cases with invalid IDs (10 cases)
     3. Multiple order queries (10 cases)

Agent: Great! I'll create a plan with 50 total test cases.
       [The agent will generate test cases and store in MLflow]
```

### Step 3: Log Your Agent in MLflow

The dataset builder automatically logs your agent, or you can log it separately using the MLflow lifecycle scripts.

### Step 4: Run Evaluation

```bash
mlflow-eval-tools agent-analysis abc123runid support_agent_eval_v1
```

### Step 5: Review Results

Open MLflow UI:

```bash
mlflow ui
```

Review:
- Overall pass rates
- Per-category performance
- Tool usage accuracy
- Response quality scores
- Detailed failure analysis
- Recommendations for improvement

## For Package Distributors

### Building the Package

```bash
# Install build tools
poetry install

# Build distribution files
poetry build
```

This creates:
- `dist/mlflow_eval_tools-0.1.0-py3-none-any.whl`
- `dist/mlflow_eval_tools-0.1.0.tar.gz`

### Publishing to PyPI

```bash
# Configure PyPI credentials
poetry config pypi-token.pypi your-token-here

# Publish
poetry publish
```

### Internal Distribution

For internal distribution without PyPI:

```bash
# Build the package
poetry build

# Share the wheel file with other teams
# They can install with:
pip install mlflow_eval_tools-0.1.0-py3-none-any.whl
```

## Architecture

The package consists of two main components:

### Dataset Builder Agent
- Built with OpenAI Agents SDK
- Uses structured outputs (Pydantic models) for reliable test case generation
- Supports parallel batch generation with worker agents
- Implements checkpointing for resumable dataset creation
- Validates test case diversity to avoid duplicates
- Logs all artifacts to MLflow

### Agent Analysis Tool
- Uses MLflow's evaluation framework
- Custom scorers with trace analysis
- LLM-as-judge for quality assessment
- Generates comprehensive analysis reports
- Logs results and artifacts to MLflow

## Requirements

- Python 3.12+
- OpenAI API key
- Dependencies: mlflow, openai-agents, pydantic, click, python-dotenv, and others (see pyproject.toml)

## Troubleshooting

### "OPENAI_API_KEY not found"

Set your API key in a `.env` file or export it:

```bash
export OPENAI_API_KEY=your-key-here
```

### "Dataset not found in MLflow"

Ensure the dataset was created and finalized:

1. Check MLflow UI for the dataset experiment
2. Verify the dataset name matches exactly
3. Try listing datasets with `mlflow-eval-tools dataset-builder --list-agents`

### "Agent run ID not found"

The agent must be logged in MLflow first. Use the dataset builder to log it, or check MLflow UI for the correct run ID.

### Import Errors

Make sure the package is installed:

```bash
poetry install
# or
pip install -e .
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- GitHub Issues: https://github.com/sdeery14/llm-system-lifecycle/issues
- Documentation: See `docs/` folder for detailed guides

## Acknowledgments

Built with:
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-sdk)
- [MLflow](https://mlflow.org/)
- [Click](https://click.palletsprojects.com/)
- [Pydantic](https://docs.pydantic.dev/)
