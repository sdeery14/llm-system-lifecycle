# Quick Start Guide for Teams

This guide helps teams quickly get started with `mlflow-eval-tools` for evaluating their OpenAI Agents SDK projects.

## Installation

### Option 1: Install from Wheel (Recommended for Teams)

```bash
# Activate your virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install mlflow_eval_tools-0.1.0-py3-none-any.whl
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/sdeery14/mlflow-eval-tools.git
cd mlflow-eval-tools

# Install with poetry
poetry install

# Or with pip
pip install -e .
```

## Setup

### 1. Configure Environment

Create a `.env` file in your project root:

```bash
# Copy the example
cp .env.example .env

# Edit and add your OpenAI API key
OPENAI_API_KEY=sk-...
MAX_DATASET_INSTANCES=100
```

### 2. Verify Installation

```bash
mlflow-eval-tools info
```

You should see package information and environment status.

## Creating Your First Evaluation Dataset

### Step 1: Prepare Your Agent

Ensure your agent follows this structure:

```python
# my_agent.py
from agents import Agent, function_tool

@function_tool
def my_tool(param: str) -> str:
    """Tool description."""
    return f"Result: {param}"

class MyAgent:
    def __init__(self):
        self.agent = Agent(
            name="MyAgent",
            instructions="Your agent instructions...",
            tools=[my_tool],
            model="gpt-4o"
        )
    
    def get_agent(self):
        return self.agent
```

### Step 2: Run Dataset Builder

```bash
mlflow-eval-tools dataset-builder \
  --agent-file path/to/my_agent.py \
  --agent-class MyAgent
```

### Step 3: Follow Interactive Prompts

The agent will:
1. Analyze your agent's tools and capabilities
2. Ask about test categories
3. Generate test cases
4. Store the dataset in MLflow

Example conversation:

```
Agent: I've analyzed your MyAgent. It has 1 tool: my_tool.
       Let's create an evaluation dataset. What categories should we include?

You: I want 3 categories:
     - Basic queries: 30 test cases
     - Edge cases: 15 test cases  
     - Error handling: 5 test cases

Agent: Great! Creating a plan with 50 total test cases...
       [Generates and stores dataset]
```

### Step 4: Note the Dataset Name

The dataset will be stored with the name you provide (e.g., `my_agent_eval_v1`).

## Running Evaluation

### Step 1: Get Your Agent Run ID

After the dataset builder logs your agent, note the run ID from the output, or check MLflow UI:

```bash
mlflow ui
# Navigate to http://localhost:5000
# Find your agent in the "dataset-builder-targets" experiment
```

### Step 2: Run Analysis

```bash
mlflow-eval-tools agent-analysis <AGENT_RUN_ID> <DATASET_NAME>
```

Example:

```bash
mlflow-eval-tools agent-analysis abc123def456 my_agent_eval_v1
```

### Step 3: Review Results in MLflow UI

```bash
mlflow ui
```

Navigate to the "agent-evaluation" experiment to see:
- **Metrics**: Pass rates, accuracy scores
- **Artifacts**: Detailed analysis reports
- **Traces**: Tool call sequences
- **Per-category**: Performance breakdown

## Common Workflows

### Workflow 1: Iterative Development

```bash
# 1. Create initial dataset
mlflow-eval-tools dataset-builder --agent-file src/my_agent.py --agent-class MyAgent

# 2. Run evaluation
mlflow-eval-tools agent-analysis <RUN_ID> my_agent_v1

# 3. Review results, improve agent

# 4. Re-run evaluation on same dataset
mlflow-eval-tools agent-analysis <NEW_RUN_ID> my_agent_v1
```

### Workflow 2: Version Comparison

```bash
# Evaluate version 1
mlflow-eval-tools agent-analysis <V1_RUN_ID> my_agent_eval_v1

# Evaluate version 2  
mlflow-eval-tools agent-analysis <V2_RUN_ID> my_agent_eval_v1

# Compare in MLflow UI
```

### Workflow 3: Multiple Datasets

```bash
# Create regression test dataset
mlflow-eval-tools dataset-builder  # Create "regression_tests_v1"

# Create edge case dataset
mlflow-eval-tools dataset-builder  # Create "edge_cases_v1"

# Run both evaluations
mlflow-eval-tools agent-analysis <RUN_ID> regression_tests_v1
mlflow-eval-tools agent-analysis <RUN_ID> edge_cases_v1
```

### Workflow 4: Reuse Previous Agent

```bash
# List previously logged agents
mlflow-eval-tools dataset-builder --list-agents

# Create new dataset using existing agent
mlflow-eval-tools dataset-builder --use-previous <RUN_ID>
```

## Understanding Evaluation Results

### Scorers

The evaluation includes 5 scorers:

1. **exact_match**: Binary (pass/fail) - exact output match
2. **contains_expected_content**: Binary - 50%+ word overlap
3. **uses_correct_tools**: yes/no - correct tools called
4. **tool_call_efficiency**: optimal/under/over - efficiency rating
5. **response_quality**: 0-100 score - LLM-as-judge quality assessment

### Analysis Report

The report (in MLflow artifacts) includes:

- **Executive Summary**: Overall metrics
- **Scorer Performance**: Per-scorer statistics
- **Category Breakdown**: Performance by test category
- **Tool Analysis**: Tool usage accuracy
- **Failures**: Top failure cases
- **Recommendations**: Improvement suggestions

### Example Report Snippet

```markdown
## Executive Summary

- Total Test Cases: 50
- Tool Correctness: 92% (46/50)
- Content Match: 88% (44/50)

## Recommendations

1. Improve tool selection for "edge_cases" category (60% accuracy)
2. Consider adding validation for tool parameter "order_id"
3. Response quality could be improved for complex queries
```

## Advanced Usage

### Custom Batch Sizes

For large datasets, adjust batch size:

```bash
mlflow-eval-tools dataset-builder \
  --agent-file src/my_agent.py \
  --agent-class MyAgent \
  --max-size 500 \
  --batch-size 50
```

### Custom Models

Use different models for cost optimization:

```bash
mlflow-eval-tools dataset-builder \
  --agent-file src/my_agent.py \
  --agent-class MyAgent \
  --model gpt-4o \
  --worker-model gpt-4o-mini
```

### Custom Experiments

Organize evaluations by experiment:

```bash
mlflow-eval-tools agent-analysis <RUN_ID> <DATASET> \
  --dataset-experiment my-datasets \
  --eval-experiment my-team-evals
```

## Troubleshooting

### "Command not found"

Ensure the package is installed and your virtual environment is activated:

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip list | grep mlflow-eval-tools
```

### "OPENAI_API_KEY not set"

Set your API key:

```bash
# In .env file
OPENAI_API_KEY=sk-...

# Or export directly
export OPENAI_API_KEY=sk-...
```

### "Dataset not found"

Verify the dataset name exactly matches:

```bash
# Check MLflow UI experiments for dataset name
mlflow ui
```

### "Agent run ID not found"

Ensure the agent was logged. Check the dataset builder output or MLflow UI for the run ID.

## Best Practices

### 1. Version Your Datasets

Use descriptive names with versions:
- `customer_support_v1`
- `order_flow_regression_v2`
- `edge_cases_2024_01`

### 2. Start Small

Begin with 20-50 test cases, then expand:

```bash
# Initial dataset
mlflow-eval-tools dataset-builder --max-size 50

# After validation, create comprehensive dataset
mlflow-eval-tools dataset-builder --max-size 200
```

### 3. Organize by Category

Structure test cases by:
- Functionality (order status, refunds, etc.)
- Complexity (simple, moderate, complex)
- Edge cases (errors, boundaries, unusual inputs)

### 4. Regular Evaluation

Run evaluations on:
- Every major change
- Before releases
- Weekly/monthly regression tests

### 5. Track in CI/CD

Integrate into your pipeline:

```bash
# In CI/CD script
mlflow-eval-tools agent-analysis $AGENT_RUN_ID $DATASET_NAME
# Parse results and fail if below threshold
```

## Getting Help

- **Documentation**: See [PACKAGE_README.md](PACKAGE_README.md) for full details
- **Examples**: Check agent_analysis/ and dataset_builder/ folders for guides
- **Issues**: Report bugs on GitHub
- **MLflow UI**: Use `mlflow ui` to explore results

## Next Steps

1. Create your first dataset
2. Run an evaluation
3. Review results in MLflow UI
4. Iterate on your agent based on findings
5. Create additional datasets for comprehensive testing

Happy evaluating! 🚀
