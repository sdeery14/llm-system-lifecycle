# Dataset Builder Agent

A sophisticated OpenAI Agents SDK-based agent that helps you create high-quality evaluation datasets for your LLM agents.

## Overview

The Dataset Builder Agent automates the process of creating comprehensive test datasets for agent evaluation. It:

1. **Logs target agents** in MLflow for tracking and versioning
2. **Collaborates with users** to understand dataset requirements
3. **Creates structured plans** with categories and test case distributions
4. **Generates test cases** in batches to stay within LLM context limits
5. **Tracks progress** to ensure exact variance as planned
6. **Stores datasets** in MLflow for easy evaluation

## Features

### Intelligent Batch Processing
- Generates test cases in batches of up to 20 instances
- Prevents LLM context overflow for large datasets
- Ensures consistent quality across all batches

### State Management
- Tracks what has been created to avoid duplicates
- Maintains exact category distribution as planned
- Provides progress updates throughout the process

### MLflow Integration
- Logs target agents with full metadata
- Stores datasets with proper versioning
- Links datasets to specific agent versions

### Configurable Limits
- Maximum dataset size controlled by `MAX_DATASET_INSTANCES` environment variable
- Default limit: 100 instances (configurable)
- Batch size: 20 instances per generation

## Installation

Ensure you have the required dependencies:

```bash
poetry install
```

## Configuration

Set up your environment variables in a `.env` file:

```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional - defaults to 100
MAX_DATASET_INSTANCES=100
```

## Usage

### Interactive Mode

Run the agent interactively to create a dataset:

```bash
python scripts/dataset_builder_demo.py
```

Choose option 2 for interactive mode, then follow the agent's guidance.

### Automated Demo

See a full demonstration of the dataset creation workflow:

```bash
python scripts/dataset_builder_demo.py
```

Choose option 1 for the automated demo.

### Direct Python Usage

```python
import asyncio
from app_agents.dataset_builder import DatasetBuilderAgent
from agents import Runner

async def create_dataset():
    # Create the agent
    builder = DatasetBuilderAgent()
    agent = builder.get_agent()
    
    # Start the conversation
    query = """I want to create an evaluation dataset for my customer service agent.
    
    The agent is in: src/dev_agents/customer_service_agent.py
    Class name: CustomerServiceAgent
    Description: A customer service agent that handles inquiries."""
    
    result = await Runner.run(agent, query)
    print(result.final_output)

asyncio.run(create_dataset())
```

## Workflow

The agent follows a structured workflow:

### 1. Log Target Agent

First, the agent logs your target agent in MLflow:

```
User: I want to create a dataset for CustomerServiceAgent in 
      src/dev_agents/customer_service_agent.py

Agent: [Logs agent and confirms]
```

### 2. Discuss Requirements

The agent collaborates with you to understand:
- What aspects need testing
- Realistic scenarios and edge cases
- Expected behaviors
- Category distribution

```
User: I need to test order status queries, refunds, and knowledge base search.

Agent: [Asks clarifying questions about distribution, expectations, etc.]
```

### 3. Create Dataset Plan

The agent formalizes the plan:

```
Agent: 📋 Dataset Plan Created: customer_service_eval_v1

Total Test Cases: 50

Categories:
1. order_status (20 cases)
   Description: Test order status checking
2. refund_requests (15 cases)
   Description: Test refund processing
3. knowledge_base (15 cases)
   Description: Test information retrieval
```

### 4. Generate Test Cases

Once approved, the agent generates test cases in batches:

```
Agent: ✓ Generated 20 test cases for category 'order_status'.
       Progress: 20/50 (40.0%)
       
Agent: ✓ Generated 15 test cases for category 'refund_requests'.
       Progress: 35/50 (70.0%)
```

### 5. Store in MLflow

Finally, the complete dataset is stored in MLflow:

```
Agent: ✓ Successfully created and stored dataset in MLflow!

Dataset Details:
  Name: customer_service_eval_v1
  Total Test Cases: 50
  Target Agent: CustomerServiceAgent

Category Breakdown:
  - order_status: 20 cases
  - refund_requests: 15 cases
  - knowledge_base: 15 cases
```

## Dataset Structure

Generated datasets follow MLflow's format:

```python
{
    "inputs": {
        "category": "order_status",
        "case_number": 1,
        "description": "Test order status checking - Test case 1",
        "example_based_on": "Can you check order ORD12345?"
    },
    "expectations": {
        "contains_order_id": True,
        "includes_status": True,
        "tone": "helpful"
    },
    "metadata": {
        "category": "order_status",
        "generated_at": "2025-10-30T12:00:00",
        "batch_number": 1
    }
}
```

## Available Tools

The agent uses the following tools:

### `log_target_agent_in_mlflow`
Logs the target agent in MLflow with metadata and tracking.

### `create_dataset_plan`
Creates a structured plan with categories and distributions.

### `approve_dataset_plan`
Approves the plan and prepares for dataset generation.

### `generate_test_cases_batch`
Generates a batch of test cases (max 20 at a time).

### `finalize_and_store_dataset`
Stores the complete dataset in MLflow.

### `get_builder_state_summary`
Provides a summary of current progress.

### `reset_dataset_creation`
Resets dataset creation while keeping the plan.

## Architecture

```
┌─────────────────────────────────────────┐
│     Dataset Builder Agent (GPT-4o)      │
│  - Conversational interface             │
│  - Plan validation                      │
│  - Batch coordination                   │
└─────────────────────────────────────────┘
                   │
                   ├─────────────────────────┐
                   │                         │
         ┌─────────▼────────┐      ┌────────▼────────┐
         │   State Manager  │      │  MLflow Client  │
         │  - Track progress│      │  - Log agents   │
         │  - Avoid dupes   │      │  - Store data   │
         └──────────────────┘      └─────────────────┘
```

## Example: Customer Service Agent Dataset

Here's a complete example of creating a dataset for the customer service agent:

```bash
$ python scripts/dataset_builder_demo.py

Choose option 2 for interactive mode

You: I want to create a dataset for the CustomerServiceAgent in 
     src/dev_agents/customer_service_agent.py. It's a customer 
     service agent that handles orders, refunds, and inquiries.

Agent: Great! I'll log that agent in MLflow... [logs agent]
       Now, what categories would you like to test?

You: I want to test:
     - Order status queries (20 cases)
     - Refund processing (15 cases)  
     - Knowledge base search (15 cases)
     Total: 50 cases

Agent: [Creates plan and shows summary]

You: Yes, please create this dataset.

Agent: [Generates all test cases in batches and stores in MLflow]
       ✓ Dataset complete and stored!
```

## Tips for Best Results

1. **Be Specific**: Provide clear descriptions of what each category should test
2. **Include Edge Cases**: Mention scenarios like invalid IDs, errors, etc.
3. **Define Expectations**: Specify what makes a good response
4. **Start Small**: Try a small dataset (10-20 cases) first to validate
5. **Review the Plan**: Always review the plan before approving

## Integration with Evaluation

Once created, use the dataset to evaluate your agent:

```python
import mlflow
from mlflow.genai.datasets import load_dataset

# Load the dataset
dataset = load_dataset("customer_service_eval_v1")

# Use with your evaluation framework
for test_case in dataset:
    inputs = test_case["inputs"]
    expectations = test_case["expectations"]
    # Run your agent and evaluate against expectations
```

## Troubleshooting

### Dataset Too Large Error

```
Error: Total instances (200) exceeds maximum allowed (100)
```

**Solution**: Increase `MAX_DATASET_INSTANCES` in your `.env` file or reduce dataset size.

### Plan Not Approved Error

```
Error: Dataset plan not approved. Please approve the plan first.
```

**Solution**: Make sure to explicitly approve the plan when the agent asks.

### MLflow Connection Issues

**Solution**: Check your MLflow tracking URI:
```python
import mlflow
print(mlflow.get_tracking_uri())
```

## Contributing

To extend the dataset builder agent:

1. Add new tools in `src/app_agents/dataset_builder.py`
2. Update the agent's instructions to use new tools
3. Test with the demo script

## License

See LICENSE file in the project root.
