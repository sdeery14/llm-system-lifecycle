# MLflow Agent Lifecycle Demo

This script demonstrates the complete lifecycle of an AI agent using MLflow, following the official MLflow documentation for ResponsesAgent logging and serving.

## Overview

The `mlflow_agent_lifecycle.py` script demonstrates:

1. **Logging** - Log the customer service agent to MLflow using models-from-code approach
2. **Registration** - Register the model in the MLflow Model Registry
3. **Promotion** - Promote the model version to Production stage
4. **Loading** - Load the production model from MLflow
5. **Testing** - Test the loaded model with sample queries
6. **Interactive Chat** - Run an interactive chat session with the deployed model

## Based on MLflow Documentation

This implementation follows the official MLflow documentation:
- [ResponsesAgent for Model Serving](https://mlflow.org/docs/3.4.0/genai/serving/responses-agent/#logging-and-serving)

## Prerequisites

1. **Python Environment**: Python 3.12 or higher
2. **Dependencies**: All dependencies listed in `pyproject.toml` must be installed
3. **OpenAI API Key**: Set your OpenAI API key in a `.env` file or environment variable

```bash
# Install dependencies using Poetry
poetry install

# Or activate the virtual environment
poetry shell
```

## Setup

1. **Create a .env file** in the project root with your OpenAI API key:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

2. **Ensure all dependencies are installed** (already done if you followed prerequisites)

## Running the Script

### Full Lifecycle Demo

Run the complete lifecycle (log, register, promote, load, test, chat):

```bash
python scripts/mlflow_agent_lifecycle.py
```

This will:
- Create a `customer_service_responses_agent.py` file in the scripts directory
- Log the agent to MLflow (stored in `./mlruns` directory)
- Register the model in the Model Registry as "customer-service-agent"
- Promote the model to Production stage
- Load the production model
- Run test queries
- Prompt you to start an interactive chat session

## What Happens Step by Step

### Step 1: Logging Agent to MLflow

The script creates a MLflow-compatible wrapper (`CustomerServiceResponsesAgent`) that:
- Extends `mlflow.pyfunc.ResponsesAgent`
- Implements `predict()` for batch inference
- Implements `predict_stream()` for streaming responses
- Wraps the existing `CustomerServiceAgent`

The agent is logged using MLflow's **models-from-code** approach:

```python
mlflow.pyfunc.log_model(
    python_model="scripts/customer_service_responses_agent.py",
    artifact_path="agent",
    pip_requirements=[...],
    metadata={"task": "customer_service", "version": "1.0.0"},
)
```

### Step 2: Registering Model

The logged model is registered in the MLflow Model Registry:

```python
mlflow.register_model(
    model_uri=model_uri,
    name="customer-service-agent",
    tags={"task": "customer_service", "framework": "openai-agents"}
)
```

### Step 3: Promoting to Production

The registered model version is promoted to Production stage:

```python
client.transition_model_version_stage(
    name="customer-service-agent",
    version=version,
    stage="Production",
    archive_existing_versions=True
)
```

### Step 4: Loading Production Model

The production model is loaded from MLflow:

```python
loaded_model = mlflow.pyfunc.load_model("models:/customer-service-agent/Production")
```

### Step 5: Testing

The loaded model is tested with sample queries:

```python
result = loaded_model.predict({
    "input": [{"role": "user", "content": "Can you check the status of order ORD12345?"}],
    "context": {"conversation_id": "test-001", "user_id": "test-user"}
})
```

### Step 6: Interactive Chat

An interactive chat session is available where you can:
- Ask questions about orders, accounts, shipping, etc.
- The agent will use its tools (check order status, search knowledge base, etc.)
- Type 'quit', 'exit', or 'bye' to end the session

## File Structure

After running the script, you'll see:

```
llm-system-lifecycle/
├── scripts/
│   ├── mlflow_agent_lifecycle.py          # Main lifecycle script
│   ├── customer_service_responses_agent.py # Generated agent wrapper (auto-created)
│   └── README_MLFLOW_LIFECYCLE.md         # This file
├── mlruns/                                # MLflow tracking data (auto-created)
│   └── ...                                # Experiment runs, models, artifacts
└── ...
```

## MLflow UI

To view the logged experiments, models, and artifacts in the MLflow UI:

```bash
mlflow ui
```

Then open your browser to `http://localhost:5000`

In the UI, you can:
- View experiment runs and parameters
- Inspect logged models and artifacts
- Compare different model versions
- View model registry and stage transitions
- Explore traces and debugging information

## Model Serving (Optional)

To serve the production model as a REST API:

```bash
mlflow models serve -m models:/customer-service-agent/Production -p 5000
```

Test the served model:

```python
import requests

response = requests.post(
    "http://localhost:5000/invocations",
    json={
        "input": [{"role": "user", "content": "How long does shipping take?"}],
        "context": {"conversation_id": "api-test", "user_id": "test"}
    }
)

print(response.json())
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Not Found**
   - Ensure you've created a `.env` file with `OPENAI_API_KEY=your-key`
   - Or set the environment variable: `export OPENAI_API_KEY=your-key`

2. **Import Errors**
   - Ensure all dependencies are installed: `poetry install`
   - Activate the virtual environment: `poetry shell`

3. **MLflow Tracking Issues**
   - The script uses a local file-based tracking store (`./mlruns`)
   - Ensure you have write permissions in the project directory

4. **Model Loading Errors**
   - Make sure you've completed all previous steps (log, register, promote)
   - Check that the model exists in the registry: `mlflow models list`

### Enable Tracing

MLflow tracing is automatically enabled via the `@mlflow.trace` decorator. To view traces:

1. Run the script
2. Open MLflow UI: `mlflow ui`
3. Navigate to the experiment run
4. Click on "Traces" tab to see detailed execution traces

## Architecture

The script follows MLflow's ResponsesAgent pattern:

```
User Input
    ↓
CustomerServiceResponsesAgent (MLflow wrapper)
    ↓
CustomerServiceAgent (OpenAI Agents SDK)
    ↓
Agent Tools (check_order_status, get_account_balance, etc.)
    ↓
Response (ResponsesAgentResponse format)
    ↓
User Output
```

## Key Features

- ✅ **Framework-agnostic**: Works with OpenAI Agents SDK
- ✅ **MLflow integration**: Full lifecycle management
- ✅ **Model versioning**: Track multiple versions in registry
- ✅ **Stage management**: Promote models to Production
- ✅ **Tracing enabled**: Debug and monitor agent execution
- ✅ **OpenAI API compatible**: Uses standard chat completion format
- ✅ **Streaming support**: Both batch and streaming inference
- ✅ **Interactive testing**: Chat interface for manual testing

## Next Steps

1. **Customize the agent**: Modify `customer_service_agent.py` to add more tools or change behavior
2. **Deploy to production**: Use MLflow's deployment options (Docker, Kubernetes, cloud platforms)
3. **Monitor performance**: Use MLflow tracking to monitor agent performance over time
4. **A/B testing**: Register multiple versions and compare their performance
5. **Integration**: Integrate with your application using the MLflow REST API

## References

- [MLflow ResponsesAgent Documentation](https://mlflow.org/docs/3.4.0/genai/serving/responses-agent/)
- [MLflow Models from Code](https://mlflow.org/docs/3.4.0/ml/model/models-from-code/)
- [MLflow Model Registry](https://mlflow.org/docs/3.4.0/model-registry/)
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
