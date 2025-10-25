# Quick Start: MLflow Agent Lifecycle

## TL;DR

Run the complete MLflow lifecycle (log, register, promote, load, chat) with one command:

```bash
python scripts/mlflow_agent_lifecycle.py
```

## What You Need

1. **OpenAI API Key** - Set in `.env` file:
   ```
   OPENAI_API_KEY=your-key-here
   ```

2. **Python Environment** - Install dependencies:
   ```bash
   poetry install
   ```

## What This Does

The script will:

1. ✅ **Log** the customer service agent to MLflow
2. ✅ **Register** it in the Model Registry as "customer-service-agent"  
3. ✅ **Promote** the model to Production stage
4. ✅ **Load** the production model from the registry
5. ✅ **Test** with sample queries
6. ✅ **Chat** interactively (optional)

## View Results

Open MLflow UI to see logged models, experiments, and traces:

```bash
mlflow ui
```

Navigate to http://localhost:5000

## Serve the Model

Serve as REST API:

```bash
mlflow models serve -m models:/customer-service-agent/Production -p 5000
```

Test the API:

```python
import requests

response = requests.post(
    "http://localhost:5000/invocations",
    json={
        "input": [{"role": "user", "content": "How long does shipping take?"}]
    }
)

print(response.json())
```

## Files Created

- `scripts/customer_service_responses_agent.py` - MLflow-compatible agent wrapper
- `mlruns/` - MLflow tracking data and artifacts

## Next Steps

See `README_MLFLOW_LIFECYCLE.md` for full documentation.
