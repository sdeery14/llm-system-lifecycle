# Agent Analysis - Quick Start Guide

## Prerequisites

1. **MLflow Server Running**
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
   ```

2. **Agent Logged in MLflow**
   - Use the Dataset Builder to log your agent
   - Get the agent's run ID from MLflow UI

3. **Dataset Created**
   - Use the Dataset Builder to create a dataset
   - Dataset should include `tool_calls` in expectations
   - Dataset stored in MLflow's "evaluation-datasets" experiment

## Quick Start (5 Minutes)

### Option 1: Interactive Demo

```bash
# Run the interactive demo
poetry run python scripts/agent_analysis_demo.py

# Select option 1: "Evaluate with MLflow Dataset"
# Follow the prompts to select your agent and dataset
```

### Option 2: Command Line

```bash
# Get your agent run ID and dataset name from MLflow UI
poetry run python src/app_agents/agent_analysis.py <agent_run_id> <dataset_name>

# Example:
poetry run python src/app_agents/agent_analysis.py abc123def456 customer_service_demo_dataset
```

### Option 3: Programmatic

```python
import asyncio
from src.app_agents.agent_analysis import main

asyncio.run(main(
    agent_run_id="your_agent_run_id",
    dataset_name="your_dataset_name"
))
```

## What Gets Evaluated?

The analysis runs 5 custom scorers on your agent:

1. **exact_match** - Does output match expected answer exactly?
2. **contains_expected_content** - Does output contain key content?
3. **uses_correct_tools** ⭐ - Did agent call the right tools? (trace-based)
4. **tool_call_efficiency** - Did agent use optimal number of tools?
5. **response_quality** - Is the response well-formed?

## Output

### Console Output
```
Total Test Cases: 15
Tool Usage Pass Rate: 86.7%
Content Accuracy: 93.3%
```

### MLflow Artifacts
- `evaluation_analysis/analysis.json` - Full analysis data
- `evaluation_analysis/report.md` - Formatted report

### MLflow Metrics
- `eval_uses_correct_tools_pass_rate`
- `eval_contains_expected_content_pass_rate`
- `eval_category_{category}_pass_rate`
- And more...

## View Results

1. Open MLflow UI: http://localhost:5000
2. Navigate to your agent's run
3. Click on "Artifacts" tab
4. Open `evaluation_analysis/report.md`

## Example Workflow

```bash
# 1. Create dataset (if not already done)
poetry run python scripts/dataset_builder_demo.py
# Select "Automated Demo" to create customer_service_demo_dataset

# 2. Get the agent run ID from the output
# Example: Run ID: 48456247580c45a0abc6f17dda570e80

# 3. Run evaluation
poetry run python scripts/agent_analysis_demo.py

# 4. Select:
#    - Option 1: Evaluate with MLflow Dataset
#    - Choose your dataset
#    - Choose your agent

# 5. View results in MLflow UI
```

## Troubleshooting

### "Dataset not found"
- Ensure dataset was created using Dataset Builder
- Check it's in the "evaluation-datasets" experiment
- Verify dataset name matches exactly

### "Agent loading failed"
- Agent must be logged using Dataset Builder's `log_target_agent_in_mlflow`
- Verify agent run ID is correct
- Check agent is in "dataset-builder-targets" experiment

### "Tool calls not validated"
- Ensure your dataset includes `tool_calls` in expectations
- Check MLflow tracing is enabled (automatic with OpenAI Agents SDK)
- Verify agent actually calls tools

## Next Steps

- [Full Evaluation Guide](agent_evaluation_guide.md) - Detailed documentation
- [Custom Scorers](agent_evaluation_guide.md#custom-scorers) - Create your own
- [Analysis Reports](agent_evaluation_guide.md#analysis-report-structure) - Understand the output

## Common Commands

```bash
# List available datasets
mlflow experiments list

# List agents
mlflow runs list --experiment-id 1  # dataset-builder-targets

# View specific run
mlflow runs describe --run-id <run_id>
```

## Tips

💡 **Use descriptive dataset names** - Makes it easier to find later

💡 **Run evaluations regularly** - After each agent modification

💡 **Compare versions** - Evaluate multiple agent versions with same dataset

💡 **Act on recommendations** - The report includes actionable suggestions

## Support

- Check [docs/agent_analysis/](.) for detailed guides
- Review [MLflow Agent Evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/agents/)
- Inspect MLflow traces in UI for debugging
