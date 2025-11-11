# Agent Analysis Tool

Comprehensive evaluation framework for LLM-based agents using MLflow.

## Quick Start

```python
from src.app_agents.agent_analysis import (
    load_agent,
    run_evaluation,
    analyze_evaluation_results,
    generate_analysis_report
)

# Load agent and run evaluation
agent_model = load_agent("agent_run_id")
predict_fn = create_predict_function(agent_model)

results = run_evaluation(
    dataset=your_dataset,
    predict_fn=predict_fn
)

# Analyze and report
analysis = analyze_evaluation_results(results, your_dataset)
report = generate_analysis_report(analysis, "AgentName", "dataset_name")
```

## Features

### ✅ Custom Scorers

- **exact_match** - Strict output validation
- **contains_expected_content** - Lenient content check
- **uses_correct_tools** ⭐ - Trace-based tool validation
- **tool_call_efficiency** - Optimize tool usage
- **response_quality** - Overall quality assessment

### ✅ Trace-Based Evaluation

Validates agent behavior using MLflow traces:
```python
{
    "expectations": {
        "answer": "Expected response",
        "tool_calls": ["tool1", "tool2"]  # ← Validated via traces
    }
}
```

### ✅ Comprehensive Reports

- Executive summary with key metrics
- Scorer performance breakdown
- Category-level analysis
- Tool usage statistics
- Failed test case details
- Actionable recommendations

### ✅ MLflow Integration

- Auto-logs metrics and artifacts
- Attaches reports to agent runs
- Tracks evaluation history

## Usage

### Command Line

```bash
python src/app_agents/agent_analysis.py <agent_run_id> <dataset_name>
```

### Interactive Demo

```bash
python scripts/agent_analysis_demo.py
```

### Programmatic

```python
import asyncio
from src.app_agents.agent_analysis import main

asyncio.run(main(
    agent_run_id="abc123",
    dataset_name="customer_service_demo"
))
```

## Files

- `src/app_agents/agent_analysis.py` - Main analysis tool
- `scripts/agent_analysis_demo.py` - Interactive demo
- `docs/dataset_builder/agent_evaluation_guide.md` - Full documentation

## Requirements

- MLflow >= 3.3
- OpenAI Agents SDK
- Dataset with `tool_calls` in expectations

## Example Report

```markdown
# Agent Evaluation Report

**Agent:** CustomerServiceAgent
**Total Test Cases:** 15
**Tool Usage Pass Rate:** 86.7%

## Performance by Category

| Category | Pass Rate |
|----------|-----------|
| order_status | 100.0% |
| refunds | 100.0% |
| knowledge_base | 66.7% |

## Recommendations

🟡 Category 'knowledge_base' has low pass rate
   → Add more training examples for knowledge_base scenarios
```

## Integration with Dataset Builder

Works seamlessly with datasets created by the Dataset Builder:

```python
# 1. Create dataset (includes tool_calls)
from src.app_agents.dataset_builder import DatasetBuilderAgent

# 2. Evaluate agent
from src.app_agents.agent_analysis import run_evaluation
```

## See Also

- [Dataset Builder](../README.md)
- [Evaluation with Tool Calls](evaluation_with_tool_calls.md)
- [MLflow Agent Evaluation Docs](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/agents/)
