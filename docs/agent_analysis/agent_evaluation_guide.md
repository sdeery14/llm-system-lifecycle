# Agent Evaluation Guide

## Overview

The Agent Analysis tool provides a comprehensive framework for evaluating LLM-based agents using MLflow's evaluation capabilities. It implements custom scorers, generates detailed analysis reports, and integrates seamlessly with datasets created by the Dataset Builder.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Evaluation Flow                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  1. Load Agent from MLflow           │
        │     - Agent logged by Dataset Builder │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  2. Load Evaluation Dataset          │
        │     - Dataset created by Dataset     │
        │       Builder with tool_calls        │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  3. Run MLflow Evaluation            │
        │     - Custom Scorers:                │
        │       • exact_match                  │
        │       • contains_expected_content    │
        │       • uses_correct_tools (TRACE)   │
        │       • tool_call_efficiency         │
        │       • response_quality             │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  4. Analyze Results                  │
        │     - Compute statistics             │
        │     - Category breakdown             │
        │     - Tool usage analysis            │
        │     - Identify failures              │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  5. Generate Report                  │
        │     - Markdown/Text format           │
        │     - Summary stats                  │
        │     - Recommendations                │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  6. Save to MLflow                   │
        │     - Attach to agent run            │
        │     - Metrics logged                 │
        │     - Report as artifact             │
        └──────────────────────────────────────┘
```

## Custom Scorers

### 1. `exact_match`

Checks if the agent's output exactly matches the expected answer.

```python
@scorer
def exact_match(outputs: str, expectations: dict) -> bool:
    """
    Compare output to expected answer (case-insensitive).
    
    Returns:
        True if exact match, False otherwise.
    """
```

**Use Case:** Strict validation where the exact wording matters (e.g., order numbers, account balances).

### 2. `contains_expected_content`

More lenient scorer that checks if the output contains key elements from the expected answer.

```python
@scorer
def contains_expected_content(outputs: str, expectations: dict) -> bool:
    """
    Check if output contains at least 50% of expected words.
    
    Returns:
        True if sufficient overlap, False otherwise.
    """
```

**Use Case:** Flexible validation where the exact phrasing is less important.

### 3. `uses_correct_tools` ⭐ (Trace-Based)

Validates that the agent called the correct tools using MLflow's trace functionality.

```python
@scorer
def uses_correct_tools(trace: Trace, expectations: dict) -> Feedback:
    """
    Extract tools from trace and compare to expected tool_calls.
    
    Returns:
        Feedback with score ('yes'/'no') and detailed rationale.
    """
```

**Use Case:** Verify the agent's reasoning path and tool selection logic.

**Key Features:**
- ✅ Extracts actual tool calls from trace spans
- ✅ Compares to `expectations["tool_calls"]`
- ✅ Provides detailed rationale (missing/extra tools)
- ✅ Order-independent comparison

### 4. `tool_call_efficiency`

Evaluates whether the agent used an optimal number of tool calls.

```python
@scorer
def tool_call_efficiency(trace: Trace, expectations: dict) -> Feedback:
    """
    Check if number of tool calls matches expected count.
    
    Returns:
        Feedback with score ('optimal'/'over'/'under').
    """
```

**Use Case:** Identify inefficient agents that make unnecessary tool calls.

### 5. `response_quality`

Assesses the overall quality of the response based on completeness and relevance.

```python
@scorer
def response_quality(outputs: str, expectations: dict) -> Feedback:
    """
    Evaluate response quality (length, content overlap).
    
    Returns:
        Feedback with score ('good'/'fair'/'poor'/'verbose').
    """
```

**Use Case:** Catch edge cases like empty responses or overly verbose outputs.

## Usage

### Basic Usage

```python
from src.app_agents.agent_analysis import (
    load_agent,
    create_predict_function,
    run_evaluation,
    analyze_evaluation_results,
    generate_analysis_report,
    save_analysis_artifacts
)

# 1. Load agent from MLflow
agent_model = load_agent(agent_run_id="abc123")

# 2. Prepare dataset (from Dataset Builder)
dataset = [
    {
        "inputs": {"query": "What is order #12345's status?"},
        "expectations": {
            "answer": "Your order is shipped.",
            "tool_calls": ["check_order_status"]
        }
    },
    # ... more test cases
]

# 3. Create prediction function
predict_fn = create_predict_function(agent_model)

# 4. Run evaluation
results = run_evaluation(
    dataset=dataset,
    predict_fn=predict_fn,
    experiment_name="my-agent-evaluation"
)

# 5. Analyze results
analysis = analyze_evaluation_results(results, dataset)

# 6. Generate report
report = generate_analysis_report(
    analysis=analysis,
    agent_name="CustomerServiceAgent",
    dataset_name="customer_service_v1"
)

# 7. Save to MLflow
save_analysis_artifacts(
    analysis=analysis,
    report=report,
    agent_run_id="abc123"
)
```

### Command-Line Usage

```bash
# Run analysis on an agent with a dataset
python src/app_agents/agent_analysis.py <agent_run_id> <dataset_name>

# Example
python src/app_agents/agent_analysis.py abc123 customer_service_demo_dataset
```

### Interactive Demo

```bash
# Run the demo script for guided walkthrough
python scripts/agent_analysis_demo.py
```

## Analysis Report Structure

The generated report includes:

### 1. Executive Summary
- Total test cases evaluated
- Overall pass rates for key metrics
- Agent and dataset information

### 2. Scorer Performance
- Pass/fail rates for each scorer
- Distribution of categorical scores
- Detailed breakdowns

### 3. Performance by Category
- Table showing results per category
- Pass rates, failures, totals
- Sorted by performance

### 4. Tool Usage Analysis
- Accuracy rate per tool
- Expected vs actual usage counts
- Tool-specific insights

### 5. Failed Test Cases
- Top 10 failures with details
- Category, query, expected tools
- Failed scorers for each case

### 6. Recommendations
- High priority issues
- Medium priority improvements
- Actionable suggestions

### Example Report

```markdown
# Agent Evaluation Report

**Agent:** CustomerServiceAgent  
**Dataset:** customer_service_demo_dataset  
**Evaluation Date:** 2025-11-02 17:30:00

---

## Executive Summary

- **Total Test Cases:** 15
- **Tool Usage Pass Rate:** 86.7% (13/15)
- **Content Accuracy:** 93.3% (14/15)

## Scorer Performance

### uses_correct_tools

- **Pass Rate:** 86.7%
- **Passed:** 13 / 15
- **Failed:** 2 / 15

### contains_expected_content

- **Pass Rate:** 93.3%
- **Passed:** 14 / 15
- **Failed:** 1 / 15

## Performance by Category

| Category | Total | Passed | Failed | Pass Rate |
|----------|-------|--------|--------|-----------|
| order_status | 3 | 3 | 0 | 100.0% |
| account_info | 3 | 3 | 0 | 100.0% |
| knowledge_base | 3 | 2 | 1 | 66.7% |
| refunds | 3 | 3 | 0 | 100.0% |
| contact_updates | 3 | 2 | 1 | 66.7% |

## Tool Usage Analysis

| Tool | Expected | Correct | Incorrect | Accuracy |
|------|----------|---------|-----------|----------|
| check_order_status | 3 | 3 | 0 | 100.0% |
| get_account_balance | 3 | 3 | 0 | 100.0% |
| process_refund | 3 | 3 | 0 | 100.0% |
| search_knowledge_base | 3 | 2 | 1 | 66.7% |
| update_customer_contact | 3 | 2 | 1 | 66.7% |

## Recommendations

### 🟡 Medium Priority

- **category_performance:** Category 'knowledge_base' has low pass rate (66.7%)
  - *Suggestion:* Add more training examples or refine prompts for knowledge_base scenarios.

- **tool_performance:** Tool 'search_knowledge_base' has low accuracy (66.7%)
  - *Suggestion:* Review the implementation and documentation of the 'search_knowledge_base' tool.
```

## Integration with Dataset Builder

The Agent Analysis tool is designed to work seamlessly with datasets created by the Dataset Builder:

```python
# 1. Create dataset with Dataset Builder
from src.app_agents.dataset_builder import DatasetBuilderAgent

builder = DatasetBuilderAgent()
# ... interactive dataset creation with tool_calls included

# 2. Evaluate with Agent Analysis
from src.app_agents.agent_analysis import main as run_analysis

await run_analysis(
    agent_run_id="abc123",
    dataset_name="customer_service_demo_dataset"
)
```

## MLflow Integration

### Metrics Logged

The tool automatically logs metrics to MLflow:

- `eval_{scorer}_pass_rate`: Pass rate for each binary scorer
- `eval_category_{category}_pass_rate`: Pass rate per category
- Tool-specific accuracy metrics

### Artifacts Saved

- `evaluation_analysis/analysis.json`: Full analysis data
- `evaluation_analysis/report.md`: Formatted report

### Viewing Results

1. Open MLflow UI: `http://localhost:5000`
2. Navigate to the agent's run
3. View artifacts under "evaluation_analysis"
4. Check metrics tab for quick stats

## Best Practices

### 1. Create Comprehensive Datasets

Use the Dataset Builder to create datasets with:
- ✅ Diverse test cases covering all scenarios
- ✅ Expected tool calls for trace-based evaluation
- ✅ Clear, specific expectations

### 2. Run Regular Evaluations

- Evaluate after each agent modification
- Compare results across versions
- Track metrics over time in MLflow

### 3. Act on Recommendations

- Prioritize high-priority recommendations
- Address category-specific failures
- Improve tools with low accuracy

### 4. Iterate Based on Insights

- Use failed cases to create new training examples
- Refine prompts based on content accuracy
- Optimize tool selection based on trace analysis

## Advanced Usage

### Custom Scorers

Create your own scorers for domain-specific evaluation:

```python
from mlflow.genai import scorer
from mlflow.entities import Trace, Feedback

@scorer
def custom_domain_scorer(trace: Trace, expectations: dict) -> Feedback:
    """
    Custom scorer for your specific domain requirements.
    """
    # Your scoring logic here
    score = "yes" if condition else "no"
    rationale = "Detailed explanation..."
    
    return Feedback(value=score, rationale=rationale)

# Use in evaluation
results = run_evaluation(
    dataset=dataset,
    predict_fn=predict_fn,
    scorers=[
        uses_correct_tools,
        custom_domain_scorer,  # Your custom scorer
    ]
)
```

### Batch Evaluation

Evaluate multiple agents or datasets:

```python
agents = ["run_id_1", "run_id_2", "run_id_3"]
datasets = ["dataset_v1", "dataset_v2"]

for agent_id in agents:
    for dataset_name in datasets:
        await run_analysis(
            agent_run_id=agent_id,
            dataset_name=dataset_name
        )
```

### Programmatic Access to Results

```python
# Get detailed failure analysis
failures = analysis["failures"]

for failure in failures:
    print(f"Category: {failure['category']}")
    print(f"Query: {failure['query']}")
    print(f"Failed: {failure['failed_scorers']}")
    
    # Take corrective action
    # ...
```

## Troubleshooting

### Issue: Agent Loading Fails

**Solution:** Ensure the agent was logged using the Dataset Builder's `log_target_agent_in_mlflow` function, which creates MLflow-compatible models.

### Issue: Traces Not Available

**Solution:** Make sure MLflow tracing is enabled and the agent SDK supports it. OpenAI Agents SDK automatically creates traces.

### Issue: Dataset Format Mismatch

**Solution:** Verify your dataset has the correct structure:
```python
{
    "inputs": {"query": "..."},
    "expectations": {
        "answer": "...",
        "tool_calls": [...]
    }
}
```

## Summary

The Agent Analysis tool provides:

✅ **Comprehensive Evaluation** - Multiple scorers covering different aspects  
✅ **Trace-Based Validation** - Verify tool usage using MLflow traces  
✅ **Detailed Reports** - Actionable insights and recommendations  
✅ **MLflow Integration** - Seamless logging and tracking  
✅ **Dataset Builder Compatibility** - Works perfectly with created datasets

Use it to continuously improve your agents and ensure they meet quality standards!
