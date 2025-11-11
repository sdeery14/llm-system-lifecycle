# Response Quality Scorer Update

## Overview

The `response_quality` scorer in the Agent Analysis tool has been updated to use an **OpenAI Agents SDK Agent** with **Pydantic output types** for intelligent, structured evaluation with numerical scoring (0-100) for granular comparison.

## What Changed

### Version 1: Heuristic-Based (Original)

The original implementation used basic text analysis:
- Length checks (too short, too long)
- Simple word overlap ratio
- Fixed scoring rules

**Limitations:**
- Could not understand semantic meaning
- Limited to word-matching heuristics
- No contextual understanding
- Fixed scoring thresholds

### Version 2: LLM Agent with JSON Parsing

The second version used an LLM agent but required manual JSON parsing and keyword extraction.

**Limitations:**
- Fragile JSON parsing with regex
- Fallback keyword extraction
- Categorical scores only (no numerical comparison)
- Complex error handling

### Version 3: Pydantic Output Types with Numerical Scoring (Current)

The new implementation uses:
- **Pydantic BaseModel** for structured output (`ResponseQualityEvaluation`)
- **Type-safe** agent responses (no JSON parsing needed)
- **Numerical scoring** (0-100) for granular comparison
- **Categorical labels** + rationale for interpretability

**Benefits:**
- ✅ Guaranteed structured output (Pydantic validation)
- ✅ No manual JSON parsing required
- ✅ Numerical scores enable precise ranking and comparison
- ✅ Type-safe and less error-prone
- ✅ Cleaner, more maintainable code

## Implementation Details

### Pydantic Output Model

```python
from pydantic import BaseModel, Field
from typing import Literal

class ResponseQualityEvaluation(BaseModel):
    """Structured output for response quality evaluation."""
    
    score: Literal["excellent", "good", "fair", "poor", "empty"] = Field(
        description="Overall quality rating of the response"
    )
    quality_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Numerical quality score from 0-100 for granular comparison"
    )
    rationale: str = Field(
        description="Brief explanation of the rating (1-2 sentences)"
    )
```

### Response Quality Evaluator Agent

```python
response_quality_agent = Agent(
    name="ResponseQualityEvaluator",
    model="gpt-4o-mini",
    instructions="""You are an expert evaluator of AI agent responses.

Your task is to evaluate the quality of an agent's response based on the expected answer.

Evaluation Criteria:
1. **Completeness**: Does the response address all aspects of the expected answer?
2. **Accuracy**: Is the information in the response correct and aligned with expectations?
3. **Clarity**: Is the response clear and easy to understand?
4. **Conciseness**: Is the response appropriately detailed without being unnecessarily verbose?
5. **Relevance**: Does the response stay on topic and address the query?

Rating Scale and Quality Scores:
- **excellent** (90-100): Response meets or exceeds all criteria
- **good** (70-89): Response meets most criteria with minor issues
- **fair** (50-69): Response partially meets criteria with notable gaps
- **poor** (10-49): Response has significant issues or missing content
- **empty** (0-9): Response is empty or completely missing

Provide both a categorical score and a precise numerical quality_score (0-100) for granular comparison.
Be objective and fair in your assessment. Focus on the substance of the response rather than style.""",
    output_type=ResponseQualityEvaluation  # ← Pydantic output type
)
```

### Simplified Scorer Function

```python
@scorer
def response_quality(outputs: str, expectations: dict) -> Feedback:
    """
    Evaluate the quality of the response using an OpenAI Agents SDK Agent.
    
    This scorer uses a specialized LLM agent with Pydantic output types to assess 
    response quality based on completeness, accuracy, clarity, conciseness, and relevance.
    """
    expected_answer = expectations.get("answer", "")
    
    # Quick check for empty responses
    if not outputs or len(str(outputs).strip()) == 0:
        return Feedback(
            value=0.0,  # Numerical score
            rationale="Empty or missing response."
        )
    
    output_str = str(outputs).strip()
    
    # Prepare evaluation prompt
    evaluation_prompt = f"""Please evaluate the following agent response:

**Expected Answer:** 
{expected_answer}

**Actual Response:**
{output_str}

Evaluate the response quality based on completeness, accuracy, clarity, conciseness, and relevance."""
    
    try:
        # Run the agent - returns ResponseQualityEvaluation Pydantic model
        result = loop.run_until_complete(
            Runner.run(response_quality_agent, evaluation_prompt)
        )
        
        # Extract structured evaluation (no JSON parsing needed!)
        evaluation: ResponseQualityEvaluation = result.final_output
        
        # Return Feedback with numerical score + categorical label
        return Feedback(
            value=evaluation.quality_score,  # 0-100 numerical score
            rationale=f"[{evaluation.score}] {evaluation.rationale}"
        )
    
    except Exception as e:
        # Fallback with numerical score
        return Feedback(
            value=50.0,
            rationale=f"Could not evaluate with agent: {str(e)}"
        )
```

## Scoring System

### Dual Scoring Approach

Each evaluation provides:

1. **Categorical Score**: Human-readable label (`excellent`, `good`, `fair`, `poor`, `empty`)
2. **Numerical Score**: Precise 0-100 value for comparison and ranking

### Scoring Scale

| Categorical | Numerical Range | Meaning |
|-------------|----------------|---------|
| `excellent` | 90-100 | Meets or exceeds all criteria |
| `good` | 70-89 | Meets most criteria with minor issues |
| `fair` | 50-69 | Partially meets criteria with notable gaps |
| `poor` | 10-49 | Significant issues or missing content |
| `empty` | 0-9 | No response provided |

### Why Numerical Scoring?

**Advantages:**

1. **Granular Comparison**: Distinguish between responses in the same category
   - Example: Score of 85 vs 72 (both "good" but different quality levels)

2. **Precise Ranking**: Sort and compare multiple responses objectively
   ```python
   sorted_results = sorted(results, key=lambda x: x['quality_score'], reverse=True)
   ```

3. **Statistical Analysis**: Calculate averages, distributions, trends
   ```python
   avg_score = sum(scores) / len(scores)
   improvement = new_score - baseline_score
   ```

4. **MLflow Metrics**: Log as numerical metrics for tracking over time
   ```python
   mlflow.log_metric("response_quality_score", evaluation.quality_score)
   ```

5. **Threshold-based Alerts**: Set quality gates
   ```python
   if quality_score < 60:
       alert("Response quality below acceptable threshold")
   ```

## Example Evaluations

### Test Case 1: Good Response (80/100)

**Expected Answer**: "Your order #12345 is currently being shipped."

**Actual Output**: "Your order #12345 has been shipped and will arrive in 2-3 business days. You can track it using tracking number ABC123."

**Evaluation:**
- **Quality Score**: 80.0/100
- **Categorical**: `good`
- **Rationale**: "The response is clear and provides useful information about the shipping status and tracking details, but it overstates the current status by stating the order 'has been shipped' instead of 'is currently being shipped', which slightly affects accuracy."

### Test Case 2: Poor Response (0/100)

**Expected Answer**: "Your refund for order #67890 has been processed and will appear in your account within 5-7 business days."

**Actual Output**: "OK"

**Evaluation:**
- **Quality Score**: 0.0/100
- **Categorical**: `empty`
- **Rationale**: "The actual response is completely inadequate as it does not address the query about the refund status at all, providing no relevant information or clarity."

### Test Case 3: Excellent Response (75/100)

**Expected Answer**: "Your account balance is $150.00."

**Actual Output**: "I've checked your account balance. Your current balance is $150.00. This includes your recent payment of $50.00 that was credited yesterday. If you have any questions about specific transactions, feel free to ask!"

**Evaluation:**
- **Quality Score**: 75.0/100
- **Categorical**: `good`
- **Rationale**: "The response is accurate and clear, providing the account balance as expected. However, it includes additional information about a recent transaction that, while relevant, is unnecessary for simply stating the balance, impacting conciseness."

### Granular Comparison Example

When evaluating multiple responses, numerical scores enable precise ranking:

```
Rank   Test Case                      Quality Score
--------------------------------------------------------------
1      Good Response                  80.0/100
2      Excellent Response             75.0/100
3      Poor Response - Too Short       0.0/100
4      Empty Response                  0.0/100
```

Notice how "Good Response" (80) ranks higher than "Excellent Response" (75) - the numerical scores reveal that despite the name, the first response was actually better quality in this case.

## Usage in Evaluation

The scorer is used as part of MLflow's evaluation framework:

```python
from app_agents.agent_analysis import (
    exact_match,
    contains_expected_content,
    uses_correct_tools,
    tool_call_efficiency,
    response_quality,  # ← Updated scorer
)

results = run_evaluation(
    dataset=dataset,
    predict_fn=predict_fn,
    scorers=[
        exact_match,
        contains_expected_content,
        uses_correct_tools,
        tool_call_efficiency,
        response_quality,  # Uses LLM agent for evaluation
    ],
    experiment_name="agent-evaluation"
)
```

## Error Handling

The scorer includes robust error handling with numerical fallback scores:

1. **Empty response check**: Immediately returns 0.0 score
2. **Pydantic validation**: Automatic type checking and validation
3. **Agent execution errors**: Falls back to numerical heuristics
4. **Event loop issues**: Creates new loop if needed

**Fallback behavior:**
```python
except Exception as e:
    # Fallback with numerical score if agent fails
    if len(output_str) < 10:
        quality_score = 20.0
        rationale = f"Response too short. Agent evaluation failed: {str(e)}"
    elif len(output_str) > 1000:
        quality_score = 50.0
        rationale = f"Response may be verbose. Agent evaluation failed: {str(e)}"
    else:
        quality_score = 50.0
        rationale = f"Could not evaluate with agent: {str(e)}"
    
    return Feedback(value=quality_score, rationale=rationale)
```

**Advantages over previous version:**
- No regex/JSON parsing errors (Pydantic handles it)
- Cleaner fallback logic with numerical scores
- Type-safe error recovery

## Testing

A test script is provided to verify the implementation:

```bash
poetry run python test_response_quality_scorer.py
```

This tests the scorer with various response types:
- Good responses
- Poor/short responses
- Empty responses
- Verbose responses

## Advantages

1. **Intelligent Evaluation**: Uses LLM understanding rather than simple patterns
2. **Structured Output**: Pydantic models guarantee type-safe, validated responses
3. **No JSON Parsing**: Direct access to structured data (no regex, no manual parsing)
4. **Numerical Scoring**: Enables granular comparison, ranking, and statistical analysis
5. **Contextual Assessment**: Evaluates semantic meaning, not just word matches
6. **Natural Language Rationale**: Provides human-readable explanations
7. **Flexible Criteria**: Can understand nuanced quality aspects
8. **Consistent with Modern Practices**: Uses the same agent framework as the system being evaluated
9. **Type Safety**: Pydantic validation prevents runtime errors
10. **Easier Maintenance**: Cleaner code with fewer edge cases to handle

## Performance Considerations

- **API Calls**: Each evaluation makes an LLM API call (GPT-4o-mini)
- **Latency**: ~1-2 seconds per evaluation (with Pydantic parsing)
- **Cost**: Uses GPT-4o-mini for cost-effectiveness (~$0.15 per 1M input tokens)
- **Accuracy**: Structured outputs reduce hallucination and parsing errors
- **Reliability**: Pydantic validation ensures consistent output format

## Comparison: Before vs After

| Aspect | Heuristic | JSON Agent | Pydantic Agent |
|--------|-----------|------------|----------------|
| Output Type | Categorical | Categorical | Categorical + Numerical |
| Parsing | N/A | Regex + JSON | Pydantic (automatic) |
| Type Safety | ❌ | ❌ | ✅ |
| Granular Comparison | ❌ | ❌ | ✅ (0-100 scale) |
| Error Handling | Simple | Complex | Clean |
| Code Complexity | Low | High | Medium |
| Semantic Understanding | ❌ | ✅ | ✅ |
| Statistical Analysis | ❌ | ❌ | ✅ |
| Ranking Capability | Limited | Limited | Precise |

## Future Enhancements

Potential improvements:
1. **Dimensional Scores**: Return separate numerical scores for each criterion (completeness, accuracy, clarity, etc.)
2. **Few-shot Examples**: Add example evaluations to agent instructions for calibration
3. **Custom Criteria**: Allow passing custom evaluation criteria per use case
4. **Batch Evaluation**: Evaluate multiple responses in one agent call
5. **Score Calibration**: Fine-tune scoring ranges with labeled evaluation data
6. **Caching**: Implement response caching for identical evaluations
7. **Confidence Scores**: Add confidence level to evaluations
8. **Trend Analysis**: Track quality scores over time with MLflow metrics
9. **A/B Testing**: Compare quality scores across different agent versions
10. **Multi-model Ensemble**: Use multiple LLMs and aggregate scores

## Migration Guide

If upgrading from the previous version:

### Update Import (if needed)
```python
# No changes needed - same import
from app_agents.agent_analysis import response_quality
```

### Handle Numerical Scores

**Before:**
```python
# Categorical score
if result.value == "good":
    print("Response is good")
```

**After:**
```python
# Numerical score with categorical in rationale
if result.value >= 70:  # "good" range
    print(f"Response quality: {result.value}/100")

# Or extract categorical from rationale
if "[good]" in result.rationale:
    print("Response is good")
```

### MLflow Metrics

**Before:**
```python
# Could only log categorical counts
mlflow.log_param("response_quality", "good")
```

**After:**
```python
# Log numerical metric for tracking
mlflow.log_metric("response_quality_score", result.value)

# Also log categorical for filtering
categorical = result.rationale.split("]")[0].strip("[")
mlflow.log_param("response_quality_category", categorical)
```

### Analysis Code

**Before:**
```python
# Count categorical scores
good_count = sum(1 for r in results if r.value == "good")
```

**After:**
```python
# Calculate statistics
avg_score = sum(r.value for r in results) / len(results)
max_score = max(r.value for r in results)
min_score = min(r.value for r in results)

# Sort by quality
sorted_results = sorted(results, key=lambda r: r.value, reverse=True)

# Filter by threshold
high_quality = [r for r in results if r.value >= 80]
```

## References

- [OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-python)
- [MLflow Evaluation Guide](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
- [MLflow Custom Scorers](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#using-custom-llm-evaluation-metrics)
