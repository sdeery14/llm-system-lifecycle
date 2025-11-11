# Structured Output Generation Update

## Overview

The dataset builder agent has been updated to use **OpenAI Agents SDK's Pydantic-defined output types** for generating test case batches. This ensures reliable, schema-compliant JSON generation using [structured outputs](https://platform.openai.com/docs/guides/structured-outputs).

## What Changed?

### Before: Manual JSON Parsing
Previously, the worker agent would generate free-form text that needed to be manually parsed:

```python
# Old approach - prone to parsing errors
worker = Agent(
    name="TestCaseWorker",
    model=worker_model,
    instructions=worker_instructions
)
result = await Runner.run(worker, "Generate test cases...")
# Had to manually parse JSON from text response
```

### After: Structured Outputs with Pydantic
Now, the worker agent uses Pydantic models to guarantee proper structure:

```python
# New approach - guaranteed schema compliance
worker = Agent(
    name="TestCaseWorker",
    model=worker_model,
    instructions=worker_instructions,
    output_type=TestCaseBatch  # Pydantic model defines exact structure
)
result = await Runner.run(worker, "Generate test cases...")
test_case_batch = result.final_output  # Already validated Pydantic object
```

## New Pydantic Models

### TestCaseInputs
Defines the structure of test case inputs:
```python
class TestCaseInputs(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    
    query: str  # User query or input (required)
    context: str = ""  # Optional context (defaults to empty)
    category: str  # Category name (required)
    test_scenario: str  # What this tests (required)
```

### TestCaseExpectations
Defines expected outputs/behaviors:
```python
class TestCaseExpectations(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    
    answer: str = ""  # Expected response (defaults to empty)
    tool_calls: list[str] = []  # Expected tools to call (defaults to empty list)
```

**Note on Strict Schema**: All Pydantic models use `strict=True` in their config, which is required by OpenAI's structured outputs. This means:
- No `Optional[T]` or `T | None` types - use default values instead
- All fields must be JSON-serializable
- Use `default_factory=list` for list fields instead of `None`

### TestCase
Complete test case structure:
```python
class TestCase(BaseModel):
    inputs: TestCaseInputs
    expectations: TestCaseExpectations
```

### TestCaseBatch
Batch of test cases (what the worker returns):
```python
class TestCaseBatch(BaseModel):
    test_cases: list[TestCase]
```

## Benefits

### 1. **Schema Compliance**
The LLM is constrained to produce exactly the structure you define - no more parsing errors or malformed JSON.

### 2. **Type Safety**
Pydantic validates all fields at runtime, catching issues immediately.

### 3. **Better LLM Performance**
Structured outputs use OpenAI's JSON mode, which produces more reliable results than prompting alone.

### 4. **MLflow Compatibility**
Output format directly matches MLflow's dataset requirements (inputs + expectations).

### 5. **Cleaner Code**
No more manual JSON parsing logic - the SDK handles it all.

## Example Output

When the worker agent runs with `output_type=TestCaseBatch`, it returns:

```python
TestCaseBatch(
    test_cases=[
        TestCase(
            inputs=TestCaseInputs(
                query="Where is my order #12345?",
                context="",
                category="order_status",
                test_scenario="Basic order status inquiry"
            ),
            expectations=TestCaseExpectations(
                answer="Provide order status information",
                tool_calls=["get_order_status"]
            )
        ),
        # ... more test cases
    ]
)
```

This is then easily converted to MLflow format:

```python
case_dict = {
    "inputs": test_case.inputs.model_dump(),
    "expectations": test_case.expectations.model_dump(exclude_none=True)
}
```

## Testing

Run the test script to see structured outputs in action:

```powershell
poetry run python scripts/test_structured_output_generation.py
```

This will:
1. Create a worker agent with structured output type
2. Generate 3 test cases for the "order_status" category
3. Display the structured output
4. Show the MLflow-compatible format

## References

- [OpenAI Agents SDK - Output Types](https://openai.github.io/openai-agents-python/agents/#output-types)
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## Implementation Details

The key change is in `_generate_batch_with_worker()` function in `src/app_agents/dataset_builder.py`:

```python
# Create worker with output_type
worker = Agent(
    name=f"TestCaseWorker_{category['name']}_{batch_index}",
    model=worker_model,
    instructions=worker_instructions,
    output_type=TestCaseBatch  # 🎯 This is the magic!
)

# Run and get structured output
result = await Runner.run(worker, ...)
test_case_batch = result.final_output  # Already a TestCaseBatch object

# Convert to dicts for MLflow
for test_case in test_case_batch.test_cases:
    case_dict = {
        "inputs": test_case.inputs.model_dump(),
        "expectations": test_case.expectations.model_dump(exclude_none=True)
    }
```

## Migration Notes

- **Backwards Compatible**: Existing datasets are unaffected
- **No Breaking Changes**: The external API remains the same
- **Better Reliability**: Reduces edge cases and parsing errors
- **Production Ready**: Uses OpenAI's official structured outputs feature
