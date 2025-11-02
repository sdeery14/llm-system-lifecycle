# Scalable Dataset Generation

## Overview

The Dataset Builder Agent now includes advanced features for generating large-scale evaluation datasets without overwhelming the agent's context window. This document explains the new scalability features and how to use them.

## Key Features

### 1. **Hierarchical Worker Pattern**

Instead of having a single agent generate all test cases, the system uses a hierarchical approach:

- **Main Agent**: Coordinates the workflow and manages state
- **Worker Agents**: Generate batches of test cases with minimal context
- **Fresh Context**: Each worker starts with a clean slate, preventing context window overflow

**Benefits:**
- Scales to thousands of test cases
- Uses cheaper models (gpt-4o-mini) for workers
- Parallel generation for speed
- No context accumulation issues

**Configuration:**
```python
config = DatasetBuilderConfig(
    enable_parallel_generation=True,  # Enable parallel workers
    worker_model="gpt-4o-mini",       # Use cheaper model for workers
    batch_size=20                      # Max cases per batch
)
```

### 2. **Automatic Checkpointing**

Dataset generation can be interrupted and resumed from checkpoints:

**Features:**
- Automatic saves every N test cases (configurable)
- Stores complete state including generated cases
- Resume from exact point of interruption
- Checkpoint files include timestamps and metadata

**Configuration:**
```python
config = DatasetBuilderConfig(
    enable_checkpointing=True,        # Enable auto-checkpointing
    checkpoint_interval=50,            # Save every 50 cases
    checkpoint_dir=Path("data/checkpoints")  # Where to store checkpoints
)
```

**Usage:**
```python
# Resume from checkpoint
await load_from_checkpoint(checkpoint_path="data/checkpoints/my_dataset_checkpoint.json")

# Or let the agent find it automatically
await load_from_checkpoint()  # Uses current dataset name
```

**Checkpoint Format:**
```json
{
  "dataset_name": "customer_service_regression_v1",
  "instances_created": 150,
  "total_instances_planned": 500,
  "created_instances": [...],
  "dataset_plan": {...},
  "timestamp": "2025-11-02T14:30:00"
}
```

### 3. **Diversity Enforcement**

Prevents duplicate or overly similar test cases:

**Features:**
- Structural hashing to detect similar patterns
- Sliding window comparison (checks last N cases)
- Automatic regeneration of duplicates
- Tracks rejection count for quality monitoring

**Configuration:**
```python
config = DatasetBuilderConfig(
    enable_diversity_check=True,      # Enable diversity checking
    diversity_window=20                # Check against last 20 cases
)
```

**How It Works:**
1. Each test case gets a structural hash based on its input pattern
2. New cases are compared against recent cases (within the diversity window)
3. If a duplicate is detected, it's rejected and regenerated
4. System tracks rejection count for monitoring

### 4. **Parallel Batch Generation**

For large batches, test cases are generated in parallel:

**Features:**
- Splits large batches into sub-batches
- Runs multiple workers concurrently using asyncio
- Automatically merges results
- Configurable sub-batch size

**Example:**
```python
# Generate 100 cases for a category
# System automatically:
# 1. Splits into 10 batches of 10 cases
# 2. Runs workers in parallel
# 3. Merges results
# 4. Applies diversity checking
# 5. Saves checkpoint if threshold reached
```

### 5. **Incremental MLflow Logging** (Planned)

Future enhancement to log batches as they're created:

**Benefits:**
- Dataset visible in MLflow during generation
- Can resume from MLflow if checkpoint lost
- Real-time progress monitoring
- Reduced memory footprint

**Configuration:**
```python
config = DatasetBuilderConfig(
    incremental_mlflow_logging=True   # Log batches incrementally
)
```

## Complete Workflow

### Basic Usage

```python
from src.app_agents.dataset_builder import DatasetBuilderAgent

# Create agent with default scalable configuration
agent = DatasetBuilderAgent(model="gpt-4o")

# The agent handles everything automatically:
# - Uses worker agents for generation
# - Applies diversity checks
# - Saves checkpoints
# - Runs batches in parallel
```

### Custom Configuration

```python
from src.app_agents.dataset_builder import DatasetBuilderAgent, DatasetBuilderConfig

# Create custom config
config = DatasetBuilderConfig(
    max_dataset_size=1000,
    batch_size=20,
    worker_model="gpt-4o-mini",
    enable_parallel_generation=True,
    enable_checkpointing=True,
    checkpoint_interval=100,
    enable_diversity_check=True,
    diversity_window=30
)

# Agent will use these settings
agent = DatasetBuilderAgent(model="gpt-4o")
```

### Environment Variables

Configure via `.env` file:

```bash
MAX_DATASET_INSTANCES=1000
OPENAI_API_KEY=your-key-here
```

## Architecture Patterns

### Pattern 1: Hierarchical Generation

```
Main Agent (gpt-4o)
  ├── Coordinates workflow
  ├── Manages state
  └── Spawns workers
       │
       ├── Worker 1 (gpt-4o-mini) → Batch 1 (20 cases)
       ├── Worker 2 (gpt-4o-mini) → Batch 2 (20 cases)
       └── Worker 3 (gpt-4o-mini) → Batch 3 (20 cases)
                                       ↓
                                  Parallel execution
                                       ↓
                                  Merge results
                                       ↓
                                Diversity check
                                       ↓
                               Save checkpoint
```

### Pattern 2: Checkpointing Flow

```
Start Generation
       ↓
Generate Batch (20 cases)
       ↓
Check: instances_created - last_checkpoint >= interval?
       ├── Yes → Save checkpoint
       └── No  → Continue
       ↓
More batches needed?
       ├── Yes → Generate next batch
       └── No  → Finalize dataset
```

### Pattern 3: Diversity Check

```
Worker generates test case
       ↓
Compute hash of case structure
       ↓
Compare to recent_case_hashes (sliding window)
       ├── Duplicate found → Reject & regenerate
       └── Unique → Accept
       ↓
Add to recent_case_hashes
       ↓
Keep window size ≤ diversity_window
```

## Performance Considerations

### Context Window Management

| Approach | Context Size | Scalability |
|----------|-------------|-------------|
| Single agent generates all | O(n) growing | Limited to ~1000 cases |
| Worker per batch | O(1) constant | Unlimited |

### Cost Optimization

- **Main Agent**: Uses gpt-4o for coordination (~5-10 calls)
- **Workers**: Use gpt-4o-mini for generation (~50-100 calls)
- **Savings**: ~80% cost reduction vs. all gpt-4o

### Speed Optimization

- **Sequential**: 100 cases in ~5 minutes
- **Parallel (5 workers)**: 100 cases in ~1 minute
- **Speedup**: ~5x with default settings

## Best Practices

### 1. Choose Appropriate Batch Sizes

- Small batches (5-10): Better for complex categories
- Large batches (15-20): Better for simple patterns
- Very large datasets: Use parallel generation

### 2. Configure Checkpointing Wisely

- Short intervals (25-50): For unstable connections
- Long intervals (100-200): For reliable environments
- Always enabled for large datasets (>100 cases)

### 3. Tune Diversity Settings

- Wide window (30-50): For categories prone to repetition
- Narrow window (10-20): For naturally diverse categories
- Monitor `diversity_rejections` to tune

### 4. Monitor Progress

```python
# Check current state
await get_builder_state_summary()

# Output shows:
# - Progress percentage
# - Checkpoint status
# - Diversity rejections
# - Category breakdown
```

## Troubleshooting

### Issue: Too Many Diversity Rejections

**Symptom:** Generation is slow, many cases rejected

**Solutions:**
1. Reduce diversity_window
2. Disable diversity checking for that category
3. Provide more varied example inputs

### Issue: Checkpoint Not Saving

**Symptom:** No checkpoint files found

**Solutions:**
1. Check checkpoint_dir exists and is writable
2. Verify checkpoint_interval is reached
3. Ensure enable_checkpointing=True

### Issue: Worker Generation Fails

**Symptom:** Errors during parallel generation

**Solutions:**
1. Reduce batch_size
2. Disable parallel_generation
3. Check API rate limits
4. Use slower worker_model

## Future Enhancements

### Planned Features

1. **Semantic Deduplication**: Use embeddings for similarity
2. **Template-Based Generation**: Pre-generate schemas for efficiency
3. **Quality Validation**: Separate agent reviews samples
4. **Incremental MLflow Logging**: Log as you generate
5. **Dynamic Batch Sizing**: Adjust based on complexity

### Experimental Features

- LLM-as-Judge quality filtering
- Multi-stage pipeline generation
- Streaming generation with backpressure

## Examples

### Generate Large Dataset (500 cases)

```python
# System automatically:
# - Splits into 25 batches of 20
# - Runs workers in parallel
# - Saves checkpoints every 50 cases
# - Checks diversity
# - All with minimal context!

plan = DatasetPlanInput(
    dataset_name="large_eval_set",
    categories=[
        DatasetCategory(
            name="order_status",
            description="Test order status queries",
            count=200,
            example_inputs="Where is my order #12345?",
            expectations={"contains_order_id": True}
        ),
        DatasetCategory(
            name="refunds",
            description="Test refund requests",
            count=200,
            example_inputs="I want a refund for order #12345",
            expectations={"initiates_refund_flow": True}
        ),
        DatasetCategory(
            name="product_info",
            description="Test product information queries",
            count=100,
            example_inputs="Tell me about the wireless headphones",
            expectations={"provides_product_details": True}
        )
    ],
    total_instances=500
)

# Agent handles the rest!
```

### Resume Interrupted Generation

```python
# If generation was interrupted at 273/500 cases:

# Load checkpoint
await load_from_checkpoint()

# Continue from 273
# Agent knows exactly where to resume!
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_dataset_size` | 100 | Maximum total cases allowed |
| `batch_size` | 20 | Cases per batch |
| `model` | gpt-4o | Main agent model |
| `worker_model` | gpt-4o-mini | Worker agent model |
| `enable_parallel_generation` | True | Use parallel workers |
| `enable_checkpointing` | True | Auto-save checkpoints |
| `checkpoint_interval` | 50 | Cases between checkpoints |
| `checkpoint_dir` | data/checkpoints | Checkpoint storage |
| `enable_diversity_check` | True | Prevent duplicates |
| `diversity_window` | 20 | Cases to check for duplicates |

## Summary

The scalable dataset generation system enables:

✅ **Unlimited Scale**: Generate thousands of test cases
✅ **Cost Efficient**: Use cheaper models for bulk generation  
✅ **Resilient**: Auto-checkpointing for recovery
✅ **High Quality**: Diversity checking prevents duplicates
✅ **Fast**: Parallel generation for speed
✅ **Transparent**: Track progress and rejections

All while maintaining a **constant context window** size!
