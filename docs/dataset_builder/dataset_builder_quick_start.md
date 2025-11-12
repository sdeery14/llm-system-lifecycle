# Dataset Builder - Quick Start Guide

## Overview

The Dataset Builder Agent now supports **scalable dataset generation** that can handle thousands of test cases without context window issues.

## Key Features at a Glance

### 🚀 Scalability Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Parallel Workers** | ✅ Enabled | Generate batches in parallel using worker agents |
| **Checkpointing** | ✅ Enabled | Auto-save every 50 cases (configurable) |
| **Diversity Checking** | ✅ Enabled | Prevent duplicate test cases |
| **Cost Optimization** | ✅ Enabled | Use gpt-4o-mini for workers |

## Quick Start

### 1. Basic Usage (Default Settings)

```python
from src.app_agents.dataset_builder import DatasetBuilderAgent
import asyncio

async def main():
    # Create agent - all scalability features enabled by default!
    agent = DatasetBuilderAgent()
    
    # Start interactive chat
    await agent.interactive_chat()

if __name__ == "__main__":
    asyncio.run(main())
```

**Or run directly:**

```powershell
cd d:/projects/mlflow-eval-tools
poetry shell
```

### 2. Environment Setup

Create a `.env` file:

```bash
OPENAI_API_KEY=your-key-here
MAX_DATASET_INSTANCES=1000  # Optional: default is 100
```

### 3. Example Conversation

```
You: I want to create a dataset for my customer service agent

Agent: Great! Let me help you. First, I need to log your agent in MLflow.
       What's the file path to your agent?

You: src/dev_agents/customer_service_agent.py

Agent: And what's the agent class name?

You: CustomerServiceAgent

Agent: Perfect! What does this agent do?

You: It helps customers with order status, refunds, and product information

Agent: [Logs agent in MLflow]
       Now let's plan your dataset. What categories do you want to test?

You: I want:
     - 200 cases for order status queries
     - 150 cases for refund requests  
     - 100 cases for product information

Agent: [Creates plan with 450 total test cases]
       [Uses parallel workers to generate in batches of 20]
       [Auto-saves checkpoints every 50 cases]
       [Checks for duplicate patterns]
       [Stores final dataset in MLflow]
```

## Configuration Options

### Default Configuration (Recommended)

```python
DatasetBuilderConfig(
    max_dataset_size=100,              # Increase via env var
    batch_size=20,                     # Cases per batch
    model="gpt-4o",                    # Main agent model
    worker_model="gpt-4o-mini",        # Worker model (cheaper)
    enable_parallel_generation=True,   # Parallel workers
    enable_checkpointing=True,         # Auto-save
    checkpoint_interval=50,            # Save every 50 cases
    checkpoint_dir="data/checkpoints", # Checkpoint location
    enable_diversity_check=True,       # Prevent duplicates
    diversity_window=20                # Check last 20 cases
)
```

### Custom Configuration

```python
from src.app_agents.dataset_builder import DatasetBuilderAgent, DatasetBuilderConfig
from pathlib import Path

config = DatasetBuilderConfig(
    max_dataset_size=1000,
    checkpoint_interval=100,
    diversity_window=30
)

agent = DatasetBuilderAgent(model="gpt-4o")
```

## Common Use Cases

### Generate Large Dataset (500+ cases)

The agent automatically:
- ✅ Splits into batches of 20
- ✅ Runs workers in parallel
- ✅ Saves checkpoints every 50 cases
- ✅ Prevents duplicates
- ✅ Uses minimal context per worker

**No manual intervention needed!**

### Resume Interrupted Generation

If generation stops at 273/500 cases:

```python
# In the interactive chat:
You: Load the checkpoint for my dataset

Agent: [Loads checkpoint]
       Checkpoint loaded! You have 273/500 cases.
       Ready to continue from where you left off.

You: Continue generating

Agent: [Resumes from case 274]
```

### Monitor Progress

```python
# During generation, ask:
You: What's the current status?

Agent: [Shows state summary]
       Progress: 273/500 (54.6%)
       Diversity rejections: 12
       Last checkpoint: 250 cases
       Categories completed: 2/3
```

## Architecture

### How It Works

```
Your Request (500 test cases)
        ↓
Main Agent (gpt-4o)
  - Plans categories
  - Coordinates workflow
        ↓
Spawns Worker Agents (gpt-4o-mini)
  - Worker 1: Batch 1 (20 cases) ━━┓
  - Worker 2: Batch 2 (20 cases) ━━┫
  - Worker 3: Batch 3 (20 cases) ━━┫━━ Run in Parallel
  - Worker 4: Batch 4 (20 cases) ━━┫
  - Worker 5: Batch 5 (20 cases) ━━┛
        ↓
Diversity Check
  - Reject duplicates
  - Keep unique cases
        ↓
Checkpoint (every 50 cases)
  - Save state to disk
  - Enable resume
        ↓
Final Dataset → MLflow
```

### Why This Scales

| Problem | Traditional Approach | Scalable Approach |
|---------|---------------------|-------------------|
| Context grows with cases | Single agent with all context | Workers with minimal context |
| Cost for large datasets | All cases use expensive model | Workers use cheap model |
| Lost progress if interrupted | Start over | Resume from checkpoint |
| Duplicate test cases | Manual review | Automatic diversity check |
| Slow generation | Sequential | Parallel workers |

## Tips & Best Practices

### 1. Start Small, Scale Up

```python
# First run: Test with small dataset
MAX_DATASET_INSTANCES=50

# Once validated: Scale up
MAX_DATASET_INSTANCES=1000
```

### 2. Use Checkpoints for Large Datasets

For datasets > 100 cases, checkpointing is essential:
- Protects against interruptions
- Allows resuming from exact point
- Saves intermediate progress

### 3. Monitor Diversity Rejections

If you see many rejections:
```
Diversity rejections: 45
```

**Solutions:**
- Provide more varied example inputs
- Reduce diversity_window
- Check if category is too narrow

### 4. Leverage Parallel Generation

Parallel generation is most effective for:
- Large batches (>10 cases)
- Simple categories
- Fast generation speed

**Disabled automatically for:**
- Small batches (<5 cases)
- Complex categories (can override)

## Troubleshooting

### "Context window exceeded"

**This shouldn't happen!** But if it does:
1. Check `batch_size` is ≤ 20
2. Verify workers are being used
3. Reduce `diversity_window`

### "Checkpoint not found"

1. Check `checkpoint_dir` exists
2. Verify dataset name matches
3. Use correct checkpoint path

### "Generation is slow"

1. Increase `batch_size` (max 20)
2. Enable `parallel_generation`
3. Use faster `worker_model`

## Advanced Features

### Manual Checkpoint Management

```python
# Save checkpoint manually
from src.app_agents.dataset_builder import _save_checkpoint

_save_checkpoint(
    Path("custom/path/checkpoint.json"),
    state_data={...}
)

# Load specific checkpoint
await load_from_checkpoint(
    checkpoint_path="custom/path/checkpoint.json"
)
```

### Disable Features Selectively

```python
config = DatasetBuilderConfig(
    enable_parallel_generation=False,  # Sequential only
    enable_diversity_check=False,      # Allow duplicates
    enable_checkpointing=False         # No auto-save
)
```

## Next Steps

1. **Read full docs**: See `docs/scalable_dataset_generation.md`
2. **Try example**: Run `scripts/dataset_builder_example.py`
3. **Explore code**: Check `src/app_agents/dataset_builder.py`

## Summary

The Dataset Builder Agent can now:

✅ Generate **unlimited test cases** (not limited by context)  
✅ Use **parallel workers** for speed  
✅ **Auto-checkpoint** for resilience  
✅ **Prevent duplicates** with diversity checking  
✅ **Cost optimize** with cheaper worker models  

**All enabled by default!** Just run and let the agent handle the complexity.
