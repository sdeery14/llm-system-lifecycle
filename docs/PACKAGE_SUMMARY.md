# Package Summary: mlflow-eval-tools

## Overview

`mlflow-eval-tools` is a Python package that provides evaluation tools for OpenAI Agents SDK projects. It enables teams to:

1. **Build comprehensive evaluation datasets** through conversational interaction
2. **Run LLM-judge analysis** with custom scorers and detailed reporting
3. **Track everything in MLflow** for versioning and comparison

## What's Included

### Package Structure

```
src/
├── mlflow_eval_tools/          # Main package
│   ├── __init__.py             # Package initialization
│   ├── __main__.py             # CLI entry point
│   └── cli.py                  # CLI commands (Click-based)
└── app_agents/                 # Agent implementations
    ├── dataset_builder.py      # Dataset builder agent
    └── agent_analysis.py       # Analysis tool with custom scorers
```

### CLI Commands

1. **`mlflow-eval-tools dataset-builder`**
   - Interactive agent for creating evaluation datasets
   - Analyzes target agent capabilities
   - Generates diverse test cases using structured outputs
   - Supports parallel batch generation
   - Stores datasets in MLflow

2. **`mlflow-eval-tools agent-analysis`**
   - Runs evaluation using MLflow's eval framework
   - 5 custom scorers (exact match, content match, tool validation, efficiency, LLM-judge)
   - Generates comprehensive analysis reports
   - Logs all results and artifacts to MLflow

3. **`mlflow-eval-tools info`**
   - Display package and environment information

### Features

**Dataset Builder:**
- ✅ Interactive conversation-based dataset creation
- ✅ Automatic agent analysis (tools, instructions, capabilities)
- ✅ Structured outputs with Pydantic models for reliability
- ✅ Parallel batch generation with worker agents
- ✅ Checkpointing for resumable creation
- ✅ Diversity checks to avoid duplicate patterns
- ✅ MLflow integration for versioning and storage
- ✅ Configurable batch sizes and limits

**Agent Analysis:**
- ✅ 5 built-in scorers for comprehensive evaluation
- ✅ LLM-as-judge for quality assessment (0-100 scores)
- ✅ Tool call validation via trace analysis
- ✅ Per-category performance breakdown
- ✅ Detailed failure analysis
- ✅ Actionable recommendations
- ✅ Full MLflow integration for tracking

## Installation Options

### Option 1: From Wheel (Recommended)
```bash
pip install mlflow_eval_tools-0.1.0-py3-none-any.whl
```

### Option 2: From Git
```bash
pip install git+https://github.com/sdeery14/mlflow-eval-tools.git
```

### Option 3: From Source
```bash
git clone https://github.com/sdeery14/mlflow-eval-tools.git
cd mlflow-eval-tools
poetry install
```

## Quick Start

```bash
# 1. Set up environment
export OPENAI_API_KEY=sk-...

# 2. Create dataset
mlflow-eval-tools dataset-builder \
  --agent-file src/my_agent.py \
  --agent-class MyAgent

# 3. Run evaluation
mlflow-eval-tools agent-analysis <RUN_ID> <DATASET_NAME>

# 4. View results
mlflow ui
```

## Dependencies

- Python 3.12+
- mlflow ^3.4.0
- openai-agents ^0.3.3
- pydantic ^2.11.10
- click ^8.1.7
- python-dotenv ^1.2.1
- And others (see pyproject.toml)

## Documentation

- **PACKAGE_README.md** - Comprehensive package documentation
- **docs/QUICK_START_TEAMS.md** - Quick start guide for teams
- **BUILD_GUIDE.md** - How to build and distribute the package
- **docs/dataset_builder/** - Dataset builder detailed docs
- **docs/agent_analysis/** - Agent analysis detailed docs

## Key Benefits

### For Development Teams

1. **Standardized Evaluation**: Consistent approach across all agent projects
2. **Easy to Use**: Simple CLI, no code changes to agents
3. **Comprehensive**: Multiple scoring methods, detailed reports
4. **Trackable**: Full MLflow integration for versioning and comparison
5. **Scalable**: Parallel generation, checkpointing for large datasets

### For Organizations

1. **Portable**: Distribute as wheel, install anywhere
2. **Reproducible**: Version-controlled datasets and evaluations
3. **Collaborative**: Teams can share datasets via MLflow
4. **Cost-Effective**: Configurable models (use cheaper models for workers)
5. **Maintainable**: Clear architecture, well-documented

## Example Workflows

### Workflow 1: Creating First Dataset
```bash
# Interactive mode - agent guides you through everything
mlflow-eval-tools dataset-builder

# Follow prompts to:
# - Specify agent file and class
# - Define test categories
# - Generate test cases
# - Store in MLflow
```

### Workflow 2: Running Evaluation
```bash
# Get agent run ID from dataset builder output or MLflow UI
mlflow-eval-tools agent-analysis abc123 my_dataset_v1

# Results logged to MLflow:
# - Metrics: pass rates, scores
# - Artifacts: analysis.json, report.md
# - Traces: tool call sequences
```

### Workflow 3: Version Comparison
```bash
# Evaluate version 1
mlflow-eval-tools agent-analysis v1_run_id dataset_v1

# Make improvements to agent

# Evaluate version 2 on same dataset
mlflow-eval-tools agent-analysis v2_run_id dataset_v1

# Compare in MLflow UI
```

## Testing the Installation

Run the included test script:

```bash
python test_installation.py
```

This verifies:
- Package imports work
- CLI is available
- Dependencies are installed
- Commands execute correctly

## Distribution

### For Internal Teams

```bash
# Build the package
poetry build

# Share the wheel file
# dist/mlflow_eval_tools-0.1.0-py3-none-any.whl

# Teams install with:
pip install mlflow_eval_tools-0.1.0-py3-none-any.whl
```

### For Public Release

```bash
# Build
poetry build

# Publish to PyPI
poetry publish
```

## Support and Maintenance

### Getting Help

- **Documentation**: Start with QUICK_START_TEAMS.md
- **Examples**: See docs/ folder
- **Issues**: GitHub issue tracker
- **MLflow UI**: Explore results at http://localhost:5000

### Maintenance Tasks

- Update dependencies: `poetry update`
- Run tests: `poetry run pytest`
- Check security: `poetry show --outdated`
- Update docs when adding features

## Versioning

Following semantic versioning (SemVer):

- **Current**: 0.1.0 (initial release)
- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes

## License

MIT License - See LICENSE file

## Contributors

- Sean Deery <sdeery14@gmail.com>

## Repository

https://github.com/sdeery14/mlflow-eval-tools

## Next Steps

1. **Read** QUICK_START_TEAMS.md for detailed guide
2. **Install** the package in your environment
3. **Run** test_installation.py to verify
4. **Create** your first dataset
5. **Evaluate** your agent
6. **Review** results in MLflow UI
7. **Share** with your team!

---

For questions or issues, please open an issue on GitHub or contact the maintainers.
