# mlflow-eval-tools Documentation

Welcome to the mlflow-eval-tools documentation! This guide will help you find the right documentation for your needs.

## 📚 Quick Navigation

### Getting Started
- **[Quick Start Guide](QUICK_START_TEAMS.md)** - Get up and running quickly
- **[Installation & Setup](../README.md)** - Installation options and basic setup
- **[Package Overview](PACKAGE_SUMMARY.md)** - High-level package summary

### Core Features

#### Dataset Builder
Build evaluation datasets through conversational interaction:
- **[Dataset Builder Guide](dataset_builder/dataset_builder.md)** - Complete guide to dataset creation
- **[Quick Start](dataset_builder/dataset_builder_quick_start.md)** - Fast-track dataset creation
- **[Scalable Generation](dataset_builder/scalable_dataset_generation.md)** - Large-scale dataset generation
- **[Structured Output Update](dataset_builder/structured_output_update.md)** - Technical implementation details

#### Agent Analysis
Evaluate agents with LLM-judge scoring:
- **[Agent Analysis README](agent_analysis/README.md)** - Overview and quick examples
- **[Evaluation Guide](agent_analysis/agent_evaluation_guide.md)** - Comprehensive evaluation guide
- **[Quick Start](agent_analysis/quick_start.md)** - Get started with evaluation
- **[Response Quality Scorer](agent_analysis/response_quality_scorer_update.md)** - LLM-as-judge details

### For Package Maintainers

- **[Build Guide](BUILD_GUIDE.md)** - How to build and distribute the package
- **[Deployment Checklist](DEPLOYMENT_CHECKLIST.md)** - Pre-deployment verification
- **[Commands Reference](COMMANDS.md)** - Quick command reference

### Complete Reference

- **[Package README](PACKAGE_README.md)** - Comprehensive package documentation
- **[Project Completion](PROJECT_COMPLETION.md)** - Development history and completion notes

## 🎯 Documentation by Role

### For End Users (Teams Using the Package)

1. Start with **[Quick Start Guide](QUICK_START_TEAMS.md)**
2. Learn about **[Dataset Builder](dataset_builder/dataset_builder.md)**
3. Learn about **[Agent Analysis](agent_analysis/agent_evaluation_guide.md)**
4. Reference **[Package README](PACKAGE_README.md)** for details

### For Package Distributors

1. Review **[Build Guide](BUILD_GUIDE.md)**
2. Follow **[Deployment Checklist](DEPLOYMENT_CHECKLIST.md)**
3. Use **[Commands Reference](COMMANDS.md)** for quick operations
4. Share **[Quick Start Guide](QUICK_START_TEAMS.md)** with teams

### For Developers/Contributors

1. Read **[Package README](PACKAGE_README.md)** for architecture
2. Review source code in `src/mlflow_eval_tools/` and `src/app_agents/`
3. Check **[Project Completion](PROJECT_COMPLETION.md)** for context
4. Follow **[Build Guide](BUILD_GUIDE.md)** for development workflow

## 📖 Documentation Structure

```
mlflow-eval-tools/
├── README.md                          # Project overview & quick start
├── LICENSE                            # License file
├── pyproject.toml                     # Package configuration
│
└── docs/                              # All documentation
    ├── README.md                      # This file - documentation index
    ├── QUICK_START_TEAMS.md          # Quick start for teams
    ├── PACKAGE_README.md              # Complete package documentation
    ├── PACKAGE_SUMMARY.md             # High-level summary
    ├── BUILD_GUIDE.md                 # Build & distribution guide
    ├── COMMANDS.md                    # Quick command reference
    ├── DEPLOYMENT_CHECKLIST.md        # Pre-deployment checklist
    ├── PROJECT_COMPLETION.md          # Development notes
    │
    ├── dataset_builder/               # Dataset creation documentation
    │   ├── dataset_builder.md        # Complete dataset builder guide
    │   ├── dataset_builder_quick_start.md
    │   ├── scalable_dataset_generation.md
    │   └── structured_output_update.md
    │
    ├── agent_analysis/                # Evaluation documentation
    │   ├── README.md                  # Agent analysis overview
    │   ├── agent_evaluation_guide.md # Complete evaluation guide
    │   ├── quick_start.md            # Quick start for evaluation
    │   └── response_quality_scorer_update.md
    │
    └── archive/                       # Historical research docs
        ├── README.md
        ├── architecture.md
        └── *.svg (diagram files)
```

## 🚀 Common Workflows

### Creating Your First Dataset

1. Read: [Quick Start Guide](QUICK_START_TEAMS.md) → "Creating Your First Evaluation Dataset"
2. Follow: [Dataset Builder Quick Start](dataset_builder/dataset_builder_quick_start.md)
3. Reference: [Dataset Builder Guide](dataset_builder/dataset_builder.md) for advanced options

### Running Your First Evaluation

1. Read: [Quick Start Guide](QUICK_START_TEAMS.md) → "Running Evaluation"
2. Follow: [Agent Analysis Quick Start](agent_analysis/quick_start.md)
3. Reference: [Evaluation Guide](agent_analysis/agent_evaluation_guide.md) for details

### Building and Distributing the Package

1. Read: [Build Guide](BUILD_GUIDE.md) → "Building the Package"
2. Check: [Deployment Checklist](DEPLOYMENT_CHECKLIST.md)
3. Use: [Commands Reference](COMMANDS.md) for quick commands

## 🔍 Finding Specific Information

### Installation
- See [README.md](../README.md) → Installation section
- Or [Quick Start Guide](QUICK_START_TEAMS.md) → Installation section

### CLI Commands
- See [COMMANDS.md](COMMANDS.md) for quick reference
- Or [PACKAGE_README.md](PACKAGE_README.md) → CLI Reference section

### Configuration
- See [PACKAGE_README.md](PACKAGE_README.md) → Configuration section
- Or [Quick Start Guide](QUICK_START_TEAMS.md) → Setup section

### Troubleshooting
- See [Quick Start Guide](QUICK_START_TEAMS.md) → Troubleshooting section
- Or [PACKAGE_README.md](PACKAGE_README.md) → Troubleshooting section

### Scorers & Evaluation Metrics
- See [Agent Analysis Guide](agent_analysis/agent_evaluation_guide.md)
- Or [Response Quality Scorer](agent_analysis/response_quality_scorer_update.md)

### Advanced Topics
- **Large Datasets**: [Scalable Dataset Generation](dataset_builder/scalable_dataset_generation.md)
- **Structured Outputs**: [Structured Output Update](dataset_builder/structured_output_update.md)
- **LLM-as-Judge**: [Response Quality Scorer](agent_analysis/response_quality_scorer_update.md)

## 💡 Tips

- **New users**: Start with [Quick Start Guide](QUICK_START_TEAMS.md)
- **Need help**: Check troubleshooting sections in relevant docs
- **Want details**: [PACKAGE_README.md](../PACKAGE_README.md) has comprehensive info
- **Building/deploying**: Use [BUILD_GUIDE.md](../BUILD_GUIDE.md) and [DEPLOYMENT_CHECKLIST.md](../DEPLOYMENT_CHECKLIST.md)

## 🔗 External Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **OpenAI Agents SDK**: https://github.com/openai/openai-agents-sdk
- **GitHub Repository**: https://github.com/sdeery14/mlflow-eval-tools
- **Issue Tracker**: https://github.com/sdeery14/mlflow-eval-tools/issues

## ❓ Still Need Help?

1. Check the **[Package README](PACKAGE_README.md)** for comprehensive documentation
2. Search for your issue in the docs using your editor's search
3. Check the **[GitHub Issues](https://github.com/sdeery14/mlflow-eval-tools/issues)**
4. Review **[Quick Start Guide](QUICK_START_TEAMS.md)** troubleshooting section

---

**Last Updated**: November 2025  
**Package Version**: 0.1.0
