# LLM Lifecycle Platform Architecture

## Overview

This document describes the architecture for a comprehensive LLM (Large Language Model) lifecycle management platform. We use the C4 model—a hierarchical technique with four diagram levels: Context (C1), Container (C2), Component (C3), and Code (C4)—to progressively zoom from the system landscape to implementation details (Note: C4 diagrams will be added as implementation evolves). The platform is designed to support the complete lifecycle of LLM development, evaluation, optimization, and deployment with human-in-the-loop oversight and MLflow as the central tracking and storage system.

The platform follows a **human-guided, agent-assisted** approach where AI agents perform heavy computational work and systematic analysis. Human evaluators provide oversight, strategic decisions, and quality control. Each transition between lifecycle stages requires human approval (with future flexibility for automation). MLflow serves as the central repository for experiments, models, artifacts, and metadata.

## System Context Diagram (C1)

- Highest level of abstraction
- Describes something that delivers value to its users

![System Context Diagram](llm-system-lifecycle-c1.drawio.svg)

The left hand side of the C1 diagram shows the **External Actors** that will interact with the platform. External Actors can include QA Testers, ML Engineers, Analysts, and Project Managers. These tasks can all be done by one user on small teams or by the appropriate specialists on larger teams. This user may also be referred to generally as the **human evaluator**.

The **LLM System Lifecycle Platform** block in the middle represents the core functionality of this project.

The right hand side of the C1 diagram shows the **External Systems** the platform will interact with. External systems this project interacts with include MLflow, GitHub, Notification Services, and LLM Providers.

## Container Diagram (C2)

- Application pieces
- Data stores 

![Container Diagram](llm-system-lifecycle-c2.drawio.svg)

The C2 Container Diagram zooms into the **LLM System Lifecycle Platform** block to show the main pieces of the application. The top of the C2 Diagram shows the **Client Container**, which holds the Web UI the human evaluator will interact with. The Web UI includes an interactive dashboard to view results and chat interfaces for agent interaction. 

The **Backend Container** on the left hand side shows where the core functionality of the app lives. The **LLM Lifecycle API (FastAPI-based)** handles RESTful API serving the client application. The **LLM Lifecycle Orchestrator** is the core workflow engine managing the evaluation pipeline. 

The **External Services** on the right hand side shows the external services the LLM Lifecycle Orchestrator uses. The **MLflow Tracking + Registry** acts as the central repository for all experiments and model artifacts. It manages the storage of evaluation metrics, comparison reports, and model versions. The **MLflow Storage** block is the object store and SQL database MLflow uses to store the files and metadata. The **LLM Provider** is the collection of external APIs for large language models that handle inference requests and responses.

## Component Diagrams (C3)

- The components that make up the core logic of the app

### LLM Lifecycle Orchestrator Component (C3)

![Component Diagram](llm-system-lifecycle-c3.drawio.svg)

The LLM Lifecycle Orchestrator Component diagram details the internal structure of the LLM Lifecycle Orchestrator Backend, organized into the different stages of the evaluation workflow.

In **Stage 1: Agent Development**, the human evaluator builds an agent to evaluate and optimize.

In **Stage 2: Dataset Builder**, the human evaluator has a chat with the Dataset Builder Agent. The agent reads the metadata for the agent being evaluated, and then ideates with the user how it should set up a dataset that will accurately evaluate how the app performs when presented with potential real world input. The dataset requires a scenario and an expected behavior. When both the human evaluator and the agent agree to create the dataset, the dataset is created and then immediately analyzed in a step called Dataset Validation. This step involves analyzing the created dataset to make sure it passed a quality threshold. The dataset along with a report about the dataset is saved to MLflow.

In **Stage 3: Simulator**, the User Simulation Agent is configured with each scenario from the test dataset and is run against the agent. The chat traces are saved to MLflow. The Human Evaluator is not involved in this step other than to trigger the simulation.

In **Stage 4: Judging**, the scenarios, expected behaviors, and chat traces from the simulations are passed to a Judge Agent who gives each simulation a score and a reason for that score. The scores and rationales are stored in MLflow.

In **Stage 5: Analytics**, the scores and rationales are aggregated, and the human evaluator has a chat with the Analyst Agent to explore the data and build a report that explains as much as possible about what can be expected by the agent in a production setting based on the simulation results. The report is stored in MLflow.

In **Stage 6: Comparison**, the human evaluator has a chat with the Comparison Agent to compare the results of two different agents. The resulting comparison analysis and recommendation report is saved to MLflow.

In **Stage 7: Deployment**, the best model is registered in MLflow, and follows a progressive rollout to eventually be fully rolled out.

In **Stage 8: Feedback Analysis**, the human evaluator has a chat with the Feedback Analyst Agent to analyze the feedback of the deployed model, which results in a production agent report.

In the **Feedback Loop**, the production agent report is used by the human AI engineer and the Dataset Builder Agent to develop better models, tools, prompts, and evaluation datasets. 

### MLflow Integration (C3)

![MLflow Component Diagram](llm-system-lifecycle-c3-mlflow.drawio.svg)

**Experiment Tracking**
- Each evaluation run is logged as an MLflow experiment
- Parameters include agent configuration, prompts, and evaluation settings
- Metrics include performance scores, latency, and quality measures

**Artifact Storage**
- Evaluation reports stored as HTML/PDF artifacts
- Chat transcripts and interaction logs preserved
- Model checkpoints and configuration files versioned

**Model Registry**
- Agents registered as MLflow models with version control
- Stage transitions (None → Staging → Production) tracked
- Model lineage and evaluation history maintained


## Development Phases

### Phase 1: End-to-End CLI + MLflow (local)
Goal: ship a thin, working pipeline that runs locally via CLI and logs to MLflow.

- Scripts-first approach
  - Start by creating:
    - Agent scripts (runnable CLIs): scripts/agents/dataset_builder.py, user_simulator.py, judge.py, analyst.py, comparator.py, feedback_analyst.py.
    - Stage/session scripts (runnable CLIs): scripts/stages/dataset_session.py, simulate_session.py, judge_session.py, analytics_session.py, compare_session.py, deploy_session.py, feedback_session.py.
  - Each script:
    - Accepts JSON via --input/--output file or stdin/stdout following the data contracts.
    - Logs its own MLflow run with standard tags and artifacts.
    - Is idempotent and can be executed standalone or composed in the pipeline.

Deliverables
- Minimal Agents: DatasetBuilder (rule/rubric driven), Simulator (calls LLM provider or stub), Judge (simple rubric + rationale), Analyst (basic aggregation/report).
- Data contracts (JSON Schemas) for: Dataset Scenario + Expected Behavior, Simulation Trace, Judgment (score + rationale), Analytics Report.
- CLI commands (examples): dataset build/validate, simulate run, judge run, analytics report, compare run, deploy register (stub).
- MLflow setup: Experiments/Runs, standard tags (project_id, agent_id, dataset_version, git_sha), artifact layout (dataset/, simulation/, judging/, reports/).
- Instrumentation: structured logs + basic metrics (counts, latency, cost estimate).
- Orchestrating pipeline script: scripts/pipeline/run_pipeline.py to chain stage scripts end-to-end.
- Pipeline self-evaluation: use the pipeline to generate an “Agents Report” that evaluates each agent (latency, cost, determinism across seeds, rubric adherence, error rates) using the same MLflow experiments and artifacts.

Acceptance criteria
- One command runs the full pipeline on a sample dataset and writes runs/artifacts to MLflow.
- A report artifact is viewable and traces/judgments are reproducible.
- Each agent and each stage/session script can be run independently with JSON I/O.
- The pipeline produces an “Agents Report” summarizing per-agent performance to guide improvements.

### Phase 2: API + Minimal Web UI
Goal: expose the pipeline via FastAPI and add a basic UI for visibility and control.

Deliverables
- FastAPI endpoints for each stage with job status (CREATED → RUNNING → SUCCEEDED/FAILED).
- Simple background worker (thread/process) for long-running steps.
- Web UI: runs list, run detail (artifacts/metrics), minimal chat panes for Dataset/Analyst agents.
- Auth optional in dev; config via env files.

Acceptance criteria
- A user can kick off a run from the UI, watch status, and open artifacts without the CLI.

### Phase 3: Deployment, Monitoring, Scheduled Feedback
Goal: make it durable and observable; close the production feedback loop.

Deliverables
- Containerization and deploy to a minimal environment (single VM or managed container service) with MLflow backend store + object storage.
- Observability: logs, metrics, traces (OpenTelemetry or equivalent) and dashboards.
- Alerts for failed jobs and abnormal metrics.
- Scheduled jobs to ingest production feedback and run periodic evaluation/analytics.

Acceptance criteria
- Deployed system runs scheduled feedback evaluations; on failure, alerts fire; dashboards show run health.

### Phase 4: Agentic Automation (opt-in)
Goal: progressively automate human steps with guardrails.

Deliverables
- Auto dataset expansion and rubric refinement proposals.
- Auto-judging + confidence checks; bias and drift checks.
- Auto-promotion gates tied to MLflow model stages with thresholds; human override and audit artifacts.
- Feature flags to turn automation on/off per project.

Acceptance criteria
- With automation enabled, the system proposes or executes steps and records justification; disabling flags reverts to human-in-the-loop.

## Conclusion

This architecture provides a comprehensive foundation for LLM lifecycle management with human oversight and systematic evaluation. The modular design allows for incremental development and future enhancement while maintaining the core principle of human-guided, agent-assisted evaluation processes.

The integration with MLflow ensures that all evaluation activities are properly tracked and reproducible, while the flexible gate system allows for both careful human oversight and future automation as the system matures.