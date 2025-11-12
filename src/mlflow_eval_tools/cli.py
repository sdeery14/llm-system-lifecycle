"""
CLI for mlflow-eval-tools.

Provides commands for:
- Building evaluation datasets (dataset-builder)
- Running LLM-judge analysis (agent-analysis)
"""

import os
import sys
import asyncio
import click
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@click.group()
@click.version_option(version="0.1.0", prog_name="mlflow-eval-tools")
def cli():
    """
    MLflow Evaluation Tools for OpenAI Agents SDK.
    
    Build evaluation datasets and run LLM-judge analysis for your agents.
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        click.echo(
            click.style("⚠️  Warning: OPENAI_API_KEY not found in environment.", fg="yellow"),
            err=True
        )
        click.echo("Set it in your .env file or export it:", err=True)
        click.echo("  export OPENAI_API_KEY=your-key-here", err=True)
        click.echo()


@cli.command(name="dataset-builder")
@click.option(
    "--agent-file",
    type=click.Path(exists=True),
    help="Path to the agent file (relative to project root)"
)
@click.option(
    "--agent-class",
    type=str,
    help="Name of the agent class (e.g., 'CustomerServiceAgent')"
)
@click.option(
    "--use-previous",
    type=str,
    metavar="RUN_ID",
    help="Use a previously logged agent from MLflow by run ID"
)
@click.option(
    "--list-agents",
    is_flag=True,
    help="List previously logged agents from MLflow"
)
@click.option(
    "--max-size",
    type=int,
    help="Maximum dataset size (overrides MAX_DATASET_INSTANCES env var)"
)
@click.option(
    "--batch-size",
    type=int,
    default=20,
    help="Batch size for test case generation (default: 20)"
)
@click.option(
    "--model",
    type=str,
    default="gpt-4o",
    help="LLM model for main agent (default: gpt-4o)"
)
@click.option(
    "--worker-model",
    type=str,
    default="gpt-4o-mini",
    help="LLM model for worker agents (default: gpt-4o-mini)"
)
def dataset_builder(
    agent_file, 
    agent_class, 
    use_previous, 
    list_agents, 
    max_size, 
    batch_size, 
    model, 
    worker_model
):
    """
    Interactive dataset builder agent.
    
    Create high-quality evaluation datasets for your agents through
    conversational interaction. The agent will:
    
    \b
    1. Log your target agent in MLflow (or reuse a previous one)
    2. Discuss dataset requirements with you
    3. Create a structured plan with categories
    4. Generate test cases in batches
    5. Store the dataset in MLflow for evaluation
    
    Examples:
    
    \b
      # Interactive mode (recommended)
      mlflow-eval-tools dataset-builder
    
    \b
      # Start with a specific agent
      mlflow-eval-tools dataset-builder \\
        --agent-file src/dev_agents/customer_service_agent.py \\
        --agent-class CustomerServiceAgent
    
    \b
      # Use a previously logged agent
      mlflow-eval-tools dataset-builder --use-previous abc123runid
    
    \b
      # List available agents
      mlflow-eval-tools dataset-builder --list-agents
    """
    # Import here to avoid slow startup
    from app_agents.dataset_builder import DatasetBuilderAgent, interactive_chat
    from app_agents.dataset_builder import DatasetBuilderConfig
    from agents import Runner
    
    click.echo()
    click.echo("=" * 70)
    click.echo(click.style("Dataset Builder Agent", fg="cyan", bold=True))
    click.echo("=" * 70)
    
    # Handle --list-agents flag
    if list_agents:
        from app_agents.dataset_builder import list_previously_logged_agents
        from agents import RunContextWrapper
        
        click.echo("\nFetching previously logged agents from MLflow...\n")
        
        # Create a dummy context for the tool
        class DummyContext:
            pass
        
        result = list_previously_logged_agents(RunContextWrapper(DummyContext()), max_results=10)
        click.echo(result)
        return
    
    # Set max_size if provided
    if max_size:
        os.environ["MAX_DATASET_INSTANCES"] = str(max_size)
    
    # Show configuration
    config = DatasetBuilderConfig(
        batch_size=batch_size,
        model=model,
        worker_model=worker_model
    )
    
    click.echo("\nConfiguration:")
    click.echo(f"  Max Dataset Size: {config.max_dataset_size}")
    click.echo(f"  Batch Size: {config.batch_size}")
    click.echo(f"  Model: {config.model}")
    click.echo(f"  Worker Model: {config.worker_model}")
    
    # Prepare initial context based on options
    initial_context = ""
    
    if use_previous:
        click.echo(f"\nUsing previously logged agent: {use_previous}")
        initial_context = f"I want to use the previously logged agent with run ID: {use_previous}"
    elif agent_file and agent_class:
        click.echo("\nTarget Agent:")
        click.echo(f"  File: {agent_file}")
        click.echo(f"  Class: {agent_class}")
        initial_context = f"""I want to create an evaluation dataset for my agent.
Agent file: {agent_file}
Agent class: {agent_class}
Please analyze the agent and help me create a comprehensive dataset."""
    else:
        click.echo("\nStarting interactive mode...")
        click.echo("The agent will guide you through the process.\n")
    
    # Run the interactive chat
    async def run_chat():
        if initial_context:
            # Create agent and run with initial context
            builder = DatasetBuilderAgent(model=config.model)
            agent = builder.get_agent()
            
            result = await Runner.run(agent, initial_context)
            click.echo(f"\nAgent: {result.final_output}\n")
            
            # Continue with interactive chat
            while True:
                try:
                    user_input = input("You: ").strip()
                    
                    if user_input.lower() in ["quit", "exit", "bye"]:
                        click.echo("\nGoodbye! Your progress is saved in MLflow.")
                        break
                    
                    if not user_input:
                        continue
                    
                    result = await Runner.run(agent, user_input)
                    click.echo(f"\nAgent: {result.final_output}\n")
                    
                except KeyboardInterrupt:
                    click.echo("\n\nInterrupted. Goodbye!")
                    break
                except EOFError:
                    click.echo("\n\nGoodbye!")
                    break
        else:
            # Run full interactive chat
            await interactive_chat()
    
    asyncio.run(run_chat())


@cli.command(name="agent-analysis")
@click.argument("agent_run_id", type=str)
@click.argument("dataset_name", type=str)
@click.option(
    "--dataset-experiment",
    type=str,
    default="evaluation-datasets",
    help="Experiment containing the dataset (default: evaluation-datasets)"
)
@click.option(
    "--eval-experiment",
    type=str,
    default="agent-evaluation",
    help="Experiment to log evaluation results (default: agent-evaluation)"
)
@click.option(
    "--no-artifacts",
    is_flag=True,
    help="Skip saving analysis artifacts to MLflow"
)
def agent_analysis(agent_run_id, dataset_name, dataset_experiment, eval_experiment, no_artifacts):
    """
    Run LLM-judge analysis on an agent using an evaluation dataset.
    
    Evaluates agent performance using custom scorers:
    - Exact match
    - Content similarity
    - Tool usage correctness
    - Tool call efficiency
    - Response quality (LLM-as-judge)
    
    Results are logged to MLflow with comprehensive analysis reports.
    
    Arguments:
    
    \b
      AGENT_RUN_ID    MLflow run ID of the agent to evaluate
      DATASET_NAME    Name of the evaluation dataset in MLflow
    
    Examples:
    
    \b
      # Basic evaluation
      mlflow-eval-tools agent-analysis abc123runid customer_service_eval_v1
    
    \b
      # Custom experiment names
      mlflow-eval-tools agent-analysis abc123runid my_dataset \\
        --dataset-experiment my-datasets \\
        --eval-experiment my-evaluations
    
    \b
      # Skip artifact saving
      mlflow-eval-tools agent-analysis abc123runid my_dataset --no-artifacts
    """
    # Import here to avoid slow startup
    from app_agents.agent_analysis import main as analysis_main
    
    click.echo()
    click.echo("=" * 70)
    click.echo(click.style("Agent Analysis - MLflow Evaluation Framework", fg="green", bold=True))
    click.echo("=" * 70)
    click.echo()
    click.echo(f"Agent Run ID: {agent_run_id}")
    click.echo(f"Dataset: {dataset_name}")
    click.echo(f"Dataset Experiment: {dataset_experiment}")
    click.echo(f"Evaluation Experiment: {eval_experiment}")
    click.echo(f"Save Artifacts: {not no_artifacts}")
    click.echo()
    
    # Run the analysis
    asyncio.run(analysis_main(
        agent_run_id=agent_run_id,
        dataset_name=dataset_name,
        dataset_experiment=dataset_experiment,
        eval_experiment=eval_experiment,
        save_artifacts=not no_artifacts
    ))


@cli.command(name="info")
def info():
    """
    Display package and environment information.
    """
    import mlflow
    
    click.echo()
    click.echo(click.style("MLflow Evaluation Tools", fg="cyan", bold=True))
    click.echo("=" * 70)
    click.echo()
    click.echo("Package Information:")
    click.echo("  Version: 0.1.0")
    click.echo(f"  Python: {sys.version.split()[0]}")
    click.echo()
    click.echo("Environment:")
    click.echo(f"  OPENAI_API_KEY: {'✓ Set' if os.getenv('OPENAI_API_KEY') else '✗ Not set'}")
    click.echo(f"  MAX_DATASET_INSTANCES: {os.getenv('MAX_DATASET_INSTANCES', '100 (default)')}")
    click.echo()
    click.echo("MLflow Configuration:")
    click.echo(f"  Tracking URI: {mlflow.get_tracking_uri()}")
    click.echo()
    click.echo("Documentation:")
    click.echo("  GitHub: https://github.com/sdeery14/llm-system-lifecycle")
    click.echo()


if __name__ == "__main__":
    cli()
