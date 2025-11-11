"""
Dataset Builder Agent using OpenAI Agents SDK.

This agent assists users in creating high-quality evaluation datasets for target agents.
It logs the target agent in MLflow, chats with users to understand requirements,
and creates and stores datasets in MLflow.
"""

from __future__ import annotations
import os
import asyncio
import ast
import json
from pathlib import Path
from typing import Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import mlflow
from mlflow.genai.datasets import create_dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ConfigDict
from agents import Agent, function_tool, RunContextWrapper, Runner

# Load environment variables
load_dotenv()


# ==================== HELPER FOR CONFIGURATION ====================

def _parse_max_dataset_size() -> int:
    """
    Parse the MAX_DATASET_INSTANCES environment variable with validation.
    
    Returns:
        The max dataset size from the environment variable, or 100 if invalid/not set.
    """
    try:
        return int(os.getenv("MAX_DATASET_INSTANCES", "100"))
    except (ValueError, TypeError):
        # If the env var is set to an invalid value, return default
        return 100


# ==================== CONFIGURATION ====================

class DatasetBuilderConfig(BaseModel):
    """Configuration for the dataset builder agent."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    max_dataset_size: int = Field(
        default_factory=lambda: _parse_max_dataset_size(),
        description="Maximum number of dataset instances allowed"
    )
    batch_size: int = Field(
        default=20,
        description="Create datasets in batches of this size"
    )
    model: str = Field(
        default="gpt-4o",
        description="LLM model to use for main agent"
    )
    worker_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use for worker agents (cheaper/faster)"
    )
    enable_parallel_generation: bool = Field(
        default=True,
        description="Enable parallel batch generation using worker agents"
    )
    enable_checkpointing: bool = Field(
        default=True,
        description="Enable checkpointing for resumable dataset creation"
    )
    checkpoint_interval: int = Field(
        default=50,
        description="Save checkpoint every N test cases"
    )
    checkpoint_dir: Path = Field(
        default_factory=lambda: Path("data/checkpoints"),
        description="Directory to store checkpoints"
    )
    enable_diversity_check: bool = Field(
        default=True,
        description="Enable structural diversity checks to avoid duplicate patterns"
    )
    diversity_window: int = Field(
        default=20,
        description="Number of recent cases to check for diversity"
    )
    incremental_mlflow_logging: bool = Field(
        default=True,
        description="Log batches to MLflow incrementally instead of all at once"
    )


# ==================== PYDANTIC MODELS ====================

class TestCaseInputs(BaseModel):
    """Input structure for a single test case."""
    model_config = ConfigDict(extra='forbid', strict=True)
    
    query: str = Field(
        description="The user query or input text"
    )
    context: str = Field(
        default="",
        description="Any relevant context for the query (optional)"
    )
    category: str = Field(
        description="Category name this test case belongs to"
    )
    test_scenario: str = Field(
        description="Brief description of what this test case tests"
    )


class TestCaseExpectations(BaseModel):
    """Expected outputs/behaviors for a test case."""
    model_config = ConfigDict(extra='forbid', strict=True)
    
    answer: str = Field(
        default="",
        description="Expected answer or response text"
    )
    tool_calls: list[str] = Field(
        default_factory=list,
        description="List of expected tool names the agent should call"
    )


class TestCase(BaseModel):
    """A single test case with inputs and expectations."""
    model_config = ConfigDict(extra='forbid', strict=True)
    
    inputs: TestCaseInputs = Field(
        description="Test case inputs"
    )
    expectations: TestCaseExpectations = Field(
        description="Expected outputs/behaviors"
    )


class TestCaseBatch(BaseModel):
    """A batch of generated test cases."""
    model_config = ConfigDict(extra='forbid', strict=True)
    
    test_cases: list[TestCase] = Field(
        description="List of generated test cases"
    )


class DatasetCategory(BaseModel):
    """Definition of a dataset category with all required fields."""
    model_config = ConfigDict(extra='forbid')
    
    name: str = Field(
        ..., 
        description="Category name (e.g., 'order_status', 'refunds')"
    )
    description: str = Field(
        ..., 
        description="What this category tests"
    )
    count: int = Field(
        ..., 
        gt=0,
        description="Number of test cases for this category"
    )
    example_inputs: str = Field(
        ..., 
        description="Example inputs for this category to guide test case generation"
    )
    expectations: str = Field(
        ..., 
        description=(
            "Expected outputs/behaviors for this category. "
            "Can be a JSON string with 'answer' and 'tool_calls' fields for agent evaluation. "
            "Example: {\"answer\": \"expected response\", \"tool_calls\": [\"tool1\", \"tool2\"]}"
        )
    )
    expected_tools: list[str] | None = Field(
        default=None,
        description=(
            "List of tool names the agent should call for this category. "
            "This will be merged into expectations as 'tool_calls' during dataset creation."
        )
    )
    
    @field_validator('count')
    @classmethod
    def count_must_be_positive(cls, v: int) -> int:
        """Validate that count is positive."""
        if v <= 0:
            raise ValueError('count must be greater than 0')
        return v


class DatasetPlanInput(BaseModel):
    """Input model for creating a dataset plan."""
    model_config = ConfigDict(extra='forbid')
    
    dataset_name: str = Field(
        ..., 
        description="Name for the dataset (e.g., 'customer_service_regression_v1')"
    )
    categories: list[DatasetCategory] = Field(
        ..., 
        min_length=1,
        description="List of category definitions"
    )
    total_instances: int = Field(
        ..., 
        gt=0,
        description="Total number of test instances to create"
    )
    
    @field_validator('total_instances')
    @classmethod
    def validate_total_instances(cls, v: int) -> int:
        """Validate that total instances is positive."""
        if v <= 0:
            raise ValueError('total_instances must be greater than 0')
        return v
    
    @field_validator('categories')
    @classmethod
    def validate_category_counts(cls, v: list[DatasetCategory]) -> list[DatasetCategory]:
        """Validate that categories is not empty."""
        if not v:
            raise ValueError('At least one category must be defined')
        return v


class TestCasesBatchInput(BaseModel):
    """Input model for generating a batch of test cases."""
    model_config = ConfigDict(extra='forbid')
    
    category_name: str = Field(
        ...,
        description="Name of the category (e.g., 'order_status')"
    )
    num_cases: int = Field(
        ...,
        gt=0,
        description="Number of test cases to generate in this batch (max 20)"
    )
    category_description: str = Field(
        ...,
        description="Description of what this category tests"
    )
    example_inputs: str = Field(
        ...,
        description="Example inputs to guide generation"
    )
    expectations: str = Field(
        ...,
        description="Expected outputs/behaviors for this category (describe as text or JSON string)"
    )
    
    @field_validator('num_cases')
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate that num_cases is positive and not too large."""
        if v <= 0:
            raise ValueError('num_cases must be greater than 0')
        # Note: max batch size will be checked against config in the function
        return v


# ==================== STATE MANAGEMENT ====================

class DatasetBuilderState(BaseModel):
    """State tracking for dataset builder agent."""
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
    
    target_agent_name: str | None = None
    target_agent_description: str | None = None
    target_agent_logged: bool = False
    target_agent_run_id: str | None = None
    
    dataset_plan: dict[str, Any] | None = None
    categories_approved: bool = False
    
    total_instances_planned: int = 0
    instances_created: int = 0
    created_instances: list[dict[str, Any]] = Field(default_factory=list)
    
    dataset_name: str | None = None
    dataset_experiment_id: str | None = None
    dataset_created: bool = False
    
    # Checkpointing state
    checkpoint_file: Path | None = None
    last_checkpoint_at: int = 0
    
    # Diversity tracking
    recent_case_hashes: list[str] = Field(default_factory=list)
    diversity_rejections: int = 0
    
    def reset_dataset_creation(self) -> None:
        """Reset dataset creation state while keeping the plan."""
        self.instances_created = 0
        self.created_instances = []
        self.dataset_created = False
        self.last_checkpoint_at = 0
        self.recent_case_hashes = []
        self.diversity_rejections = 0
    
    def to_summary(self) -> str:
        """Generate a summary of the current state."""
        summary = []
        if self.target_agent_name:
            summary.append(f"Target Agent: {self.target_agent_name}")
            summary.append(f"  Logged: {self.target_agent_logged}")
        
        if self.dataset_plan:
            summary.append("\nDataset Plan:")
            summary.append(f"  Name: {self.dataset_plan.get('name', 'N/A')}")
            summary.append(f"  Total Instances: {self.total_instances_planned}")
            summary.append(f"  Categories: {len(self.dataset_plan.get('categories', []))}")
        
        summary.append("\nProgress:")
        summary.append(f"  Instances Created: {self.instances_created}/{self.total_instances_planned}")
        summary.append(f"  Dataset Finalized: {self.dataset_created}")
        
        if self.diversity_rejections > 0:
            summary.append(f"  Diversity Rejections: {self.diversity_rejections}")
        
        if self.checkpoint_file:
            summary.append(f"\nCheckpoint: {self.checkpoint_file}")
            summary.append(f"  Last checkpoint at: {self.last_checkpoint_at} cases")
        
        return "\n".join(summary)


# Module-level state
_builder_state = DatasetBuilderState()


# ==================== HELPER FUNCTIONS ====================

def _migrate_test_case_structure(case: dict[str, Any]) -> dict[str, Any]:
    """
    Migrate old test case structure to new MLflow-compatible format.
    
    Old structure (with metadata field):
        {
            "inputs": {...},
            "expectations": {...},
            "metadata": {...}  # Not supported by MLflow
        }
    
    New structure (MLflow-compatible):
        {
            "inputs": {...},  # metadata fields merged here
            "expectations": {...}
        }
    
    Args:
        case: Test case dict (may have old or new structure)
    
    Returns:
        Migrated test case dict
    """
    if "metadata" not in case:
        # Already correct structure
        return case
    
    # Migrate: remove metadata field, merge into inputs
    migrated_case = {
        "inputs": dict(case.get("inputs", {})),  # Copy inputs
        "expectations": case.get("expectations", {})
    }
    
    # Merge metadata fields into inputs (avoid overwriting existing fields)
    metadata = case.get("metadata", {})
    for key, value in metadata.items():
        if key not in migrated_case["inputs"]:
            migrated_case["inputs"][key] = value
    
    return migrated_case

def _compute_case_hash(test_case: dict[str, Any]) -> str:
    """
    Compute a structural hash for a test case to check for duplicates.
    
    This uses a simple hash of the input structure to detect similar patterns.
    
    Args:
        test_case: The test case to hash.
    
    Returns:
        Hash string representing the case structure.
    """
    # Extract key structural elements
    inputs = test_case.get("inputs", {})
    
    # Create a simplified representation focusing on structure
    structure_key = f"{inputs.get('category', '')}_{len(str(inputs))}"
    
    # For more sophisticated diversity, could use the actual input text
    input_text = str(inputs).lower()
    
    return f"{structure_key}_{hash(input_text) % 10000}"


def _is_diverse_enough(test_case: dict[str, Any], recent_cases: list[str], threshold: int = 3) -> bool:
    """
    Check if a test case is structurally diverse from recent cases.
    
    Args:
        test_case: The test case to check.
        recent_cases: List of recent case hashes.
        threshold: Minimum number of hash differences required.
    
    Returns:
        True if the case is diverse enough.
    """
    if not recent_cases:
        return True
    
    case_hash = _compute_case_hash(test_case)
    
    # Check if this exact hash appears in recent cases
    if case_hash in recent_cases:
        return False
    
    # For structural diversity, we just ensure it's not identical
    return True


def _save_checkpoint(checkpoint_file: Path, state_data: dict[str, Any]) -> None:
    """
    Save a checkpoint of the current generation state.
    
    Args:
        checkpoint_file: Path to save the checkpoint.
        state_data: State data to save.
    """
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(checkpoint_file, 'w') as f:
        json.dump(state_data, f, indent=2, default=str)


def _load_checkpoint(checkpoint_file: Path) -> dict[str, Any] | None:
    """
    Load a checkpoint if it exists.
    
    Args:
        checkpoint_file: Path to the checkpoint file.
    
    Returns:
        Checkpoint data or None if file doesn't exist.
    """
    if not checkpoint_file.exists():
        return None
    
    with open(checkpoint_file, 'r') as f:
        return json.load(f)


async def _generate_batch_with_worker(
    category: dict[str, Any],
    num_cases: int,
    worker_model: str = "gpt-4o-mini",
    batch_index: int = 0
) -> list[dict[str, Any]]:
    """
    Generate a batch of test cases using a worker agent with structured outputs.
    
    This function creates a separate worker agent that uses Pydantic-defined output types
    to generate properly structured test cases with guaranteed JSON schema compliance.
    
    Args:
        category: Category information including name, description, examples, expectations.
        num_cases: Number of test cases to generate.
        worker_model: Model to use for the worker agent.
        batch_index: Index of this batch for tracking.
    
    Returns:
        List of generated test cases.
    """
    # Parse expectations if it's a JSON string
    expectations_data = category.get('expectations', {})
    if isinstance(expectations_data, str):
        try:
            expectations_data = json.loads(expectations_data)
        except (json.JSONDecodeError, ValueError):
            # If it's not valid JSON, create a simple dict with the string as answer
            expectations_data = {"answer": expectations_data}
    elif not isinstance(expectations_data, dict):
        # If it's neither string nor dict, wrap it
        expectations_data = {"answer": str(expectations_data)}
    
    # Add tool_calls if provided in category (for agent evaluation)
    if category.get('expected_tools') and 'tool_calls' not in expectations_data:
        expectations_data['tool_calls'] = category['expected_tools']
    
    # Build the worker instructions
    worker_instructions = f"""You are a test case generator. Your task is to generate {num_cases} diverse, 
realistic test cases for the following category:

Category: {category['name']}
Description: {category['description']}
Example Inputs: {category.get('example_inputs', 'N/A')}

Requirements:
1. Generate exactly {num_cases} test cases
2. Make each case unique and realistic
3. Vary the input patterns, edge cases, and scenarios
4. Follow the category description and examples
5. Each test case should test different aspects or edge cases

For each test case:
- The "query" should be a realistic user input
- The "context" can provide additional context if needed (can be empty)
- The "category" must be "{category['name']}"
- The "test_scenario" should briefly describe what specific aspect this test covers

Expected behavior for this category:
{json.dumps(expectations_data, indent=2)}

Generate diverse, realistic test cases that would thoroughly test an agent's capabilities."""

    # Create a worker agent with structured output type
    worker = Agent(
        name=f"TestCaseWorker_{category['name']}_{batch_index}",
        model=worker_model,
        instructions=worker_instructions,
        output_type=TestCaseBatch  # Use Pydantic model for structured outputs
    )
    
    # Generate test cases with structured outputs
    result = await Runner.run(
        worker, 
        f"Generate {num_cases} diverse test cases for category '{category['name']}'"
    )
    
    # Extract the structured output
    test_case_batch = result.final_output
    
    # Convert Pydantic models to dicts for MLflow compatibility
    generated_cases = []
    for test_case in test_case_batch.test_cases:
        # Convert to dict and clean up empty defaults
        expectations_dict = test_case.expectations.model_dump()
        # Remove empty defaults to keep the output clean
        if not expectations_dict.get('answer'):
            expectations_dict.pop('answer', None)
        if not expectations_dict.get('tool_calls'):
            expectations_dict.pop('tool_calls', None)
        
        case_dict = {
            "inputs": test_case.inputs.model_dump(),
            "expectations": expectations_dict
        }
        generated_cases.append(case_dict)
    
    # Debug: Verify structure is correct
    if generated_cases:
        print(f"\n[DEBUG] Generated {len(generated_cases)} test cases for '{category['name']}'")
        print(f"[DEBUG] First case structure: {json.dumps(generated_cases[0], indent=2)}")
        print(f"[DEBUG] Keys in first case: {list(generated_cases[0].keys())}\n")
    
    return generated_cases


async def _generate_batches_parallel(
    category: dict[str, Any],
    total_cases: int,
    batch_size: int,
    worker_model: str = "gpt-4o-mini"
) -> list[dict[str, Any]]:
    """
    Generate test cases in parallel batches for scalability.
    
    Args:
        category: Category information.
        total_cases: Total number of cases to generate for this category.
        batch_size: Size of each batch.
        worker_model: Model to use for worker agents.
    
    Returns:
        List of all generated test cases.
    """
    # Calculate number of batches needed
    num_batches = (total_cases + batch_size - 1) // batch_size
    
    # Create tasks for parallel generation
    tasks = []
    for batch_idx in range(num_batches):
        cases_in_batch = min(batch_size, total_cases - (batch_idx * batch_size))
        task = _generate_batch_with_worker(
            category=category,
            num_cases=cases_in_batch,
            worker_model=worker_model,
            batch_index=batch_idx
        )
        tasks.append(task)
    
    # Run all batches in parallel
    batch_results = await asyncio.gather(*tasks)
    
    # Flatten results
    all_cases = []
    for batch in batch_results:
        all_cases.extend(batch)
    
    return all_cases


# ==================== TOOLS ====================

def _create_mlflow_agent_file(
    agent_file_path: str,
    agent_class_name: str,
    output_dir: Path
) -> Path:
    """
    Create a self-contained MLflow-compatible agent file by inlining all dependencies.
    
    This function uses AST parsing to properly extract and combine the source agent
    file with the MLflow wrapper template, creating a single file with no external 
    dependencies (except pip packages).
    
    Args:
        agent_file_path: Path to the source agent file (relative to project root).
        agent_class_name: Name of the agent class.
        output_dir: Directory to save the generated file.
    
    Returns:
        Path to the created MLflow-compatible agent file.
    """
    project_root = Path(__file__).parent.parent.parent
    templates_dir = project_root / "scripts" / "templates"
    
    # Helper function to extract code using AST
    def extract_code_ast(source_file: Path) -> tuple[list[str], str]:
        """Extract imports and code from file using AST parsing."""
        with open(source_file, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source)
        
        # Separate imports from other code
        imports = []
        code_nodes = []
        
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Only include top-level imports
                imports.append(ast.unparse(node))
            elif isinstance(node, ast.If):
                # Skip if __name__ == "__main__" blocks
                if (isinstance(node.test, ast.Compare) and
                    isinstance(node.test.left, ast.Name) and
                    node.test.left.id == '__name__'):
                    continue
                else:
                    code_nodes.append(node)
            else:
                code_nodes.append(node)
        
        # Unparse the code nodes back to source
        code_lines = []
        for node in code_nodes:
            try:
                code_lines.append(ast.unparse(node))
            except Exception:
                # Skip nodes that can't be unparsed
                pass
        
        code = '\n\n'.join(code_lines)
        return imports, code
    
    # Read and parse source files
    agent_source_path = project_root / agent_file_path
    template_file = templates_dir / "mlflow_responses_agent_wrapper.py"
    
    agent_imports, agent_code = extract_code_ast(agent_source_path)
    wrapper_imports, wrapper_code = extract_code_ast(template_file)
    
    # Combine imports (deduplicate while preserving order)
    all_imports = list(dict.fromkeys(agent_imports + wrapper_imports))
    
    # Separate future imports
    future_imports = [imp for imp in all_imports if imp.startswith('from __future__')]
    regular_imports = [imp for imp in all_imports if not imp.startswith('from __future__')]
    
    # Build the final file
    parts = [
        '"""',
        f'MLflow-compatible {agent_class_name} for serving.',
        '',
        'This file is auto-generated by the Dataset Builder Agent.',
        'All dependencies are inlined to avoid import issues when served by MLflow.',
        '"""',
        ''
    ]
    
    # Add future imports first
    if future_imports:
        parts.extend(future_imports)
        parts.append('')
    
    # Add regular imports
    parts.extend(regular_imports)
    parts.append('')
    parts.append('')
    
    # Add agent code
    parts.append('# ==================== AGENT CODE (INLINED) ====================')
    parts.append('')
    parts.append(agent_code)
    parts.append('')
    parts.append('')
    
    # Add wrapper code
    parts.append('# ==================== MLFLOW WRAPPER ====================')
    parts.append('')
    parts.append(wrapper_code)
    parts.append('')
    parts.append('')
    
    # Add instantiation
    parts.append('# ==================== AGENT INSTANTIATION ====================')
    parts.append('')
    parts.append('from mlflow.models import set_model')
    parts.append('')
    parts.append('agent = MLflowResponsesAgentWrapper(')
    parts.append(f'    agent_class={agent_class_name},')
    parts.append('    model="gpt-4o"')
    parts.append(')')
    parts.append('set_model(agent)')
    
    combined_code = '\n'.join(parts)
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the combined file
    output_file = output_dir / f"{agent_class_name.lower()}_mlflow.py"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_code)
    
    return output_file


@function_tool
def analyze_target_agent_metadata(
    ctx: RunContextWrapper[Any],
    agent_file_path: str,
    agent_class_name: str
) -> str:
    """
    Analyze the target agent by instantiating it and extracting metadata from the Agent object.
    
    This function dynamically imports and instantiates the agent class (built with OpenAI Agents SDK),
    then extracts:
    - Agent instructions/system prompt
    - Available tools with descriptions
    - Model configuration
    - Agent name and other attributes
    
    Args:
        agent_file_path: Path to the Python file containing the agent class (relative to project root).
        agent_class_name: Name of the agent class (e.g., 'CustomerServiceAgent').
    
    Returns:
        A detailed analysis of the agent's metadata formatted for user review.
    """
    try:
        project_root = Path(__file__).parent.parent.parent
        full_agent_path = project_root / agent_file_path
        
        if not full_agent_path.exists():
            return f"✗ Error: Agent file not found at {full_agent_path}"
        
        # Import the agent class dynamically
        import sys
        import importlib.util
        
        # Add src directory to path if not already there
        src_dir = str(project_root / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        
        # Load the module
        spec = importlib.util.spec_from_file_location("target_agent_module", full_agent_path)
        if spec is None or spec.loader is None:
            return f"✗ Error: Could not load module from {full_agent_path}"
        
        module = importlib.util.module_from_spec(spec)
        
        # Register the module in sys.modules before executing
        # This is required for decorators like @dataclass to work properly
        sys.modules[spec.name] = module
        
        try:
            spec.loader.exec_module(module)
        except Exception:
            # Clean up if execution fails
            sys.modules.pop(spec.name, None)
            raise
        
        # Get the agent class
        if not hasattr(module, agent_class_name):
            # Clean up the module from sys.modules
            sys.modules.pop(spec.name, None)
            return f"✗ Error: Class '{agent_class_name}' not found in {agent_file_path}"
        
        agent_class = getattr(module, agent_class_name)
        
        # Instantiate the agent (typically with default parameters)
        try:
            agent_instance = agent_class()
        except TypeError:
            # If it requires parameters, try with a model parameter
            agent_instance = agent_class(model="gpt-4o")
        
        # Get the underlying Agent object
        if hasattr(agent_instance, 'get_agent'):
            agent = agent_instance.get_agent()
        elif hasattr(agent_instance, 'agent'):
            agent = agent_instance.agent
        else:
            # Assume the instance itself is the agent
            agent = agent_instance
        
        # Extract metadata from the Agent object
        analysis = {
            "agent_class": agent_class_name,
            "file_path": agent_file_path,
            "name": getattr(agent, 'name', 'Unknown'),
            "instructions": getattr(agent, 'instructions', None),
            "model": getattr(agent, 'model', 'Unknown'),
            "tools": [],
            "docstring": agent_class.__doc__
        }
        
        # Extract tool information
        if hasattr(agent, 'tools') and agent.tools:
            for tool in agent.tools:
                tool_info = {
                    "name": getattr(tool, 'name', getattr(tool, '__name__', 'Unknown')),
                    "description": getattr(tool, 'description', None)
                }
                
                # If description is None, try to get it from the function docstring
                if tool_info["description"] is None and hasattr(tool, '__doc__'):
                    doc = tool.__doc__
                    if doc:
                        # Get first line of docstring as description
                        tool_info["description"] = doc.strip().split('\n')[0].strip()
                
                analysis["tools"].append(tool_info)
        
        # Format the analysis for display
        result = [
            "=" * 70,
            f"📊 Target Agent Analysis: {agent_class_name}",
            "=" * 70,
            ""
        ]
        
        if analysis["docstring"]:
            result.append("**Agent Purpose:**")
            result.append(analysis["docstring"].strip())
            result.append("")
        
        result.append(f"**Agent Name:** {analysis['name']}")
        result.append(f"**Model:** {analysis['model']}")
        result.append("")
        
        if analysis["instructions"]:
            result.append("**Agent Instructions (System Prompt):**")
            # Truncate long instructions for readability
            instructions = analysis["instructions"]
            if len(instructions) > 500:
                instructions = instructions[:500] + "...\n[Truncated - full instructions captured]"
            result.append(instructions)
            result.append("")
        
        if analysis["tools"]:
            result.append(f"**Available Tools ({len(analysis['tools'])}):**")
            for i, tool in enumerate(analysis["tools"], 1):
                desc = tool['description'] if tool['description'] else "No description available"
                result.append(f"{i}. `{tool['name']}`: {desc}")
            result.append("")
        
        result.append("=" * 70)
        result.append("")
        result.append("**Based on this analysis, here's my understanding of what to test:**")
        result.append("")
        
        # Generate suggested test categories based on tools
        if analysis["tools"]:
            result.append("**Suggested Test Categories (with expected tool usage):**")
            for i, tool in enumerate(analysis["tools"], 1):
                # Derive category from tool name
                category_name = tool['name'].replace('_', ' ').title()
                desc = tool['description'] if tool['description'] else "this functionality"
                result.append(f"{i}. **{category_name}** - Test {desc.lower()}")
                result.append(f"   Expected tools: [`{tool['name']}`]")
            result.append("")
            result.append("💡 **Tip**: Including expected tool calls in your dataset enables MLflow's")
            result.append("   trace-based evaluation to verify not just the final answer, but also")
            result.append("   whether the agent used the correct tools to arrive at that answer.")
            result.append("")
        
        # Extract key capabilities from instructions
        if analysis["instructions"]:
            result.append("**Key Capabilities (from instructions):**")
            instructions_lower = analysis["instructions"].lower()
            
            capabilities = []
            if "order" in instructions_lower:
                capabilities.append("- Order management and tracking")
            if "refund" in instructions_lower or "return" in instructions_lower:
                capabilities.append("- Refund and return processing")
            if "account" in instructions_lower or "balance" in instructions_lower:
                capabilities.append("- Account information and management")
            if "knowledge" in instructions_lower or "search" in instructions_lower:
                capabilities.append("- Knowledge base search and Q&A")
            if "contact" in instructions_lower or "update" in instructions_lower:
                capabilities.append("- Contact information updates")
            
            if capabilities:
                result.extend(capabilities)
            else:
                result.append("- [General agent capabilities based on tools listed above]")
            result.append("")
        
        result.append("**Next Steps:**")
        result.append("Please review this analysis and let me know:")
        result.append("1. Are these the right categories to test?")
        result.append("2. Are there any edge cases or scenarios I should add?")
        result.append("3. How many test cases would you like per category?")
        result.append("4. What specific behaviors or outputs should I validate?")
        result.append("")
        
        # Clean up: remove the module from sys.modules to avoid conflicts
        sys.modules.pop(spec.name, None)
        
        return "\n".join(result)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"✗ Error analyzing agent metadata: {str(e)}\n\nDetails:\n{error_details}"


@function_tool
def log_target_agent_in_mlflow(
    ctx: RunContextWrapper[Any],
    agent_file_path: str,
    agent_class_name: str,
    agent_description: str,
    experiment_name: str = "dataset-builder-targets"
) -> str:
    """
    Log a target agent in MLflow for tracking and reference using models-from-code.
    
    This function creates an MLflow-compatible agent file and logs it as a model
    so that datasets can be associated with specific agent versions.
    
    Args:
        agent_file_path: Path to the Python file containing the agent class (relative to project root).
        agent_class_name: Name of the agent class (e.g., 'CustomerServiceAgent').
        agent_description: Description of what the agent does.
        experiment_name: MLflow experiment name (default: 'dataset-builder-targets').
    
    Returns:
        Confirmation message with the run ID.
    """
    global _builder_state
    
    def _log_in_thread():
        """Execute MLflow logging in a separate thread to avoid event loop conflicts."""
        try:
            # Set up paths
            project_root = Path(__file__).parent.parent.parent
            temp_agents_dir = project_root / "src" / "app_agents" / "temp_mlflow_agents"
            
            # Create MLflow-compatible agent file from template
            mlflow_agent_file = _create_mlflow_agent_file(
                agent_file_path=agent_file_path,
                agent_class_name=agent_class_name,
                output_dir=temp_agents_dir
            )
            
            # Get the relative path from project root for MLflow
            relative_agent_path = mlflow_agent_file.relative_to(project_root)
            
            # Set the experiment
            mlflow.set_experiment(experiment_name)
            
            # Start an MLflow run and log the model
            with mlflow.start_run(run_name=f"{agent_class_name}_reference") as run:
                run_id = run.info.run_id
                
                # Log the agent using models-from-code
                logged_agent_info = mlflow.pyfunc.log_model(
                    python_model=str(relative_agent_path),
                    artifact_path="agent",
                    pip_requirements=[
                        "mlflow>=3.4.0",
                        "pydantic>=2.0.0",
                        "openai-agents>=0.3.0",
                        "faiss-cpu>=1.12.0",
                        "sentence-transformers>=5.0.0",
                        "numpy>=2.0.0",
                        "python-dotenv>=1.0.0",
                    ],
                    metadata={
                        "task": "agent_evaluation",
                        "agent_class": agent_class_name,
                        "description": agent_description,
                        "purpose": "dataset_creation_reference"
                    },
                )
                
                # Log additional parameters
                mlflow.log_param("agent_class", agent_class_name)
                mlflow.log_param("agent_file", agent_file_path)
                mlflow.log_param("model_type", "gpt-4o")
                mlflow.set_tag("description", agent_description)
                mlflow.set_tag("purpose", "dataset_creation_reference")
                mlflow.set_tag("timestamp", datetime.now().isoformat())
                
                # Try to log the original source file as an artifact
                full_agent_path = project_root / agent_file_path
                if full_agent_path.exists():
                    mlflow.log_artifact(str(full_agent_path), artifact_path="original_source")
                
                # Update state
                _builder_state.target_agent_name = agent_class_name
                _builder_state.target_agent_description = agent_description
                _builder_state.target_agent_logged = True
                _builder_state.target_agent_run_id = run_id
                
                return {
                    "success": True,
                    "run_id": run_id,
                    "model_uri": logged_agent_info.model_uri,
                    "experiment": experiment_name
                }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Run in a thread pool to avoid event loop conflicts
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_log_in_thread)
        result = future.result(timeout=120)  # 2 minute timeout
    
    if result["success"]:
        return (f"✓ Successfully logged {agent_class_name} in MLflow using models-from-code.\n"
                f"  Run ID: {result['run_id']}\n"
                f"  Experiment: {result['experiment']}\n"
                f"  Model URI: {result['model_uri']}\n"
                f"  Description: {agent_description}\n\n"
                f"The agent is now tracked in MLflow and ready for dataset creation!")
    else:
        return f"✗ Error logging agent in MLflow: {result['error']}"


@function_tool
def use_previously_logged_agent(
    ctx: RunContextWrapper[Any],
    run_id: str
) -> str:
    """
    Use a previously logged agent from MLflow instead of logging a new one.
    
    This saves time by reusing an agent that was already logged in MLflow.
    You can get the run_id from the MLflow UI or by listing recent runs.
    
    Args:
        run_id: The MLflow run ID of the previously logged agent.
    
    Returns:
        Confirmation message with agent details.
    """
    global _builder_state
    
    try:
        # Get the run details from MLflow
        run = mlflow.get_run(run_id)
        
        # Extract agent information from the run
        agent_class = run.data.params.get('agent_class', 'Unknown')
        agent_file = run.data.params.get('agent_file', 'Unknown')
        description = run.data.tags.get('description', 'A target agent for evaluation')
        
        # Update state
        _builder_state.target_agent_name = agent_class
        _builder_state.target_agent_description = description
        _builder_state.target_agent_logged = True
        _builder_state.target_agent_run_id = run_id
        
        return (f"✓ Using previously logged agent: {agent_class}\n"
                f"  Run ID: {run_id}\n"
                f"  Agent File: {agent_file}\n"
                f"  Description: {description}\n\n"
                f"The agent is ready for dataset creation! "
                f"I'll now analyze it to suggest test categories.")
    
    except Exception as e:
        return f"✗ Error loading agent from run {run_id}: {str(e)}\n\nPlease check the run ID and try again."


@function_tool
def list_previously_logged_agents(
    ctx: RunContextWrapper[Any],
    max_results: int = 10
) -> str:
    """
    List recently logged agents from MLflow.
    
    This helps you find the run_id of a previously logged agent to reuse it.
    
    Args:
        max_results: Maximum number of recent agents to show (default: 10).
    
    Returns:
        List of recently logged agents with their details.
    """
    try:
        # Get the dataset-builder-targets experiment
        experiment = mlflow.get_experiment_by_name("dataset-builder-targets")
        
        if not experiment:
            return ("✗ No 'dataset-builder-targets' experiment found.\n\n"
                    "You haven't logged any agents yet. Use log_target_agent_in_mlflow to log your first agent.")
        
        # Search for recent runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=max_results
        )
        
        if runs.empty:
            return "✗ No logged agents found in the 'dataset-builder-targets' experiment."
        
        # Format the results
        result = [
            f"Recently Logged Agents ({len(runs)} found):",
            "=" * 70,
            ""
        ]
        
        for idx, row in runs.iterrows():
            agent_class = row.get('params.agent_class', 'Unknown')
            agent_file = row.get('params.agent_file', 'Unknown')
            run_id = row['run_id']
            start_time = row['start_time'].strftime('%Y-%m-%d %H:%M:%S') if 'start_time' in row else 'Unknown'
            description = row.get('tags.description', 'No description')
            
            result.append(f"{idx + 1}. {agent_class}")
            result.append(f"   Run ID: {run_id}")
            result.append(f"   File: {agent_file}")
            result.append(f"   Logged: {start_time}")
            result.append(f"   Description: {description}")
            result.append("")
        
        result.append("To use one of these agents, call: use_previously_logged_agent(run_id='<run_id>')")
        
        return "\n".join(result)
    
    except Exception as e:
        return f"✗ Error listing logged agents: {str(e)}"


@function_tool
async def create_dataset_plan(
    ctx: RunContextWrapper[Any],
    plan: DatasetPlanInput
) -> str:
    """
    Create a plan for the evaluation dataset based on user discussion.
    
    This function validates and stores the dataset plan, including categories
    and the distribution of test cases.
    
    Args:
        plan: The dataset plan containing dataset_name, categories, and total_instances.
            Each category should include:
            - name: Category name (e.g., 'order_status')
            - description: What this category tests
            - count: Number of test cases for this category
            - example_inputs: Example inputs for this category
            - expectations: Expected outputs/behaviors for this category
    
    Returns:
        Summary of the dataset plan for user approval.
    """
    global _builder_state
    
    # Get max allowed from config
    config = DatasetBuilderConfig()
    
    # Validate total instances against config
    if plan.total_instances > config.max_dataset_size:
        return (f"✗ Error: Total instances ({plan.total_instances}) exceeds maximum allowed "
                f"({config.max_dataset_size}). Please reduce the dataset size or increase "
                f"the MAX_DATASET_INSTANCES environment variable.")
    
    # Validate categories sum to total
    category_sum = sum(cat.count for cat in plan.categories)
    if category_sum != plan.total_instances:
        return (f"✗ Error: Category counts ({category_sum}) don't match total instances "
                f"({plan.total_instances}). Please adjust the distribution.")
    
    # Store the plan (convert Pydantic models to dicts for state storage)
    # Merge expected_tools into expectations if provided
    categories_data = []
    for cat in plan.categories:
        cat_dict = cat.model_dump()
        # If expected_tools is provided, ensure it's included in expectations
        if cat.expected_tools:
            # Store expected_tools in the category for worker agents to use
            cat_dict['expected_tools'] = cat.expected_tools
        categories_data.append(cat_dict)
    
    _builder_state.dataset_plan = {
        'name': plan.dataset_name,
        'categories': categories_data,
        'total_instances': plan.total_instances
    }
    _builder_state.total_instances_planned = plan.total_instances
    _builder_state.dataset_name = plan.dataset_name
    
    # Generate summary
    summary = [
        f"📋 Dataset Plan Created: {plan.dataset_name}",
        f"\nTotal Test Cases: {plan.total_instances}",
        "\nCategories:"
    ]
    
    for i, cat in enumerate(plan.categories, 1):
        summary.append(f"\n{i}. {cat.name} ({cat.count} cases)")
        summary.append(f"   Description: {cat.description}")
        if cat.example_inputs:
            example_preview = cat.example_inputs[:100] + "..." if len(cat.example_inputs) > 100 else cat.example_inputs
            summary.append(f"   Examples: {example_preview}")
        if cat.expected_tools:
            summary.append(f"   Expected Tools: {', '.join(cat.expected_tools)}")
    
    summary.append("\n\nThis plan will be executed in batches of up to 20 test cases at a time.")
    summary.append("\nPlease review and let me know if you'd like to proceed or make changes.")
    
    return "\n".join(summary)


@function_tool
async def approve_dataset_plan(ctx: RunContextWrapper[Any]) -> str:
    """
    Approve the current dataset plan and prepare for dataset creation.
    
    Returns:
        Confirmation message.
    """
    global _builder_state
    
    if not _builder_state.dataset_plan:
        return "✗ Error: No dataset plan exists. Please create a plan first."
    
    _builder_state.categories_approved = True
    
    return (f"✓ Dataset plan approved!\n\n"
            f"I will now create {_builder_state.total_instances_planned} test cases "
            f"in batches of up to {DatasetBuilderConfig().batch_size}.\n\n"
            f"This ensures high quality and stays within LLM context limits.")


@function_tool
async def generate_test_cases_batch(
    ctx: RunContextWrapper[Any],
    batch: TestCasesBatchInput
) -> str:
    """
    Generate a batch of test cases for a specific category.
    
    This is an internal tool that generates test cases based on the approved plan.
    The agent should call this multiple times to build up the full dataset.
    Uses worker agents for parallel generation and includes diversity checking.
    
    Args:
        batch: The batch input containing category_name, num_cases, category_description,
            example_inputs, and expectations for the test cases to generate.
    
    Returns:
        Summary of generated test cases.
    """
    global _builder_state
    
    config = DatasetBuilderConfig()
    
    # Validate batch size
    if batch.num_cases > config.batch_size:
        return f"✗ Error: Batch size ({batch.num_cases}) exceeds maximum ({config.batch_size}). Please reduce."
    
    # Validate plan approved
    if not _builder_state.categories_approved:
        return "✗ Error: Dataset plan not approved. Please approve the plan first."
    
    # Check if we're exceeding the total planned
    if _builder_state.instances_created + batch.num_cases > _builder_state.total_instances_planned:
        remaining = _builder_state.total_instances_planned - _builder_state.instances_created
        return (f"✗ Error: Generating {batch.num_cases} cases would exceed the planned total. "
                f"Only {remaining} cases remaining.")
    
    # Prepare category info for worker
    category_info = {
        'name': batch.category_name,
        'description': batch.category_description,
        'example_inputs': batch.example_inputs,
        'expectations': batch.expectations
    }
    
    # Generate test cases using worker agents with parallel generation if enabled
    if config.enable_parallel_generation and batch.num_cases > 5:
        # For larger batches, use parallel generation
        generated_cases = await _generate_batches_parallel(
            category=category_info,
            total_cases=batch.num_cases,
            batch_size=min(10, batch.num_cases),  # Sub-batches of 10
            worker_model=config.worker_model
        )
    else:
        # For small batches, use single worker
        generated_cases = await _generate_batch_with_worker(
            category=category_info,
            num_cases=batch.num_cases,
            worker_model=config.worker_model,
            batch_index=_builder_state.instances_created // config.batch_size
        )
    
    # Apply diversity checking if enabled
    accepted_cases = []
    rejected_count = 0
    
    for case in generated_cases:
        if config.enable_diversity_check:
            if _is_diverse_enough(case, _builder_state.recent_case_hashes):
                accepted_cases.append(case)
                # Update diversity tracking
                case_hash = _compute_case_hash(case)
                _builder_state.recent_case_hashes.append(case_hash)
                # Keep only recent hashes within the diversity window
                if len(_builder_state.recent_case_hashes) > config.diversity_window:
                    _builder_state.recent_case_hashes.pop(0)
            else:
                rejected_count += 1
                _builder_state.diversity_rejections += 1
        else:
            accepted_cases.append(case)
    
    # If we rejected too many, generate replacements
    if rejected_count > 0 and len(accepted_cases) < batch.num_cases:
        needed = batch.num_cases - len(accepted_cases)
        # Generate additional cases to compensate
        additional_cases = await _generate_batch_with_worker(
            category=category_info,
            num_cases=needed,
            worker_model=config.worker_model,
            batch_index=_builder_state.instances_created // config.batch_size + 1
        )
        for case in additional_cases:
            if len(accepted_cases) >= batch.num_cases:
                break
            if not config.enable_diversity_check or _is_diverse_enough(case, _builder_state.recent_case_hashes):
                accepted_cases.append(case)
                if config.enable_diversity_check:
                    case_hash = _compute_case_hash(case)
                    _builder_state.recent_case_hashes.append(case_hash)
                    if len(_builder_state.recent_case_hashes) > config.diversity_window:
                        _builder_state.recent_case_hashes.pop(0)
    
    # Add accepted cases to state
    _builder_state.created_instances.extend(accepted_cases)
    _builder_state.instances_created += len(accepted_cases)
    
    # Checkpointing if enabled
    if config.enable_checkpointing:
        if (_builder_state.instances_created - _builder_state.last_checkpoint_at) >= config.checkpoint_interval:
            checkpoint_file = config.checkpoint_dir / f"{_builder_state.dataset_name}_checkpoint.json"
            _builder_state.checkpoint_file = checkpoint_file
            
            checkpoint_data = {
                'dataset_name': _builder_state.dataset_name,
                'instances_created': _builder_state.instances_created,
                'total_instances_planned': _builder_state.total_instances_planned,
                'created_instances': _builder_state.created_instances,
                'dataset_plan': _builder_state.dataset_plan,
                'timestamp': datetime.now().isoformat()
            }
            
            _save_checkpoint(checkpoint_file, checkpoint_data)
            _builder_state.last_checkpoint_at = _builder_state.instances_created
    
    progress = (_builder_state.instances_created / _builder_state.total_instances_planned) * 100
    
    result_msg = (f"✓ Generated {len(accepted_cases)} test cases for category '{batch.category_name}'.\n"
                  f"  Progress: {_builder_state.instances_created}/{_builder_state.total_instances_planned} "
                  f"({progress:.1f}%)\n"
                  f"  Total categories in dataset: {len(set(c['inputs']['category'] for c in _builder_state.created_instances))}")
    
    if rejected_count > 0:
        result_msg += f"\n  Diversity rejections in this batch: {rejected_count}"
    
    if config.enable_checkpointing and _builder_state.checkpoint_file:
        result_msg += f"\n  Checkpoint saved at {_builder_state.instances_created} cases"
    
    return result_msg


@function_tool
async def finalize_and_store_dataset(
    ctx: RunContextWrapper[Any],
    experiment_name: str = "evaluation-datasets"
) -> str:
    """
    Finalize the dataset and store it in MLflow.
    
    This function takes all generated test cases and creates an MLflow dataset
    that can be used for evaluation.
    
    Args:
        experiment_name: MLflow experiment name to associate the dataset with 
            (default: "evaluation-datasets"). The experiment will be created if it doesn't exist.
    
    Returns:
        Confirmation message with dataset details.
    """
    global _builder_state
    
    # Validate all instances created
    if _builder_state.instances_created < _builder_state.total_instances_planned:
        remaining = _builder_state.total_instances_planned - _builder_state.instances_created
        return (f"✗ Error: Dataset incomplete. Still need {remaining} more test cases. "
                f"Progress: {_builder_state.instances_created}/{_builder_state.total_instances_planned}")
    
    if not _builder_state.created_instances:
        return "✗ Error: No test cases have been generated yet."
    
    # Debug: Log the structure of the first test case to verify it's correct
    import json
    first_case = _builder_state.created_instances[0]
    print("\n" + "="*60)
    print("DEBUG: First test case structure:")
    print(json.dumps(first_case, indent=2))
    print("="*60 + "\n")
    
    # Verify all test cases have correct structure (must have "inputs" and "expectations")
    # MLflow may add additional fields like "source" and "tags", which we allow
    for i, case in enumerate(_builder_state.created_instances):
        case_keys = set(case.keys())
        required_keys = {"inputs", "expectations"}
        if not required_keys.issubset(case_keys):
            missing_keys = required_keys - case_keys
            return (f"✗ Error: Test case {i+1} is missing required fields.\n"
                    f"  Required keys: {list(required_keys)}\n"
                    f"  Missing keys: {list(missing_keys)}\n"
                    f"  Actual keys: {list(case_keys)}\n"
                    f"  This case needs to be regenerated.")
        
        # Additional validation: ensure expectations has the right structure
        expectations = case.get("expectations", {})
        if not isinstance(expectations, dict):
            return (f"✗ Error: Test case {i+1} has invalid expectations structure.\n"
                    f"  Expectations must be a dictionary, got {type(expectations).__name__}")

    
    try:
        # Get or create the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✓ Created new experiment: '{experiment_name}' (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✓ Using existing experiment: '{experiment_name}' (ID: {experiment_id})")
        
        # Create the MLflow dataset
        dataset = create_dataset(
            name=_builder_state.dataset_name,
            experiment_id=[experiment_id],
            tags={
                "target_agent": _builder_state.target_agent_name or "unknown",
                "target_agent_run_id": _builder_state.target_agent_run_id or "unknown",
                "created_by": "dataset_builder_agent",
                "created_at": datetime.now().isoformat(),
                "total_instances": str(_builder_state.instances_created),
                "categories": str(len(_builder_state.dataset_plan.get('categories', []))),
            }
        )
        
        # Add test cases to the dataset in batches to avoid overwhelming the server
        batch_size = 5  # Smaller batch size for more reliable uploads
        total_records = len(_builder_state.created_instances)
        records_added = 0
        
        for i in range(0, total_records, batch_size):
            batch = _builder_state.created_instances[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_records + batch_size - 1) // batch_size
            
            try:
                # Add this batch of records
                dataset.merge_records(batch)
                records_added += len(batch)
                
                # Small delay between batches to avoid overwhelming the server
                if i + batch_size < total_records:
                    await asyncio.sleep(0.5)
                    
            except Exception as batch_error:
                return (f"✗ Error adding records batch {batch_num}/{total_batches} to MLflow: {str(batch_error)}\n"
                        f"  Successfully added {records_added}/{total_records} records before error.\n"
                        f"  You may need to retry or check MLflow server logs.")
        
        _builder_state.dataset_created = True
        _builder_state.dataset_experiment_id = experiment_id
        
        # Generate category breakdown
        category_counts = {}
        for case in _builder_state.created_instances:
            cat = case["inputs"]["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        breakdown = "\n".join([f"  - {cat}: {count} cases" for cat, count in category_counts.items()])
        
        return (f"✓ Successfully created and stored dataset in MLflow!\n\n"
                f"Dataset Details:\n"
                f"  Name: {_builder_state.dataset_name}\n"
                f"  Experiment: {experiment_name}\n"
                f"  Experiment ID: {experiment_id}\n"
                f"  Total Test Cases: {_builder_state.instances_created}\n"
                f"  Target Agent: {_builder_state.target_agent_name}\n\n"
                f"Category Breakdown:\n{breakdown}\n\n"
                f"The dataset is now ready for agent evaluation!")
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return (f"✗ Error storing dataset in MLflow: {str(e)}\n\n"
                f"Troubleshooting:\n"
                f"1. Check that MLflow server is running (http://localhost:5000)\n"
                f"2. Verify network connectivity to the MLflow server\n"
                f"3. Check MLflow server logs for details\n"
                f"4. Try reducing the dataset size if the issue persists\n\n"
                f"Error details:\n{error_details[:500]}")


@function_tool
async def get_builder_state_summary(ctx: RunContextWrapper[Any]) -> str:
    """
    Get a summary of the current dataset builder state.
    
    This helps track progress and ensures the agent stays on track.
    
    Returns:
        Summary of the current state.
    """
    global _builder_state
    return _builder_state.to_summary()


@function_tool
async def reset_dataset_creation(ctx: RunContextWrapper[Any]) -> str:
    """
    Reset the dataset creation state while keeping the plan.
    
    Use this if you need to restart dataset generation from scratch.
    
    Returns:
        Confirmation message.
    """
    global _builder_state
    _builder_state.reset_dataset_creation()
    return "✓ Dataset creation state reset. The plan is preserved, but all generated test cases have been cleared."


@function_tool
async def reset_builder_state(ctx: RunContextWrapper[Any]) -> str:
    """
    Completely reset the dataset builder state (clears everything including plan and target agent).
    
    Use this to start fresh with a new dataset or target agent.
    
    Returns:
        Confirmation message.
    """
    global _builder_state
    _builder_state = DatasetBuilderState()
    return "✓ Builder state completely reset. You can start fresh with a new target agent and dataset."


@function_tool
async def load_from_checkpoint(
    ctx: RunContextWrapper[Any],
    checkpoint_path: str | None = None
) -> str:
    """
    Load dataset creation progress from a checkpoint file.
    
    This allows resuming dataset creation if it was interrupted.
    
    Args:
        checkpoint_path: Optional path to the checkpoint file. If not provided,
            will look for the most recent checkpoint for the current dataset.
    
    Returns:
        Summary of loaded state.
    """
    global _builder_state
    
    config = DatasetBuilderConfig()
    
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
    else:
        # Look for checkpoint based on current dataset name
        if not _builder_state.dataset_name:
            return "✗ Error: No dataset name set. Cannot locate checkpoint."
        
        checkpoint_file = config.checkpoint_dir / f"{_builder_state.dataset_name}_checkpoint.json"
    
    # Load checkpoint
    checkpoint_data = _load_checkpoint(checkpoint_file)
    
    if not checkpoint_data:
        return f"✗ Error: No checkpoint found at {checkpoint_file}"
    
    # Restore state from checkpoint
    _builder_state.dataset_name = checkpoint_data.get('dataset_name')
    _builder_state.instances_created = checkpoint_data.get('instances_created', 0)
    _builder_state.total_instances_planned = checkpoint_data.get('total_instances_planned', 0)
    
    # Load created instances and migrate old structure if needed
    loaded_instances = checkpoint_data.get('created_instances', [])
    _builder_state.created_instances = []
    
    # Migrate old test case structure (remove "metadata" field, merge into "inputs")
    migrated_count = 0
    for case in loaded_instances:
        migrated_case = _migrate_test_case_structure(case)
        if migrated_case != case:
            migrated_count += 1
        _builder_state.created_instances.append(migrated_case)
    
    _builder_state.dataset_plan = checkpoint_data.get('dataset_plan')
    _builder_state.checkpoint_file = checkpoint_file
    _builder_state.last_checkpoint_at = _builder_state.instances_created
    _builder_state.categories_approved = True  # Must have been approved to create instances
    
    # Rebuild diversity hashes from last N cases
    if config.enable_diversity_check and _builder_state.created_instances:
        recent_cases = _builder_state.created_instances[-config.diversity_window:]
        _builder_state.recent_case_hashes = [_compute_case_hash(case) for case in recent_cases]
    
    progress = (_builder_state.instances_created / _builder_state.total_instances_planned * 100) if _builder_state.total_instances_planned > 0 else 0
    
    migration_msg = f"\n✓ Migrated {migrated_count} test cases to new structure" if migrated_count > 0 else ""
    
    return (f"✓ Checkpoint loaded successfully!{migration_msg}\n\n"
            f"Dataset: {_builder_state.dataset_name}\n"
            f"Progress: {_builder_state.instances_created}/{_builder_state.total_instances_planned} ({progress:.1f}%)\n"
            f"Checkpoint timestamp: {checkpoint_data.get('timestamp', 'unknown')}\n\n"
            f"You can continue generating test cases from where you left off.")


# ==================== AGENT CLASS ====================

class DatasetBuilderAgent:
    """
    An agent that helps users create high-quality evaluation datasets for target agents.
    
    The agent:
    1. Logs the target agent in MLflow for tracking
    2. Chats with users to understand dataset requirements
    3. Creates a structured plan with categories and distributions
    4. Generates test cases in batches (max 20 at a time)
    5. Tracks progress to ensure exact variance as planned
    6. Stores the final dataset in MLflow for evaluation
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the dataset builder agent.
        
        Args:
            model: The OpenAI model to use (default: gpt-4o).
        """
        config = DatasetBuilderConfig()
        
        self.agent = Agent(
            name="DatasetBuilderAgent",
            model=model,
            instructions=f"""You are an expert dataset builder agent specialized in creating high-quality 
evaluation datasets for LLM-based agents.

Your workflow:

1. **Choose Target Agent**: The user can either:
   a) **Use a previously logged agent** (faster, recommended if available):
      - Use list_previously_logged_agents to show recent logged agents
      - Use use_previously_logged_agent with the run_id to load that agent
   b) **Log a new agent** (if not logged before):
      - Use log_target_agent_in_mlflow with:
        * Agent file path (e.g., 'src/dev_agents/customer_service_agent.py')
        * Agent class name (e.g., 'CustomerServiceAgent')
        * Agent description (what the agent does)

2. **Analyze Agent & Present Understanding**: IMMEDIATELY after logging the agent, use 
   analyze_target_agent_metadata to read the agent's code and extract:
   - Agent instructions/system prompt
   - Available tools and their purposes
   - Model configuration
   - Suggested test categories based on tools
   
   Present this analysis to the user with your understanding of:
   - What the agent does (based on instructions and tools)
   - Suggested test categories (derived from tools)
   - Key capabilities to validate
   - Potential edge cases to test
   
   Ask the user to review and tweak as needed:
   - "Are these the right categories?"
   - "Should I add/remove/modify any categories?"
   - "How many test cases per category?"
   - "What specific outputs/behaviors should I validate?"
   - "Which tools should the agent use for each category?" (for tool usage evaluation)
   
3. **Refine Requirements Through Discussion**: Based on the user's feedback, have a 
   conversation to refine:
   - Test categories and their descriptions
   - Real-world scenarios and edge cases
   - Expected behaviors and success criteria (answer/response)
   - Expected tool usage patterns (which tools should be called)
   - Distribution of test cases across categories
   
4. **Create Dataset Plan**: Use create_dataset_plan to formalize the plan. Include:
   - Dataset name
   - Categories with descriptions
   - Distribution of test cases
   - Example inputs for each category
   - Expected outputs/behaviors (in "expectations" field)
   - Expected tool calls (in "expected_tools" list) - this enables MLflow's trace-based evaluation
   
   Note: The expectations can be a JSON string like:
   {{"answer": "expected response", "tool_calls": ["tool1", "tool2"]}}
   
   Or you can use the expected_tools field separately, which will be merged into expectations.
   
4. **Get Approval**: Present the plan clearly and get user approval using approve_dataset_plan.

5. **Generate Test Cases in Batches**: Once approved, systematically generate test cases:
   - Use generate_test_cases_batch for each category
   - Create at most {config.batch_size} cases per batch
   - The system uses worker agents for parallel generation (scalable!)
   - Diversity checking ensures unique test cases
   - Automatic checkpointing every {config.checkpoint_interval} cases
   - Track what you've created to avoid duplicates
   - Ensure the exact distribution as planned
   - Break large categories into multiple batches
   
6. **Resume from Checkpoints**: If generation is interrupted:
   - Use load_from_checkpoint to resume from where you left off
   - Checkpoints are saved automatically in {config.checkpoint_dir}
   
7. **Finalize Dataset**: Once all test cases are generated, use finalize_and_store_dataset
   to store everything in MLflow. By default, datasets are stored in the 'evaluation-datasets' 
   experiment for better organization. You can specify a different experiment name if needed.

**Advanced Features**:
- **Parallel Generation**: Uses worker agents to generate batches in parallel (enabled: {config.enable_parallel_generation})
- **Diversity Checking**: Prevents duplicate patterns (enabled: {config.enable_diversity_check})
- **Checkpointing**: Auto-saves every {config.checkpoint_interval} cases (enabled: {config.enable_checkpointing})
- **Worker Model**: Uses {config.worker_model} for cost-efficient batch generation

Important guidelines:
- Always check the builder state to track progress
- Generate diverse, realistic test cases
- Maintain the exact category distribution as planned
- Never exceed {config.max_dataset_size} total instances
- Be conversational and collaborative with the user
- Ask clarifying questions to understand their needs
- Explain what you're doing at each step

Remember: Quality over quantity! Help users create meaningful datasets that truly test their agents.
""",
            tools=[
                log_target_agent_in_mlflow,
                use_previously_logged_agent,
                list_previously_logged_agents,
                analyze_target_agent_metadata,
                create_dataset_plan,
                approve_dataset_plan,
                generate_test_cases_batch,
                finalize_and_store_dataset,
                get_builder_state_summary,
                reset_dataset_creation,
                reset_builder_state,
                load_from_checkpoint,
            ],
        )
    
    def get_agent(self) -> Agent:
        """Get the underlying Agent instance."""
        return self.agent
    
    def list_tools(self) -> None:
        """Print information about available tools."""
        print("Dataset Builder Agent Tools:")
        print("=" * 70)
        for tool in self.agent.tools:
            if hasattr(tool, '__name__'):
                print(f"- {tool.__name__}")


# ==================== INTERACTIVE CHAT ====================

async def interactive_chat():
    """Run an interactive chat session with the dataset builder agent."""
    print("\n" + "=" * 70)
    print("Dataset Builder Agent - Interactive Session")
    print("=" * 70)
    print("\nThis agent will help you create evaluation datasets for your agents.")
    print("It will log your target agent, discuss requirements, and create datasets.")
    print("\nType 'quit', 'exit', or 'bye' to end the conversation.")
    print("=" * 70 + "\n")
    
    # Ask user if they want to use a previously logged agent or log a new one
    print("How would you like to specify your target agent?\n")
    print("1. Use a previously logged agent from MLflow")
    print("2. Log a new agent from source file\n")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    initial_context = ""
    
    if choice == "1":
        # User wants to use a previously logged agent
        print("\n" + "=" * 70)
        print("Previously Logged Agents")
        print("=" * 70 + "\n")
        
        # Get list of logged agents from MLflow
        try:
            experiment = mlflow.get_experiment_by_name("dataset-builder-targets")
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=10
                )
                
                if not runs.empty:
                    print("Recent logged agents:\n")
                    for idx, row in runs.iterrows():
                        agent_class = row.get('params.agent_class', 'Unknown')
                        agent_file = row.get('params.agent_file', 'Unknown')
                        run_id = row['run_id']
                        start_time = row['start_time'].strftime('%Y-%m-%d %H:%M:%S') if 'start_time' in row else 'Unknown'
                        
                        print(f"{idx + 1}. {agent_class}")
                        print(f"   File: {agent_file}")
                        print(f"   Run ID: {run_id}")
                        print(f"   Logged: {start_time}")
                        print()
                    
                    selection = input("Enter the number of the agent to use (or 'new' to log a new one): ").strip()
                    
                    if selection.isdigit() and 1 <= int(selection) <= len(runs):
                        selected_run = runs.iloc[int(selection) - 1]
                        agent_class = selected_run.get('params.agent_class', 'Unknown')
                        agent_file = selected_run.get('params.agent_file', 'Unknown')
                        run_id = selected_run['run_id']
                        description = selected_run.get('tags.description', 'A target agent for evaluation')
                        
                        # Set the state directly without logging again
                        _builder_state.target_agent_name = agent_class
                        _builder_state.target_agent_description = description
                        _builder_state.target_agent_logged = True
                        _builder_state.target_agent_run_id = run_id
                        
                        initial_context = (f"I'm using the previously logged agent: {agent_class} "
                                         f"(file: {agent_file}, run_id: {run_id}). "
                                         f"Please analyze this agent and help me create a dataset for it.")
                        
                        print(f"\n✓ Selected: {agent_class}")
                        print(f"  Run ID: {run_id}")
                        print("  Moving to dataset creation...\n")
                    else:
                        choice = "2"  # Fall through to logging new agent
                else:
                    print("No previously logged agents found.")
                    print("Let's log a new agent instead.\n")
                    choice = "2"
            else:
                print("No 'dataset-builder-targets' experiment found.")
                print("Let's log a new agent instead.\n")
                choice = "2"
        except Exception as e:
            print(f"Error retrieving logged agents: {e}")
            print("Let's log a new agent instead.\n")
            choice = "2"
    
    if choice == "2" or not initial_context:
        # User wants to log a new agent
        initial_context = ("I want to create a dataset for my agent. "
                          "Let me provide the agent file path, class name, and description so you can log it.")
    
    # Create the agent
    builder = DatasetBuilderAgent()
    agent = builder.get_agent()
    
    print("\n" + "=" * 70)
    print("Starting Conversation")
    print("=" * 70 + "\n")
    
    # Initialize conversation result with the initial context
    conversation_result = await Runner.run(agent, initial_context)
    print(f"Agent: {conversation_result.final_output}\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nDataset Builder Agent: Great working with you! Your dataset is ready for evaluation. Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Manual conversation management - use to_input_list() to maintain conversation history
            agent_input = conversation_result.to_input_list() + [
                {"role": "user", "content": user_input}
            ]
            
            # Run the agent
            conversation_result = await Runner.run(agent, agent_input)
            
            # Display the response
            print(f"\nAgent: {conversation_result.final_output}\n")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            continue


# ==================== MAIN ====================

async def main():
    """Main entry point for the dataset builder agent."""
    print("\n" + "=" * 70)
    print("Dataset Builder Agent")
    print("Create high-quality evaluation datasets for your agents")
    print("=" * 70)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        return
    
    # Show configuration
    config = DatasetBuilderConfig()
    print("\nConfiguration:")
    print(f"  Max Dataset Size: {config.max_dataset_size}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Model: {config.model}")
    print(f"  Worker Model: {config.worker_model}")
    print("\nScalability Features:")
    print(f"  Parallel Generation: {'✓ Enabled' if config.enable_parallel_generation else '✗ Disabled'}")
    print(f"  Checkpointing: {'✓ Enabled' if config.enable_checkpointing else '✗ Disabled'} (every {config.checkpoint_interval} cases)")
    print(f"  Diversity Checking: {'✓ Enabled' if config.enable_diversity_check else '✗ Disabled'} (window: {config.diversity_window})")
    print(f"  Checkpoint Directory: {config.checkpoint_dir}")
    print(f"\n  MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Run interactive chat
    await interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())
