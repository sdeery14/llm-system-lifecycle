"""
Agent Analysis Script using MLflow Evaluation Framework.

This script evaluates agents using datasets created by the Dataset Builder,
implements custom scorers for tool usage validation, and generates comprehensive
analysis reports that are logged as MLflow artifacts.
"""

from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Any, Callable
from datetime import datetime
from collections import defaultdict

import mlflow
from mlflow.entities import Trace, SpanType, Feedback
from mlflow.genai import scorer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ==================== CUSTOM SCORERS ====================

@scorer
def exact_match(outputs: str, expectations: dict) -> bool:
    """
    Check if the output exactly matches the expected answer.
    
    Args:
        outputs: The agent's output string.
        expectations: Dictionary containing the expected answer.
    
    Returns:
        True if outputs match expectations["answer"], False otherwise.
    """
    expected_answer = expectations.get("answer", "")
    
    # Convert both to strings and compare (case-insensitive)
    output_str = str(outputs).strip().lower()
    expected_str = str(expected_answer).strip().lower()
    
    return output_str == expected_str


@scorer
def contains_expected_content(outputs: str, expectations: dict) -> bool:
    """
    Check if the output contains the expected content (more lenient than exact match).
    
    Args:
        outputs: The agent's output string.
        expectations: Dictionary containing the expected answer.
    
    Returns:
        True if outputs contain key elements from expectations["answer"].
    """
    expected_answer = expectations.get("answer", "")
    
    if not expected_answer:
        return True
    
    # Convert to lowercase for comparison
    output_str = str(outputs).strip().lower()
    expected_str = str(expected_answer).strip().lower()
    
    # Check if any significant words from expected answer are in output
    expected_words = set(expected_str.split())
    output_words = set(output_str.split())
    
    # At least 50% of expected words should be in output
    if len(expected_words) == 0:
        return True
    
    overlap = len(expected_words.intersection(output_words))
    overlap_ratio = overlap / len(expected_words)
    
    return overlap_ratio >= 0.5


@scorer
def uses_correct_tools(trace: Trace, expectations: dict) -> Feedback:
    """
    Evaluate if the agent used the correct tools.
    
    This scorer uses MLflow's trace functionality to extract the tools that were
    actually called and compares them to the expected tool calls.
    
    Args:
        trace: The MLflow trace containing span information.
        expectations: Dictionary containing expected tool_calls list.
    
    Returns:
        Feedback object with score and rationale.
    """
    expected_tools = expectations.get("tool_calls", [])
    
    # If no tools are expected, check that no tools were called
    if not expected_tools:
        tool_spans = trace.search_spans(span_type=SpanType.TOOL)
        if not tool_spans:
            return Feedback(
                value="yes",
                rationale="Correctly did not call any tools when none were expected."
            )
        else:
            actual_tools = [span.name for span in tool_spans]
            return Feedback(
                value="no",
                rationale=f"Called tools {actual_tools} when no tools were expected."
            )
    
    # Parse the trace to get the actual tool calls
    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    actual_tools = [span.name for span in tool_spans]
    
    # Check if the tools match (order-independent)
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    
    if expected_set == actual_set:
        score = "yes"
        rationale = f"The agent used the correct tools: {actual_tools}"
    else:
        score = "no"
        missing_tools = expected_set - actual_set
        extra_tools = actual_set - expected_set
        
        details = []
        if missing_tools:
            details.append(f"Missing tools: {list(missing_tools)}")
        if extra_tools:
            details.append(f"Extra tools: {list(extra_tools)}")
        
        rationale = (
            f"Expected tools: {expected_tools}\n"
            f"Actual tools: {actual_tools}\n"
            f"{'; '.join(details)}"
        )
    
    return Feedback(value=score, rationale=rationale)


@scorer
def tool_call_efficiency(trace: Trace, expectations: dict) -> Feedback:
    """
    Evaluate the efficiency of tool usage (number of calls, order, etc.).
    
    Args:
        trace: The MLflow trace containing span information.
        expectations: Dictionary containing expected tool_calls list.
    
    Returns:
        Feedback object with efficiency score.
    """
    expected_tools = expectations.get("tool_calls", [])
    
    if not expected_tools:
        return Feedback(value="n/a", rationale="No tools expected for this test case.")
    
    # Get actual tool calls
    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    actual_tools = [span.name for span in tool_spans]
    
    expected_count = len(expected_tools)
    actual_count = len(actual_tools)
    
    # Calculate efficiency score
    if actual_count == expected_count:
        score = "optimal"
        rationale = f"Used exactly {actual_count} tool calls as expected."
    elif actual_count < expected_count:
        score = "under"
        rationale = f"Used fewer tools ({actual_count}) than expected ({expected_count})."
    else:
        extra_calls = actual_count - expected_count
        score = "over"
        rationale = f"Used {extra_calls} more tool calls than expected ({actual_count} vs {expected_count})."
    
    return Feedback(value=score, rationale=rationale)


@scorer
def response_quality(outputs: str, expectations: dict) -> Feedback:
    """
    Evaluate the quality of the response using direct OpenAI API call.
    
    This scorer evaluates response quality based on completeness, accuracy, 
    clarity, conciseness, and relevance.
    
    Args:
        outputs: The agent's output string.
        expectations: Dictionary containing the expected answer.
    
    Returns:
        Feedback object with quality assessment including a numerical score (0-100).
    """
    expected_answer = expectations.get("answer", "")
    
    # Quick check for empty responses
    if not outputs or len(str(outputs).strip()) == 0:
        return Feedback(
            value=0.0,  # Use numerical score
            rationale="Empty or missing response."
        )
    
    output_str = str(outputs).strip()
    
    # Prepare the evaluation prompt
    evaluation_prompt = f"""You are an expert evaluator of AI agent responses.

Evaluate the quality of the following agent response based on the expected answer.

**Expected Answer:** 
{expected_answer}

**Actual Response:**
{output_str}

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

Respond with ONLY a JSON object in this exact format:
{{
    "score": "excellent|good|fair|poor|empty",
    "quality_score": <number between 0-100>,
    "rationale": "<brief 1-2 sentence explanation>"
}}"""
    
    try:
        # Use direct OpenAI API call instead of Agent to avoid async issues
        from openai import OpenAI
        import json
        import os
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a response quality evaluator. Always respond with valid JSON."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result_text = response.choices[0].message.content
        evaluation = json.loads(result_text)
        
        # Validate and extract fields
        quality_score = float(evaluation.get("quality_score", 50.0))
        score_label = evaluation.get("score", "fair")
        rationale = evaluation.get("rationale", "Evaluated using OpenAI")
        
        # Return Feedback with numerical score for granular comparison
        return Feedback(
            value=quality_score,
            rationale=f"[{score_label}] {rationale}"
        )
    
    except Exception as e:
        # Fallback to simple heuristic if evaluation fails
        if len(output_str) < 10:
            quality_score = 20.0
            rationale = f"Response too short. Agent evaluation failed: {str(e)}"
        elif len(output_str) > 1000:
            quality_score = 50.0
            rationale = f"Response may be verbose. Agent evaluation failed: {str(e)}"
        else:
            quality_score = 50.0
            rationale = f"Could not evaluate: {str(e)}"
        
        return Feedback(value=quality_score, rationale=rationale)


# ==================== EVALUATION FUNCTIONS ====================

def load_agent(agent_run_id: str) -> Any:
    """
    Load an agent from MLflow using its run ID.
    
    Args:
        agent_run_id: The MLflow run ID where the agent was logged.
    
    Returns:
        The loaded agent model.
    """
    run = mlflow.get_run(agent_run_id)
    model_uri = f"runs:/{agent_run_id}/agent"
    
    print(f"Loading agent from: {model_uri}")
    agent_model = mlflow.pyfunc.load_model(model_uri)
    
    return agent_model


def load_dataset(dataset_name: str, experiment_name: str = "evaluation-datasets") -> list[dict]:
    """
    Load a dataset from MLflow that was created by the Dataset Builder.
    
    The Dataset Builder creates MLflow datasets using create_dataset() and stores
    test cases with merge_records(). This function retrieves those records.
    
    Args:
        dataset_name: Name of the dataset to load.
        experiment_name: Name of the experiment containing the dataset.
    
    Returns:
        List of test case dictionaries with 'inputs' and 'expectations' keys.
    """
    print(f"Loading dataset '{dataset_name}' from experiment '{experiment_name}'...")
    
    # Get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    try:
        # Use MLflow's genai datasets API to search and load
        from mlflow.genai.datasets import search_datasets
        
        # Search for dataset by name in the experiment
        datasets = search_datasets(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"name = '{dataset_name}'"
        )
        
        if not datasets or len(datasets) == 0:
            raise ValueError(f"Dataset '{dataset_name}' not found in experiment '{experiment_name}'.")
        
        # Get the first matching dataset (most recent if multiple with same name)
        dataset = datasets[0]
        print(f"✓ Found dataset: {dataset.name} (ID: {dataset.dataset_id})")
        
        # Access the records from the dataset
        records = dataset.records
        
        # Convert DatasetRecord objects to dicts
        records_list = []
        for record in records:
            # DatasetRecord has inputs and expectations attributes
            if hasattr(record, 'inputs') and hasattr(record, 'expectations'):
                records_list.append({
                    'inputs': record.inputs,
                    'expectations': record.expectations
                })
            elif isinstance(record, dict):
                records_list.append(record)
            else:
                # Try to convert to dict
                if hasattr(record, 'to_dict'):
                    records_list.append(record.to_dict())
                elif hasattr(record, '__dict__'):
                    records_list.append(record.__dict__)
        
        # Validate structure
        if records_list and isinstance(records_list[0], dict):
            if 'inputs' in records_list[0] and 'expectations' in records_list[0]:
                print(f"✓ Loaded {len(records_list)} test cases")
                return records_list
        
        print("⚠️  Dataset loaded but format may need validation")
        return records_list
        
    except ImportError:
        print("MLflow genai.datasets module not available, using alternative method...")
    
    # Method 2: Search for the dataset in MLflow runs
    print("Searching for dataset in experiment runs...")
    
    # The Dataset Builder tags datasets with 'created_by': 'dataset_builder_agent'
    # and stores the dataset name in tags
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=100
    )
    
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'.")
    
    # Look for a run that created this dataset
    matching_runs = []
    
    for idx, row in runs.iterrows():
        # Check if this run created our dataset
        run_tags = {}
        for col in runs.columns:
            if col.startswith('tags.'):
                tag_name = col.replace('tags.', '')
                run_tags[tag_name] = row[col]
        
        # Check if dataset name matches
        if run_tags.get('created_by') == 'dataset_builder_agent':
            # The dataset name might be in tags or we can infer from run
            # For now, try to load the dataset directly
            run_id = row['run_id']
            matching_runs.append(run_id)
    
    if not matching_runs:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in experiment '{experiment_name}'.\n"
            f"Make sure you created the dataset using the Dataset Builder."
        )
    
    # Try to get the dataset from MLflow's datasets registry
    # The Dataset Builder uses create_dataset which registers the dataset
    print(f"Found {len(matching_runs)} potential dataset runs")
    
    # Method 3: Direct dataset loading by name
    try:
        # MLflow 3.x dataset loading
        from mlflow.data import load_dataset as load_mlflow_dataset
        
        dataset = load_mlflow_dataset(name=dataset_name)
        
        if hasattr(dataset, 'load'):
            data = dataset.load()
        else:
            data = dataset
        
        # Convert to list of dicts
        if hasattr(data, 'to_pandas'):
            df = data.to_pandas()
            records = df.to_dict('records')
        elif hasattr(data, 'to_dict'):
            records = data.to_dict('records')
        elif isinstance(data, list):
            records = data
        else:
            records = list(data)
        
        print(f"✓ Loaded {len(records)} test cases from dataset")
        return records
        
    except Exception as e:
        print(f"Could not load dataset: {e}")
        
        # Method 4: Manual extraction from saved state
        # The Dataset Builder saves created_instances in its state
        # We can try to reconstruct from the checkpoints
        print("\n⚠️  Automatic dataset loading failed.")
        print("Please ensure the dataset was created and finalized in MLflow.")
        print(f"Dataset name: {dataset_name}")
        print(f"Experiment: {experiment_name}")
        print("\nYou can manually export the dataset from MLflow UI or")
        print("use the Dataset Builder to recreate it.")
        
        raise ValueError(
            f"Could not load dataset '{dataset_name}' from MLflow. "
            f"Tried multiple loading methods. Please check MLflow UI."
        )


def create_predict_function(agent_model: Any) -> Callable:
    """
    Create a prediction function that MLflow can call.
    
    Args:
        agent_model: The loaded agent model.
    
    Returns:
        A function that takes inputs and returns predictions.
    """
    def predict_fn(**kwargs) -> str:
        """
        Predict function for MLflow evaluation.
        
        MLflow will call this with the keys from the dataset's inputs as kwargs.
        For example, if inputs = {"query": "...", "category": "..."}, 
        this will be called as predict_fn(query="...", category="...").
        
        Args:
            **kwargs: Input data (typically includes 'query', 'category', 'context', etc.)
        
        Returns:
            The agent's response as a string.
        """
        # Extract the query from kwargs
        query = kwargs.get("query", kwargs.get("task", ""))
        
        if not query:
            return "Error: No query provided in inputs."
        
        try:
            # ResponsesAgent expects input as a list of Message objects
            # Format: [{"role": "user", "content": "query text"}]
            model_input = {
                "input": [{"role": "user", "content": query}]
            }
            
            # Call the agent model with the properly formatted input
            result = agent_model.predict(model_input)
            
            # Extract the final output
            if isinstance(result, dict):
                # ResponsesAgent returns a dict with various fields
                # Try to get the text response from common fields
                output = result.get("output", result.get("text", result.get("content", str(result))))
            elif isinstance(result, list) and len(result) > 0:
                # If result is a list, get the first item
                output = str(result[0])
            else:
                output = str(result)
            
            return output
        
        except Exception as e:
            return f"Error during prediction: {str(e)}"
    
    return predict_fn


def run_evaluation(
    dataset: list[dict],
    predict_fn: Callable,
    scorers: list[Callable] | None = None,
    experiment_name: str = "agent-evaluation"
) -> Any:
    """
    Run MLflow evaluation on the dataset.
    
    Args:
        dataset: List of test cases.
        predict_fn: Function that makes predictions.
        scorers: List of scorer functions (uses default if None).
        experiment_name: Name of the evaluation experiment.
    
    Returns:
        MLflow evaluation results.
    """
    if scorers is None:
        scorers = [
            exact_match,
            contains_expected_content,
            uses_correct_tools,
            tool_call_efficiency,
            response_quality
        ]
    
    # Set the experiment
    mlflow.set_experiment(experiment_name)
    
    print(f"\n{'='*70}")
    print(f"Running Evaluation")
    print(f"{'='*70}")
    print(f"Dataset size: {len(dataset)} test cases")
    print(f"Scorers: {[s.__name__ if hasattr(s, '__name__') else str(s) for s in scorers]}")
    print(f"{'='*70}\n")
    
    # Run the evaluation
    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_fn,
        scorers=scorers
    )
    
    return results


# ==================== ANALYSIS REPORT GENERATION ====================

def analyze_evaluation_results(results: Any, dataset: list[dict]) -> dict[str, Any]:
    """
    Analyze evaluation results and compute statistics.
    
    Args:
        results: MLflow evaluation results.
        dataset: The dataset that was evaluated.
    
    Returns:
        Dictionary containing analysis statistics.
    """
    analysis = {
        "summary": {},
        "scorer_stats": {},
        "category_breakdown": {},
        "tool_analysis": {},
        "failures": [],
        "recommendations": []
    }
    
    # Get the results dataframe - MLflow 3.x uses .result_df
    if hasattr(results, 'result_df'):
        results_df = results.result_df
    elif hasattr(results, 'tables') and 'eval_results_table' in results.tables:
        results_df = results.tables["eval_results_table"]
    elif hasattr(results, 'eval_results_table'):
        results_df = results.eval_results_table
    else:
        # Fallback: try to convert results to DataFrame
        import pandas as pd
        if isinstance(results, pd.DataFrame):
            results_df = results
        else:
            print(f"Warning: Could not extract results table. Results object type: {type(results)}")
            print(f"Available attributes: {dir(results)}")
            # Return minimal analysis
            analysis["summary"]["total_test_cases"] = len(dataset)
            return analysis
    
    # Basic summary statistics
    total_cases = len(results_df)
    analysis["summary"]["total_test_cases"] = total_cases
    
    # Analyze each scorer
    scorer_columns = [col for col in results_df.columns if col.endswith("/score") or col.endswith("/value")]
    
    for scorer_col in scorer_columns:
        scorer_name = scorer_col.replace("/score", "").replace("/value", "")
        
        if scorer_col in results_df.columns:
            values = results_df[scorer_col].dropna()
            
            # Different analysis based on value type
            if values.dtype == bool or set(values.unique()).issubset({True, False, "yes", "no"}):
                # Binary scorer
                if values.dtype == bool:
                    success_count = values.sum()
                else:
                    success_count = (values == "yes").sum() + (values == True).sum()
                
                pass_rate = (success_count / len(values) * 100) if len(values) > 0 else 0
                
                analysis["scorer_stats"][scorer_name] = {
                    "type": "binary",
                    "pass_count": int(success_count),
                    "fail_count": int(len(values) - success_count),
                    "pass_rate": round(pass_rate, 2),
                    "total": len(values)
                }
            else:
                # Categorical or numeric scorer
                value_counts = values.value_counts().to_dict()
                
                analysis["scorer_stats"][scorer_name] = {
                    "type": "categorical",
                    "value_distribution": value_counts,
                    "total": len(values)
                }
    
    # Category breakdown (if category info is in inputs)
    if "inputs" in results_df.columns:
        category_results = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
        
        for idx, row in results_df.iterrows():
            inputs = row.get("inputs", {})
            if isinstance(inputs, dict):
                category = inputs.get("category", "unknown")
            else:
                category = "unknown"
            
            category_results[category]["total"] += 1
            
            # Check if this case passed (based on uses_correct_tools scorer)
            if "uses_correct_tools/value" in row:
                if row["uses_correct_tools/value"] in ["yes", True]:
                    category_results[category]["passed"] += 1
                else:
                    category_results[category]["failed"] += 1
        
        # Convert to regular dict and calculate pass rates
        for category, stats in category_results.items():
            stats["pass_rate"] = round(
                (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0,
                2
            )
        
        analysis["category_breakdown"] = dict(category_results)
    
    # Tool usage analysis
    tool_usage = defaultdict(int)
    tool_accuracy = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    
    for idx, row in results_df.iterrows():
        expectations = row.get("expectations", {})
        if isinstance(expectations, dict):
            expected_tools = expectations.get("tool_calls", [])
            for tool in expected_tools:
                tool_usage[tool] += 1
                
                # Check if tool was used correctly
                if "uses_correct_tools/value" in row:
                    if row["uses_correct_tools/value"] in ["yes", True]:
                        tool_accuracy[tool]["correct"] += 1
                    else:
                        tool_accuracy[tool]["incorrect"] += 1
    
    # Calculate accuracy per tool
    tool_stats = {}
    for tool, count in tool_usage.items():
        accuracy = tool_accuracy[tool]
        accuracy_rate = (
            round((accuracy["correct"] / (accuracy["correct"] + accuracy["incorrect"]) * 100), 2)
            if (accuracy["correct"] + accuracy["incorrect"]) > 0
            else 0
        )
        
        tool_stats[tool] = {
            "total_expected": count,
            "correct_usage": accuracy["correct"],
            "incorrect_usage": accuracy["incorrect"],
            "accuracy_rate": accuracy_rate
        }
    
    analysis["tool_analysis"] = tool_stats
    
    # Identify failures
    failure_cases = []
    for idx, row in results_df.iterrows():
        # Check if any scorer failed
        failed_scorers = []
        
        for scorer_col in scorer_columns:
            if scorer_col in row:
                value = row[scorer_col]
                # Check for failure
                if value in [False, "no", "poor"]:
                    scorer_name = scorer_col.replace("/score", "").replace("/value", "")
                    failed_scorers.append(scorer_name)
        
        if failed_scorers:
            inputs = row.get("inputs", {})
            expectations = row.get("expectations", {})
            
            failure_cases.append({
                "test_case_id": idx,
                "category": inputs.get("category", "unknown") if isinstance(inputs, dict) else "unknown",
                "query": inputs.get("query", "N/A") if isinstance(inputs, dict) else "N/A",
                "failed_scorers": failed_scorers,
                "expected_tools": expectations.get("tool_calls", []) if isinstance(expectations, dict) else []
            })
    
    analysis["failures"] = failure_cases[:10]  # Top 10 failures
    
    # Generate recommendations
    recommendations = []
    
    # Check overall pass rate
    if "uses_correct_tools" in analysis["scorer_stats"]:
        tool_pass_rate = analysis["scorer_stats"]["uses_correct_tools"]["pass_rate"]
        if tool_pass_rate < 70:
            recommendations.append({
                "priority": "high",
                "area": "tool_usage",
                "issue": f"Low tool usage accuracy ({tool_pass_rate}%)",
                "suggestion": "Review agent's tool selection logic and system prompts."
            })
    
    # Check for problematic categories
    for category, stats in analysis["category_breakdown"].items():
        if stats["pass_rate"] < 60:
            recommendations.append({
                "priority": "medium",
                "area": "category_performance",
                "issue": f"Category '{category}' has low pass rate ({stats['pass_rate']}%)",
                "suggestion": f"Add more training examples or refine prompts for {category} scenarios."
            })
    
    # Check for problematic tools
    for tool, stats in analysis["tool_analysis"].items():
        if stats["accuracy_rate"] < 70:
            recommendations.append({
                "priority": "medium",
                "area": "tool_performance",
                "issue": f"Tool '{tool}' has low accuracy ({stats['accuracy_rate']}%)",
                "suggestion": f"Review the implementation and documentation of the '{tool}' tool."
            })
    
    analysis["recommendations"] = recommendations
    
    return analysis


def generate_analysis_report(
    analysis: dict[str, Any],
    agent_name: str,
    dataset_name: str,
    output_format: str = "markdown"
) -> str:
    """
    Generate a human-readable analysis report.
    
    Args:
        analysis: Analysis statistics dictionary.
        agent_name: Name of the agent being evaluated.
        dataset_name: Name of the dataset used.
        output_format: Format of the report ('markdown' or 'text').
    
    Returns:
        Formatted report string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if output_format == "markdown":
        report = f"""# Agent Evaluation Report

**Agent:** {agent_name}  
**Dataset:** {dataset_name}  
**Evaluation Date:** {timestamp}

---

## Executive Summary

- **Total Test Cases:** {analysis['summary']['total_test_cases']}
"""
        
        # Add overall pass rates
        if "uses_correct_tools" in analysis["scorer_stats"]:
            stats = analysis["scorer_stats"]["uses_correct_tools"]
            report += f"- **Tool Usage Pass Rate:** {stats['pass_rate']}% ({stats['pass_count']}/{stats['total']})\n"
        
        if "contains_expected_content" in analysis["scorer_stats"]:
            stats = analysis["scorer_stats"]["contains_expected_content"]
            report += f"- **Content Accuracy:** {stats['pass_rate']}% ({stats['pass_count']}/{stats['total']})\n"
        
        # Scorer Performance
        report += "\n## Scorer Performance\n\n"
        for scorer_name, stats in analysis["scorer_stats"].items():
            report += f"### {scorer_name}\n\n"
            
            if stats["type"] == "binary":
                report += f"- **Pass Rate:** {stats['pass_rate']}%\n"
                report += f"- **Passed:** {stats['pass_count']} / {stats['total']}\n"
                report += f"- **Failed:** {stats['fail_count']} / {stats['total']}\n\n"
            else:
                report += "- **Value Distribution:**\n"
                for value, count in stats["value_distribution"].items():
                    percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
                    report += f"  - `{value}`: {count} ({percentage:.1f}%)\n"
                report += "\n"
        
        # Category Breakdown
        if analysis["category_breakdown"]:
            report += "\n## Performance by Category\n\n"
            report += "| Category | Total | Passed | Failed | Pass Rate |\n"
            report += "|----------|-------|--------|--------|----------|\n"
            
            for category, stats in sorted(
                analysis["category_breakdown"].items(),
                key=lambda x: x[1]["pass_rate"]
            ):
                report += (
                    f"| {category} | {stats['total']} | {stats['passed']} | "
                    f"{stats['failed']} | {stats['pass_rate']}% |\n"
                )
        
        # Tool Analysis
        if analysis["tool_analysis"]:
            report += "\n## Tool Usage Analysis\n\n"
            report += "| Tool | Expected | Correct | Incorrect | Accuracy |\n"
            report += "|------|----------|---------|-----------|----------|\n"
            
            for tool, stats in sorted(
                analysis["tool_analysis"].items(),
                key=lambda x: x[1]["accuracy_rate"]
            ):
                report += (
                    f"| {tool} | {stats['total_expected']} | {stats['correct_usage']} | "
                    f"{stats['incorrect_usage']} | {stats['accuracy_rate']}% |\n"
                )
        
        # Failures
        if analysis["failures"]:
            report += f"\n## Failed Test Cases (Top {len(analysis['failures'])})\n\n"
            
            for i, failure in enumerate(analysis["failures"], 1):
                report += f"### {i}. Category: {failure['category']}\n\n"
                report += f"- **Query:** {failure['query']}\n"
                report += f"- **Expected Tools:** {', '.join(failure['expected_tools'])}\n"
                report += f"- **Failed Scorers:** {', '.join(failure['failed_scorers'])}\n\n"
        
        # Recommendations
        if analysis["recommendations"]:
            report += "\n## Recommendations\n\n"
            
            # Group by priority
            high_priority = [r for r in analysis["recommendations"] if r["priority"] == "high"]
            medium_priority = [r for r in analysis["recommendations"] if r["priority"] == "medium"]
            
            if high_priority:
                report += "### 🔴 High Priority\n\n"
                for rec in high_priority:
                    report += f"- **{rec['area']}:** {rec['issue']}\n"
                    report += f"  - *Suggestion:* {rec['suggestion']}\n\n"
            
            if medium_priority:
                report += "### 🟡 Medium Priority\n\n"
                for rec in medium_priority:
                    report += f"- **{rec['area']}:** {rec['issue']}\n"
                    report += f"  - *Suggestion:* {rec['suggestion']}\n\n"
        
        report += "\n---\n\n*Report generated by Agent Analysis Tool*\n"
    
    else:
        # Plain text format
        report = f"""
{'='*70}
Agent Evaluation Report
{'='*70}

Agent: {agent_name}
Dataset: {dataset_name}
Evaluation Date: {timestamp}

{'='*70}
Executive Summary
{'='*70}

Total Test Cases: {analysis['summary']['total_test_cases']}
"""
        
        # Add summary stats
        if "uses_correct_tools" in analysis["scorer_stats"]:
            stats = analysis["scorer_stats"]["uses_correct_tools"]
            report += f"Tool Usage Pass Rate: {stats['pass_rate']}% ({stats['pass_count']}/{stats['total']})\n"
        
        # Continue with text format...
        report += "\n[Text format implementation continues...]\n"
    
    return report


def save_analysis_artifacts(
    analysis: dict[str, Any],
    report: str,
    agent_run_id: str,
    artifact_path: str = "evaluation_analysis"
) -> None:
    """
    Save analysis results as MLflow artifacts.
    
    Args:
        analysis: Analysis dictionary.
        report: Generated report string.
        agent_run_id: Run ID of the agent being evaluated.
        artifact_path: Path within MLflow run to save artifacts.
    """
    # Create a temporary directory for artifacts
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save JSON analysis
        json_file = temp_path / "analysis.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save report
        report_file = temp_path / "report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # End any active run before starting a new one
        # MLflow evaluation may leave an active run
        active_run = mlflow.active_run()
        while active_run:
            mlflow.end_run()
            active_run = mlflow.active_run()
        
        # Get the agent's experiment ID
        agent_run = mlflow.get_run(agent_run_id)
        agent_experiment_id = agent_run.info.experiment_id
        
        # Set the experiment to match the agent's experiment
        mlflow.set_experiment(experiment_id=agent_experiment_id)
        
        # Log artifacts to the agent's run
        with mlflow.start_run(run_id=agent_run_id):
            mlflow.log_artifact(str(json_file), artifact_path)
            mlflow.log_artifact(str(report_file), artifact_path)
            
            # Log summary metrics
            if "scorer_stats" in analysis:
                for scorer_name, stats in analysis["scorer_stats"].items():
                    if stats["type"] == "binary":
                        mlflow.log_metric(f"eval_{scorer_name}_pass_rate", stats["pass_rate"])
            
            # Log category metrics
            if "category_breakdown" in analysis:
                for category, stats in analysis["category_breakdown"].items():
                    mlflow.log_metric(f"eval_category_{category}_pass_rate", stats["pass_rate"])
        
        print(f"\n✓ Analysis artifacts saved to run {agent_run_id}")
        print(f"  - {artifact_path}/analysis.json")
        print(f"  - {artifact_path}/report.md")


# ==================== MAIN EXECUTION ====================

async def main(
    agent_run_id: str,
    dataset_name: str,
    dataset_experiment: str = "evaluation-datasets",
    eval_experiment: str = "agent-evaluation",
    save_artifacts: bool = True
):
    """
    Main execution function for agent analysis.
    
    Args:
        agent_run_id: MLflow run ID of the agent to evaluate.
        dataset_name: Name of the dataset to use for evaluation.
        dataset_experiment: Experiment containing the dataset.
        eval_experiment: Experiment to log evaluation results.
        save_artifacts: Whether to save analysis artifacts to MLflow.
    """
    print("\n" + "="*70)
    print("Agent Analysis - MLflow Evaluation Framework")
    print("="*70)
    
    # Get agent name from run
    agent_run = mlflow.get_run(agent_run_id)
    agent_name = agent_run.data.params.get("agent_class", "Unknown Agent")
    
    print(f"\nAgent: {agent_name}")
    print(f"Agent Run ID: {agent_run_id}")
    print(f"Dataset: {dataset_name}")
    print(f"Dataset Experiment: {dataset_experiment}")
    
    # Step 1: Load the agent
    print("\n" + "="*70)
    print("Step 1: Loading Agent")
    print("="*70)
    
    try:
        agent_model = load_agent(agent_run_id)
        print(f"✓ Agent loaded successfully")
    except Exception as e:
        print(f"✗ Error loading agent: {e}")
        return
    
    # Step 2: Load the dataset
    print("\n" + "="*70)
    print("Step 2: Loading Dataset")
    print("="*70)
    
    try:
        dataset = load_dataset(dataset_name, dataset_experiment)
        print(f"✓ Dataset loaded: {len(dataset)} test cases\n")
        
        # Validate dataset structure
        if dataset and len(dataset) > 0:
            first_case = dataset[0]
            if 'inputs' not in first_case or 'expectations' not in first_case:
                print("⚠️  Warning: Dataset may not have the expected structure.")
                print(f"   Expected keys: ['inputs', 'expectations']")
                print(f"   Found keys: {list(first_case.keys())}")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Create prediction function
    print("\n" + "="*70)
    print("Step 3: Creating Prediction Function")
    print("="*70)
    
    predict_fn = create_predict_function(agent_model)
    print("✓ Prediction function created")
    
    # Step 4: Run evaluation
    print("\n" + "="*70)
    print("Step 4: Running Evaluation")
    print("="*70)
    
    try:
        results = run_evaluation(
            dataset=dataset,
            predict_fn=predict_fn,
            experiment_name=eval_experiment
        )
        print("\n✓ Evaluation completed successfully")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Analyze results
    print("\n" + "="*70)
    print("Step 5: Analyzing Results")
    print("="*70)
    
    try:
        analysis = analyze_evaluation_results(results, dataset)
        print("✓ Analysis completed")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Total Test Cases: {analysis['summary']['total_test_cases']}")
        if "uses_correct_tools" in analysis["scorer_stats"]:
            stats = analysis["scorer_stats"]["uses_correct_tools"]
            print(f"  Tool Usage Pass Rate: {stats['pass_rate']}%")
        
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Generate report
    print("\n" + "="*70)
    print("Step 6: Generating Report")
    print("="*70)
    
    try:
        report = generate_analysis_report(
            analysis=analysis,
            agent_name=agent_name,
            dataset_name=dataset_name,
            output_format="markdown"
        )
        print("✓ Report generated")
        
        # Print a preview
        print("\nReport Preview:")
        print("-" * 70)
        print(report[:500] + "\n[...truncated...]")
        print("-" * 70)
        
    except Exception as e:
        print(f"✗ Error generating report: {e}")
        return
    
    # Step 7: Save artifacts
    if save_artifacts:
        print("\n" + "="*70)
        print("Step 7: Saving Artifacts to MLflow")
        print("="*70)
        
        try:
            save_analysis_artifacts(
                analysis=analysis,
                report=report,
                agent_run_id=agent_run_id
            )
        except Exception as e:
            print(f"✗ Error saving artifacts: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nView results in MLflow UI:")
    print(f"  Agent Run: {mlflow.get_tracking_uri()}/#/experiments/{agent_run.info.experiment_id}/runs/{agent_run_id}")
    print(f"  Evaluation: {mlflow.get_tracking_uri()}/#/experiments (look for '{eval_experiment}')")
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python agent_analysis.py <agent_run_id> <dataset_name> [dataset_experiment] [eval_experiment]")
        print("\nExample:")
        print("  python agent_analysis.py abc123 customer_service_demo_dataset")
        sys.exit(1)
    
    agent_run_id = sys.argv[1]
    dataset_name = sys.argv[2]
    dataset_experiment = sys.argv[3] if len(sys.argv) > 3 else "evaluation-datasets"
    eval_experiment = sys.argv[4] if len(sys.argv) > 4 else "agent-evaluation"
    
    # Run the analysis
    asyncio.run(main(
        agent_run_id=agent_run_id,
        dataset_name=dataset_name,
        dataset_experiment=dataset_experiment,
        eval_experiment=eval_experiment
    ))
