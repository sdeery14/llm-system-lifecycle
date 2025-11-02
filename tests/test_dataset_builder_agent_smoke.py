"""
Smoke tests for the Dataset Builder Agent.

These tests verify that the dataset builder agent can be instantiated
and has the expected tools and configuration.
"""

import os
import pytest

from app_agents.dataset_builder import (
    DatasetBuilderAgent,
    DatasetBuilderConfig,
    DatasetBuilderState,
)


def test_dataset_builder_config():
    """Test that the dataset builder configuration is properly initialized."""
    config = DatasetBuilderConfig()
    
    # Check default values
    assert config.batch_size == 20
    assert config.model == "gpt-4o"
    assert config.max_dataset_size > 0


def test_dataset_builder_config_from_env(monkeypatch):
    """Test that configuration reads from environment variables."""
    monkeypatch.setenv("MAX_DATASET_INSTANCES", "200")
    
    config = DatasetBuilderConfig()
    assert config.max_dataset_size == 200


def test_dataset_builder_state():
    """Test the dataset builder state management."""
    state = DatasetBuilderState()
    
    # Check initial state
    assert state.target_agent_name is None
    assert state.target_agent_logged is False
    assert state.total_instances_planned == 0
    assert state.instances_created == 0
    assert len(state.created_instances) == 0
    assert state.dataset_created is False


def test_dataset_builder_state_reset():
    """Test that state reset works correctly."""
    state = DatasetBuilderState()
    
    # Set up some state
    state.dataset_plan = {"name": "test", "categories": []}
    state.total_instances_planned = 50
    state.instances_created = 25
    state.created_instances = [{"test": "data"}]
    
    # Reset
    state.reset_dataset_creation()
    
    # Check that dataset creation state is reset but plan is preserved
    assert state.dataset_plan is not None
    assert state.total_instances_planned == 50
    assert state.instances_created == 0
    assert len(state.created_instances) == 0
    assert state.dataset_created is False


def test_dataset_builder_state_summary():
    """Test the state summary generation."""
    state = DatasetBuilderState()
    
    # Initial summary
    summary = state.to_summary()
    assert "Progress:" in summary
    assert "Instances Created: 0/0" in summary
    
    # Add some state
    state.target_agent_name = "TestAgent"
    state.target_agent_logged = True
    state.dataset_plan = {
        "name": "test_dataset",
        "categories": [{"name": "cat1"}, {"name": "cat2"}]
    }
    state.total_instances_planned = 50
    state.instances_created = 25
    
    summary = state.to_summary()
    assert "TestAgent" in summary
    assert "test_dataset" in summary
    assert "25/50" in summary
    assert "Categories: 2" in summary


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_dataset_builder_agent_instantiation():
    """Test that the dataset builder agent can be instantiated."""
    agent = DatasetBuilderAgent()
    
    # Check that the agent has the expected attributes
    assert agent.agent is not None
    assert agent.agent.name == "DatasetBuilderAgent"
    assert len(agent.agent.tools) == 12  # 12 tools defined (updated to match current implementation)
    
    # Check that we can get the agent
    underlying_agent = agent.get_agent()
    assert underlying_agent is not None


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_dataset_builder_agent_tools():
    """Test that the dataset builder agent has all expected tools."""
    agent = DatasetBuilderAgent()
    
    # Get tool names
    tool_names = []
    for tool in agent.agent.tools:
        if hasattr(tool, '__name__'):
            tool_names.append(tool.__name__)
        elif hasattr(tool, 'name'):
            tool_names.append(tool.name)
    
    # Check for expected tools
    expected_tools = [
        'log_target_agent_in_mlflow',
        'create_dataset_plan',
        'approve_dataset_plan',
        'generate_test_cases_batch',
        'finalize_and_store_dataset',
        'get_builder_state_summary',
        'reset_dataset_creation',
    ]
    
    for expected_tool in expected_tools:
        assert any(expected_tool in name for name in tool_names), \
            f"Tool {expected_tool} not found in agent tools"


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_dataset_builder_agent_list_tools(capsys):
    """Test that the agent can list its tools."""
    agent = DatasetBuilderAgent()
    
    agent.list_tools()
    
    captured = capsys.readouterr()
    assert "Dataset Builder Agent Tools:" in captured.out


def test_dataset_builder_config_max_instances_validation():
    """Test that max instances from env is properly validated."""
    # Test with invalid env var (should use default)
    import os
    original = os.environ.get("MAX_DATASET_INSTANCES")
    
    try:
        os.environ["MAX_DATASET_INSTANCES"] = "invalid"
        # This should fall back to default (100) without raising an error
        config = DatasetBuilderConfig()
        # Should have the default value when env var is invalid
        assert config.max_dataset_size == 100
    finally:
        if original:
            os.environ["MAX_DATASET_INSTANCES"] = original
        elif "MAX_DATASET_INSTANCES" in os.environ:
            del os.environ["MAX_DATASET_INSTANCES"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
