# tests/smoke/test_customer_service_agent_smoke.py
"""
SMOKE TESTS — Customer Service Agent

Purpose: ultra-fast checks that the agent constructs correctly and exposes the
expected public surface (name, model override, tools, instructions). These tests
MUST NOT hit the network or any external services.
"""

import pytest

from src.dev_agents.customer_service_agent import (
    CustomerServiceAgent,
)
from agents import FunctionTool


# ---------- helpers ----------

EXPECTED_TOOL_NAMES = {
    "check_order_status",
    "get_account_balance",
    "process_refund",
    "search_knowledge_base",
    "update_customer_contact",
}


def _tools_by_name(agent) -> dict:
    """Return a dict of tool_name -> tool for quick lookup."""
    # Some agent SDKs wrap tools; we only consider FunctionTool instances here.
    return {t.name: t for t in getattr(agent.agent, "tools", []) if isinstance(t, FunctionTool)}


# ---------- smoke tests ----------

def test_constructs_with_default_model():
    """Agent constructs, exposes name/model, and has a tool list."""
    a = CustomerServiceAgent()
    assert a is not None
    assert hasattr(a, "agent") and a.agent is not None

    # Basic surface checks (do not assert exact strings beyond class name)
    assert hasattr(a.agent, "name")
    assert a.agent.name == "CustomerServiceAgent"

    # Model should be set to *something* (exact default may vary by env)
    assert hasattr(a.agent, "model")
    assert isinstance(a.agent.model, str) and len(a.agent.model) > 0

    # Tools list exists
    assert hasattr(a.agent, "tools")
    assert isinstance(a.agent.tools, (list, tuple))


def test_constructs_with_custom_model():
    """Agent accepts a custom model override and reflects it."""
    custom_model = "gpt-4o-mini"
    a = CustomerServiceAgent(model=custom_model)
    assert a.agent.model == custom_model


def test_expected_tools_are_present_and_functiontools():
    """Agent exposes at least the expected tool names as FunctionTool instances."""
    a = CustomerServiceAgent()
    tools_map = _tools_by_name(a)

    # Ensure all expected tools are present (allowing extras)
    missing = EXPECTED_TOOL_NAMES - set(tools_map.keys())
    assert not missing, f"Missing expected tools: {missing}"

    # All tools on the agent should be FunctionTool or ignorable other types;
    # we at least assert that our expected ones are FunctionTool.
    for name in EXPECTED_TOOL_NAMES:
        assert isinstance(tools_map[name], FunctionTool), f"{name} is not a FunctionTool"


def test_instructions_are_present_and_customer_service_focused():
    """Agent has instructions and they mention core intent (keep checks loose)."""
    a = CustomerServiceAgent()
    assert hasattr(a.agent, "instructions")
    instr = (a.agent.instructions or "").lower()
    assert instr, "Agent instructions are empty"

    # Loose keywords — avoid brittle exact strings
    assert "customer" in instr
    assert any(k in instr for k in ("help", "assist", "service")), "Instructions lack service intent keywords"


@pytest.mark.parametrize("tool_name, expected_props_subset", [
    ("check_order_status", {"order_info"}),
    ("get_account_balance", {"account_info"}),
    ("process_refund", {"refund_request"}),
    ("search_knowledge_base", {"query", "category"}),
    ("update_customer_contact", {"customer_id", "email", "phone"}),
])
def test_tool_schema_is_object_and_contains_minimal_properties(tool_name, expected_props_subset):
    """
    Each expected tool should expose a JSON schema that is an object and
    contains a minimal subset of properties we rely on. Keep this lenient.
    """
    a = CustomerServiceAgent()
    tools_map = _tools_by_name(a)
    assert tool_name in tools_map, f"{tool_name} not found on agent"

    tool = tools_map[tool_name]
    # Description should exist but we don't assert specific wording
    assert isinstance(tool.description, str) and tool.description.strip()

    schema = getattr(tool, "params_json_schema", None)
    assert isinstance(schema, dict), f"{tool_name} schema is not a dict"

    # Very loose shape checks
    assert schema.get("type") == "object", f"{tool_name} schema.type must be 'object'"
    props = schema.get("properties") or {}
    assert isinstance(props, dict), f"{tool_name} schema.properties missing or not a dict"

    missing = expected_props_subset - set(props.keys())
    assert not missing, f"{tool_name} schema missing properties: {missing}"


def test_get_agent_returns_underlying_agent_instance():
    """The wrapper's get_agent() should hand back the underlying agent object."""
    a = CustomerServiceAgent()
    assert hasattr(a, "get_agent") and callable(a.get_agent)
    got = a.get_agent()
    assert got is a.agent


def test_list_tools_prints_minimal_info(capsys):
    """
    list_tools() should produce some human-readable output mentioning the known tools.
    This avoids asserting exact formatting to keep the test resilient.
    """
    a = CustomerServiceAgent()
    assert hasattr(a, "list_tools") and callable(a.list_tools)

    a.list_tools()
    output = capsys.readouterr().out

    assert "Available" in output or "Tools" in output
    for name in EXPECTED_TOOL_NAMES:
        assert name in output, f"{name} not listed in output"
