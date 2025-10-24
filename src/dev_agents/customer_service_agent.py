"""
Customer Service Agent using OpenAI Agents SDK.

This module provides a customer service agent with realistic tools for handling
customer inquiries, order management, and account operations.
"""

from typing import Any
from typing_extensions import TypedDict

from agents import Agent, FunctionTool, RunContextWrapper, function_tool


class OrderInfo(TypedDict):
    """Order information structure."""
    order_id: str


class AccountInfo(TypedDict):
    """Account information structure."""
    customer_id: str


class RefundRequest(TypedDict):
    """Refund request structure."""
    order_id: str
    reason: str


@function_tool
async def check_order_status(order_info: OrderInfo) -> str:
    """Check the status of a customer order.
    
    Args:
        order_info: The order information containing the order ID.
    
    Returns:
        A string describing the current order status.
    """
    order_id = order_info["order_id"]
    # In a real system, this would query a database
    return f"Order {order_id} is currently being processed and will ship within 2-3 business days."


@function_tool
async def get_account_balance(account_info: AccountInfo) -> str:
    """Retrieve the current account balance for a customer.
    
    Args:
        account_info: The account information containing the customer ID.
    
    Returns:
        A string with the account balance information.
    """
    customer_id = account_info["customer_id"]
    # In a real system, this would query a database
    return f"Customer {customer_id} has an account balance of $125.50 with $25.00 in rewards points available."


@function_tool
async def process_refund(refund_request: RefundRequest) -> str:
    """Process a refund request for a customer order.
    
    Args:
        refund_request: The refund request containing order ID and reason.
    
    Returns:
        A confirmation message about the refund status.
    """
    order_id = refund_request["order_id"]
    reason = refund_request["reason"]
    # In a real system, this would initiate a refund process
    return f"Refund request for order {order_id} has been initiated. Reason: {reason}. You should see the refund in 5-7 business days."


@function_tool
def search_knowledge_base(ctx: RunContextWrapper[Any], query: str, category: str | None = None) -> str:
    """Search the customer service knowledge base for helpful articles.
    
    Args:
        query: The search query to look up in the knowledge base.
        category: Optional category to narrow down the search (e.g., 'shipping', 'returns', 'billing').
    
    Returns:
        Relevant information from the knowledge base.
    """
    # In a real system, this would perform a semantic search on a knowledge base
    category_str = f" in category '{category}'" if category else ""
    return f"Found 3 articles matching '{query}'{category_str}. Top result: Our standard shipping takes 3-5 business days. Express shipping is available for an additional fee."


@function_tool
async def update_customer_contact(ctx: RunContextWrapper[Any], customer_id: str, email: str | None = None, phone: str | None = None) -> str:
    """Update customer contact information.
    
    Args:
        customer_id: The customer ID to update.
        email: Optional new email address.
        phone: Optional new phone number.
    
    Returns:
        Confirmation of the contact information update.
    """
    updates = []
    if email:
        updates.append(f"email to {email}")
    if phone:
        updates.append(f"phone to {phone}")
    
    update_str = " and ".join(updates)
    # In a real system, this would update the database
    return f"Successfully updated {update_str} for customer {customer_id}."


class CustomerServiceAgent:
    """
    A customer service agent that can handle common customer inquiries.
    
    This agent is equipped with tools to:
    - Check order status
    - Retrieve account balances
    - Process refunds
    - Search the knowledge base
    - Update customer contact information
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the customer service agent.
        
        Args:
            model: The OpenAI model to use for the agent (default: gpt-4o).
        """
        self.agent = Agent(
            name="CustomerServiceAgent",
            model=model,
            instructions="""You are a helpful customer service representative. 
            Your goal is to assist customers with their inquiries in a friendly and professional manner.
            
            You can help with:
            - Checking order status
            - Providing account information
            - Processing refund requests
            - Answering common questions by searching the knowledge base
            - Updating customer contact information
            
            Always be polite, empathetic, and solution-oriented. If you need information from the customer
            to use a tool (like an order ID or customer ID), politely ask for it.
            """,
            tools=[
                check_order_status,
                get_account_balance,
                process_refund,
                search_knowledge_base,
                update_customer_contact,
            ],
        )
    
    def get_agent(self) -> Agent:
        """
        Get the underlying Agent instance.
        
        Returns:
            The configured Agent instance.
        """
        return self.agent
    
    def list_tools(self) -> None:
        """Print information about all available tools."""
        print("Available Tools:")
        print("=" * 50)
        for tool in self.agent.tools:
            if isinstance(tool, FunctionTool):
                print(f"\nTool Name: {tool.name}")
                print(f"Description: {tool.description}")
                print("Parameters Schema:")
                import json
                print(json.dumps(tool.params_json_schema, indent=2))
                print("-" * 50)


# Example usage
if __name__ == "__main__":
    import asyncio
    from agents import Runner
    
    async def interactive_chat():
        """Run an interactive CLI chat with the customer service agent."""
        # Create the customer service agent
        cs_agent = CustomerServiceAgent()
        agent = cs_agent.get_agent()
        
        print("\n" + "=" * 60)
        print("Customer Service Chat")
        print("=" * 60)
        print("Type your questions or requests below.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("=" * 60 + "\n")
        
        # Initialize conversation history
        conversation_history = None
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nThank you for contacting customer service. Goodbye!")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Prepare the input for the agent
                if conversation_history is None:
                    # First turn - just pass the user message
                    agent_input = user_input
                else:
                    # Subsequent turns - use to_input_list() to maintain history
                    agent_input = conversation_history.to_input_list() + [
                        {"role": "user", "content": user_input}
                    ]
                
                # Run the agent
                result = await Runner.run(agent, agent_input)
                
                # Update conversation history for next turn
                conversation_history = result
                
                # Display agent response
                print(f"\nAgent: {result.final_output}\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Thank you for contacting customer service. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
                continue
    
    # Run the interactive chat
    asyncio.run(interactive_chat())
