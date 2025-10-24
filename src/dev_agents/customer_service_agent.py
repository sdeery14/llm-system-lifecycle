"""
Customer Service Agent using OpenAI Agents SDK.

This module provides a customer service agent with realistic tools for handling
customer inquiries, order management, and account operations.
"""

from typing import Any
from typing_extensions import TypedDict
from datetime import datetime, timedelta

from agents import Agent, FunctionTool, RunContextWrapper, function_tool


# ==================== FAKE DATA ====================

# Fake customer data
FAKE_CUSTOMERS = {
    "CUST001": {
        "customer_id": "CUST001",
        "name": "Alice Johnson",
        "email": "alice.johnson@email.com",
        "phone": "+1-555-0101",
        "account_balance": 125.50,
        "rewards_points": 25.00,
        "join_date": "2023-01-15"
    },
    "CUST002": {
        "customer_id": "CUST002",
        "name": "Bob Smith",
        "email": "bob.smith@email.com",
        "phone": "+1-555-0102",
        "account_balance": 0.00,
        "rewards_points": 150.75,
        "join_date": "2022-08-22"
    },
    "CUST003": {
        "customer_id": "CUST003",
        "name": "Carol Martinez",
        "email": "carol.martinez@email.com",
        "phone": "+1-555-0103",
        "account_balance": 450.00,
        "rewards_points": 75.50,
        "join_date": "2024-03-10"
    },
    "CUST004": {
        "customer_id": "CUST004",
        "name": "David Chen",
        "email": "david.chen@email.com",
        "phone": "+1-555-0104",
        "account_balance": 89.99,
        "rewards_points": 10.00,
        "join_date": "2024-06-05"
    }
}

# Fake order data
FAKE_ORDERS = {
    "ORD12345": {
        "order_id": "ORD12345",
        "customer_id": "CUST001",
        "status": "processing",
        "items": ["Wireless Headphones", "USB-C Cable"],
        "total": 79.99,
        "order_date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
        "estimated_delivery": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
        "tracking_number": "TRK987654321"
    },
    "ORD12346": {
        "order_id": "ORD12346",
        "customer_id": "CUST002",
        "status": "shipped",
        "items": ["Laptop Stand", "Wireless Mouse"],
        "total": 145.50,
        "order_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
        "estimated_delivery": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "tracking_number": "TRK123456789"
    },
    "ORD12347": {
        "order_id": "ORD12347",
        "customer_id": "CUST003",
        "status": "delivered",
        "items": ["Mechanical Keyboard"],
        "total": 129.99,
        "order_date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
        "estimated_delivery": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
        "tracking_number": "TRK555666777"
    },
    "ORD12348": {
        "order_id": "ORD12348",
        "customer_id": "CUST001",
        "status": "cancelled",
        "items": ["Phone Case"],
        "total": 19.99,
        "order_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "estimated_delivery": None,
        "tracking_number": None
    },
    "ORD12349": {
        "order_id": "ORD12349",
        "customer_id": "CUST004",
        "status": "processing",
        "items": ["Monitor", "HDMI Cable", "Desk Lamp"],
        "total": 399.99,
        "order_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
        "estimated_delivery": (datetime.now() + timedelta(days=4)).strftime("%Y-%m-%d"),
        "tracking_number": None
    }
}

# Fake knowledge base articles
FAKE_KNOWLEDGE_BASE = [
    {
        "article_id": "KB001",
        "title": "Shipping Information",
        "category": "shipping",
        "content": "Our standard shipping takes 3-5 business days. Express shipping is available for an additional $15 fee and delivers within 1-2 business days. Free shipping on orders over $50.",
        "keywords": ["shipping", "delivery", "standard", "express", "free shipping"]
    },
    {
        "article_id": "KB002",
        "title": "Return Policy",
        "category": "returns",
        "content": "We accept returns within 30 days of purchase. Items must be in original condition with tags attached. Refunds are processed within 5-7 business days after we receive the returned item.",
        "keywords": ["return", "refund", "policy", "30 days", "original condition"]
    },
    {
        "article_id": "KB003",
        "title": "Billing and Payment Methods",
        "category": "billing",
        "content": "We accept all major credit cards, PayPal, and Apple Pay. Charges appear on your statement as 'ACME STORE'. You can update your payment method in your account settings.",
        "keywords": ["billing", "payment", "credit card", "paypal", "apple pay"]
    },
    {
        "article_id": "KB004",
        "title": "Tracking Your Order",
        "category": "shipping",
        "content": "Once your order ships, you'll receive a tracking number via email. You can track your package using the tracking link provided or by visiting our order tracking page.",
        "keywords": ["tracking", "track order", "shipping status", "delivery status"]
    },
    {
        "article_id": "KB005",
        "title": "Rewards Program",
        "category": "rewards",
        "content": "Earn 1 point for every dollar spent. Every 100 points equals $10 in rewards. Points never expire and can be used on any purchase.",
        "keywords": ["rewards", "points", "loyalty", "earn", "redeem"]
    },
    {
        "article_id": "KB006",
        "title": "Warranty Information",
        "category": "warranty",
        "content": "Most products come with a 1-year manufacturer warranty. Extended warranties are available for purchase. Contact us within the warranty period for any defects or issues.",
        "keywords": ["warranty", "guarantee", "defect", "manufacturer", "coverage"]
    },
    {
        "article_id": "KB007",
        "title": "Account Security",
        "category": "account",
        "content": "Protect your account by using a strong password and enabling two-factor authentication. Never share your password with anyone. Contact us immediately if you suspect unauthorized access.",
        "keywords": ["security", "password", "two-factor", "authentication", "account safety"]
    }
]

# Store for processed refunds (simulates a refund tracking system)
PROCESSED_REFUNDS = {}


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
    
    # Look up order in fake data
    if order_id not in FAKE_ORDERS:
        return f"Sorry, I couldn't find an order with ID {order_id}. Please check the order number and try again."
    
    order = FAKE_ORDERS[order_id]
    items_str = ", ".join(order["items"])
    
    status_messages = {
        "processing": f"Order {order_id} is currently being processed. Items: {items_str}. Total: ${order['total']:.2f}. Estimated delivery: {order['estimated_delivery']}.",
        "shipped": f"Order {order_id} has been shipped! Items: {items_str}. Total: ${order['total']:.2f}. Tracking number: {order['tracking_number']}. Estimated delivery: {order['estimated_delivery']}.",
        "delivered": f"Order {order_id} was delivered on {order['estimated_delivery']}. Items: {items_str}. Total: ${order['total']:.2f}.",
        "cancelled": f"Order {order_id} has been cancelled. Items: {items_str}. If you have questions about this cancellation, please let me know."
    }
    
    return status_messages.get(order["status"], f"Order {order_id} status: {order['status']}")


@function_tool
async def get_account_balance(account_info: AccountInfo) -> str:
    """Retrieve the current account balance for a customer.
    
    Args:
        account_info: The account information containing the customer ID.
    
    Returns:
        A string with the account balance information.
    """
    customer_id = account_info["customer_id"]
    
    # Look up customer in fake data
    if customer_id not in FAKE_CUSTOMERS:
        return f"Sorry, I couldn't find a customer account with ID {customer_id}. Please verify the customer ID."
    
    customer = FAKE_CUSTOMERS[customer_id]
    
    return (f"Account for {customer['name']} (ID: {customer_id}):\n"
            f"- Account Balance: ${customer['account_balance']:.2f}\n"
            f"- Rewards Points: ${customer['rewards_points']:.2f} available\n"
            f"- Email: {customer['email']}\n"
            f"- Phone: {customer['phone']}\n"
            f"- Member since: {customer['join_date']}")


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
    
    # Check if order exists
    if order_id not in FAKE_ORDERS:
        return f"Sorry, I couldn't find an order with ID {order_id}. Please verify the order number."
    
    order = FAKE_ORDERS[order_id]
    
    # Check if order is eligible for refund
    if order["status"] == "cancelled":
        return f"Order {order_id} has already been cancelled. No refund is needed."
    
    if order["status"] == "processing":
        # Can cancel instead of refund
        refund_amount = order["total"]
        PROCESSED_REFUNDS[order_id] = {
            "amount": refund_amount,
            "reason": reason,
            "status": "cancelled",
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        return f"Order {order_id} has been cancelled before shipping. Full refund of ${refund_amount:.2f} will be processed within 3-5 business days. Reason: {reason}"
    
    # Process refund for shipped/delivered orders
    refund_amount = order["total"]
    PROCESSED_REFUNDS[order_id] = {
        "amount": refund_amount,
        "reason": reason,
        "status": "refund_initiated",
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    
    return (f"Refund request for order {order_id} has been initiated. "
            f"Amount: ${refund_amount:.2f}. Reason: {reason}. "
            f"You should see the refund in your account within 5-7 business days. "
            f"Please return the items using the prepaid shipping label we'll email you.")


@function_tool
def search_knowledge_base(ctx: RunContextWrapper[Any], query: str, category: str | None = None) -> str:
    """Search the customer service knowledge base for helpful articles.
    
    Args:
        query: The search query to look up in the knowledge base.
        category: Optional category to narrow down the search (e.g., 'shipping', 'returns', 'billing').
    
    Returns:
        Relevant information from the knowledge base.
    """
    query_lower = query.lower()
    matching_articles = []
    
    for article in FAKE_KNOWLEDGE_BASE:
        # Check category filter
        if category and article["category"] != category.lower():
            continue
        
        # Check if query matches title, content, or keywords
        if (query_lower in article["title"].lower() or 
            query_lower in article["content"].lower() or 
            any(query_lower in keyword.lower() for keyword in article["keywords"])):
            matching_articles.append(article)
    
    if not matching_articles:
        category_str = f" in category '{category}'" if category else ""
        return f"No articles found matching '{query}'{category_str}. Please try different search terms or contact us for direct assistance."
    
    # Return the top matching article(s)
    if len(matching_articles) == 1:
        article = matching_articles[0]
        return f"Found 1 article matching '{query}':\n\n**{article['title']}**\n{article['content']}"
    else:
        result = f"Found {len(matching_articles)} articles matching '{query}':\n\n"
        for i, article in enumerate(matching_articles[:3], 1):  # Show top 3
            result += f"{i}. **{article['title']}** ({article['category']})\n   {article['content']}\n\n"
        if len(matching_articles) > 3:
            result += f"...and {len(matching_articles) - 3} more articles."
        return result


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
    # Check if customer exists
    if customer_id not in FAKE_CUSTOMERS:
        return f"Sorry, I couldn't find a customer account with ID {customer_id}. Please verify the customer ID."
    
    if not email and not phone:
        return "Please provide at least one contact detail to update (email or phone)."
    
    customer = FAKE_CUSTOMERS[customer_id]
    updates = []
    
    if email:
        old_email = customer["email"]
        customer["email"] = email
        updates.append(f"email from {old_email} to {email}")
    
    if phone:
        old_phone = customer["phone"]
        customer["phone"] = phone
        updates.append(f"phone from {old_phone} to {phone}")
    
    update_str = " and ".join(updates)
    
    return (f"Successfully updated {update_str} for {customer['name']} (ID: {customer_id}). "
            f"You'll receive a confirmation email at your updated address.")


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
