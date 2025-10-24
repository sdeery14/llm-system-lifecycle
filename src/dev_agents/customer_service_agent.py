"""
Customer Service Agent using OpenAI Agents SDK.

This module provides a customer service agent with realistic tools for handling
customer inquiries, order management, and account operations.
"""

from __future__ import annotations
from typing import Any, Tuple
from typing_extensions import TypedDict
from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from agents import Agent, FunctionTool, RunContextWrapper, function_tool

# Load environment variables from .env file
load_dotenv()


# ==================== VECTOR STORE ====================

@dataclass
class VSConfig:
    """Configuration for the vector store."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class FaissVectorStore:
    """
    Minimal FAISS vector store for semantic search over knowledge base articles.
    Uses sentence transformers for embedding and cosine similarity for search.
    """
    
    def __init__(self, cfg: VSConfig = VSConfig()):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        # Cosine similarity via inner product on L2-normalized vectors
        self.index = faiss.IndexFlatIP(self.dim)
        self.docs: list[str] = []
        self.metas: list[dict[str, Any]] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using the sentence transformer model."""
        embs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embs, dtype="float32")

    def add_texts(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> None:
        """Add texts and their metadata to the vector store."""
        texts = list(texts)
        metadatas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        assert len(texts) == len(metadatas), "texts and metadatas length mismatch"
        embs = self._embed(texts)
        self.index.add(embs)
        self.docs.extend(texts)
        self.metas.extend(metadatas)

    def search(self, query: str, k: int = 5, filter_fn: Any | None = None) -> list[Tuple[str, dict[str, Any], float]]:
        """
        Search for the top-k most similar documents to the query.
        
        Args:
            query: The search query.
            k: Number of results to return.
            filter_fn: Optional function to filter results based on metadata.
        
        Returns:
            List of tuples (document, metadata, score).
        """
        qv = self._embed([query])
        scores, idxs = self.index.search(qv, k=k * 3)  # overfetch to allow filtering
        out: list[Tuple[str, dict[str, Any], float]] = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            doc, meta = self.docs[i], self.metas[i]
            if (filter_fn is None) or filter_fn(meta):
                out.append((doc, meta, float(score)))
            if len(out) >= k:
                break
        return out


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


# ==================== VECTOR STORE INITIALIZATION ====================

# Module-level singleton for the knowledge base vector store
_kb_vector_store: FaissVectorStore | None = None


def _get_kb_vector_store() -> FaissVectorStore:
    """Get or initialize the knowledge base vector store."""
    global _kb_vector_store
    if _kb_vector_store is None:
        _kb_vector_store = FaissVectorStore()
        # Create searchable text combining title and content
        texts = [
            f"{article['title']}: {article['content']}" 
            for article in FAKE_KNOWLEDGE_BASE
        ]
        _kb_vector_store.add_texts(texts, FAKE_KNOWLEDGE_BASE)
    return _kb_vector_store


# ==================== TYPED DICTS ====================

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
def search_knowledge_base(ctx: RunContextWrapper[Any], query: str, category: str | None = None, k: int = 3) -> str:
    """Search the customer service knowledge base for helpful articles using semantic search.
    
    Args:
        query: The search query to look up in the knowledge base.
        category: Optional category to narrow down the search (e.g., 'shipping', 'returns', 'billing').
        k: Number of top results to return (default: 3).
    
    Returns:
        Relevant information from the knowledge base.
    """
    # Get the vector store
    store = _get_kb_vector_store()
    
    # Define filter function if category is specified
    filter_fn = None
    if category:
        def category_filter(meta: dict[str, Any]) -> bool:
            return meta.get("category", "").lower() == category.lower()
        filter_fn = category_filter
    
    # Perform semantic search
    hits = store.search(query, k=k, filter_fn=filter_fn)
    
    if not hits:
        category_str = f" in category '{category}'" if category else ""
        return f"No articles found matching '{query}'{category_str}. Please try different search terms or contact us for direct assistance."
    
    # Format results
    if len(hits) == 1:
        article = hits[0][1]  # metadata
        score = hits[0][2]
        return f"Found 1 article matching '{query}' (relevance: {score:.2f}):\n\n**{article['title']}**\n{article['content']}"
    else:
        result = f"Found {len(hits)} articles matching '{query}':\n\n"
        for i, (doc, meta, score) in enumerate(hits, 1):
            result += f"{i}. **{meta['title']}** ({meta['category']}) - relevance: {score:.2f}\n   {meta['content']}\n\n"
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
    
    async def run_automated_tests():
        """Run simple automated tests to verify the agent is working."""
        print("\n" + "=" * 60)
        print("Running Automated Tests")
        print("=" * 60 + "\n")
        
        cs_agent = CustomerServiceAgent()
        agent = cs_agent.get_agent()
        
        test_cases = [
            {
                "name": "Test 1: Check Order Status",
                "query": "Can you check the status of order ORD12345?",
                "expected_keywords": ["ORD12345", "processing", "Wireless Headphones"]
            },
            {
                "name": "Test 2: Get Account Balance",
                "query": "What is the account balance for customer CUST002?",
                "expected_keywords": ["CUST002", "Bob Smith", "rewards"]
            },
            {
                "name": "Test 3: Search Knowledge Base",
                "query": "How long does shipping take?",
                "expected_keywords": ["shipping", "3-5", "business days"]
            },
            {
                "name": "Test 4: Process Refund",
                "query": "I need to request a refund for order ORD12347 because the item was defective.",
                "expected_keywords": ["refund", "ORD12347", "initiated"]
            }
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for i, test in enumerate(test_cases, 1):
            print(f"{test['name']}")
            print(f"Query: {test['query']}")
            
            try:
                # Run the agent
                result = await Runner.run(agent, test['query'])
                response = result.final_output
                
                print(f"Response: {response[:150]}..." if len(response) > 150 else f"Response: {response}")
                
                # Check if expected keywords are in the response
                keywords_found = [kw for kw in test['expected_keywords'] if kw.lower() in response.lower()]
                
                if len(keywords_found) >= len(test['expected_keywords']) * 0.6:  # At least 60% of keywords
                    print("✓ PASSED")
                    passed_tests += 1
                else:
                    print(f"✗ FAILED - Expected keywords not found. Found: {keywords_found}")
                    failed_tests += 1
                    
            except Exception as e:
                print(f"✗ FAILED - Error: {e}")
                failed_tests += 1
            
            print("-" * 60 + "\n")
        
        # Print summary
        print("=" * 60)
        print(f"Test Summary: {passed_tests} passed, {failed_tests} failed out of {len(test_cases)} total")
        print("=" * 60 + "\n")
        
        return passed_tests == len(test_cases)
    
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
    
    async def main():
        """Main entry point that runs tests first, then interactive chat."""
        # Run automated tests
        tests_passed = await run_automated_tests()
        
        if not tests_passed:
            print("⚠️  Warning: Some tests failed. The agent may not be functioning correctly.")
            print("Do you want to continue to interactive chat anyway? (y/n): ", end="")
            response = input().strip().lower()
            if response != 'y':
                print("Exiting.")
                return
        
        # Run interactive chat
        await interactive_chat()
    
    # Run the main function
    asyncio.run(main())
