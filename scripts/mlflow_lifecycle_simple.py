"""
Simplified MLflow Agent Lifecycle Script

A standalone version that demonstrates logging, registering, promoting, and serving
a customer service agent using MLflow's ResponsesAgent pattern.

This version bundles everything needed and can be run independently.
"""

import os
import asyncio
from typing import Generator, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.models import set_model

# Import dependencies for the customer service agent
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from agents import Agent, function_tool, Runner
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()


# ==================== EXAMPLE USAGE ====================

def main():
    """
    Main function demonstrating the complete MLflow lifecycle:
    1. Log agent
    2. Register model
    3. Promote to production
    4. Load from registry
    5. Interactive chat
    """
    
    print("\n" + "=" * 70)
    print("MLflow Customer Service Agent Lifecycle Demo")
    print("=" * 70)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Error: OPENAI_API_KEY not found.")
        print("Please set your OpenAI API key in .env file or environment.")
        return
    
    # Configuration
    model_name = "customer-service-agent"
    experiment_name = "customer-service-experiments"
    
    # Set MLflow configuration
    mlflow.set_tracking_uri("file:///./mlruns")
    mlflow.set_experiment(experiment_name)
    
    print(f"\n📊 Using MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"📊 Experiment: {experiment_name}")
    
    # Step 1: Create and save agent code
    print("\n" + "=" * 70)
    print("STEP 1: Creating Agent Code File")
    print("=" * 70)
    
    agent_file = Path("scripts/customer_service_mlflow_agent.py")
    create_agent_file(agent_file)
    print(f"✓ Created agent file: {agent_file}")
    
    # Step 2: Log the model
    print("\n" + "=" * 70)
    print("STEP 2: Logging Model to MLflow")
    print("=" * 70)
    
    with mlflow.start_run(run_name="cs_agent_v1") as run:
        print(f"✓ Run ID: {run.info.run_id}")
        
        logged_model = mlflow.pyfunc.log_model(
            python_model=str(agent_file),
            artifact_path="model",
            pip_requirements=[
                "mlflow>=3.4.0",
                "pydantic>=2.0.0", 
                "openai-agents>=0.3.0",
                "faiss-cpu>=1.12.0",
                "sentence-transformers>=5.0.0",
                "numpy>=2.0.0",
                "python-dotenv>=1.0.0",
            ],
            metadata={"task": "customer_service"},
        )
        
        mlflow.log_param("model", "gpt-4o")
        mlflow.log_param("tools", 5)
        
        print(f"✓ Model logged: {logged_model.model_uri}")
    
    # Step 3: Register model
    print("\n" + "=" * 70)
    print("STEP 3: Registering Model")
    print("=" * 70)
    
    model_version = mlflow.register_model(
        model_uri=logged_model.model_uri,
        name=model_name
    )
    
    print(f"✓ Model registered: {model_name}")
    print(f"✓ Version: {model_version.version}")
    
    # Step 4: Promote to production
    print("\n" + "=" * 70)
    print("STEP 4: Promoting to Production")
    print("=" * 70)
    
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"✓ Version {model_version.version} promoted to Production")
    
    # Step 5: Load production model
    print("\n" + "=" * 70)
    print("STEP 5: Loading Production Model")
    print("=" * 70)
    
    prod_model_uri = f"models:/{model_name}/Production"
    loaded_model = mlflow.pyfunc.load_model(prod_model_uri)
    
    print(f"✓ Loaded from: {prod_model_uri}")
    
    # Step 6: Test the model
    print("\n" + "=" * 70)
    print("STEP 6: Testing Model")
    print("=" * 70)
    
    test_query = {
        "input": [{"role": "user", "content": "What is the status of order ORD12345?"}]
    }
    
    print(f"\nTest Query: {test_query['input'][0]['content']}")
    result = loaded_model.predict(test_query)
    print(f"\nResponse:\n{result}\n")
    
    # Step 7: Interactive chat
    print("\n" + "=" * 70)
    print("STEP 7: Interactive Chat (Optional)")
    print("=" * 70)
    
    print("\nWould you like to start interactive chat? (y/n): ", end="")
    if input().strip().lower() == 'y':
        interactive_chat(loaded_model)
    
    print("\n" + "=" * 70)
    print("✓ Demo Complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  - View in MLflow UI: mlflow ui")
    print(f"  - Serve model: mlflow models serve -m {prod_model_uri} -p 5000")


def create_agent_file(file_path: Path):
    """Create the agent Python file for MLflow models-from-code."""
    
    agent_code = '''"""
Customer Service Agent for MLflow - Standalone Implementation
"""

import os
import asyncio
from typing import Generator, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.models import set_model

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from agents import Agent, function_tool, Runner
from typing_extensions import TypedDict

load_dotenv()


# ==================== DATA STRUCTURES ====================

@dataclass
class VSConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class FaissVectorStore:
    def __init__(self, cfg: VSConfig = VSConfig()):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.docs: list[str] = []
        self.metas: list[dict[str, Any]] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        embs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embs, dtype="float32")

    def add_texts(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None):
        texts = list(texts)
        metadatas = list(metadatas) if metadatas else [{} for _ in texts]
        embs = self._embed(texts)
        self.index.add(embs)
        self.docs.extend(texts)
        self.metas.extend(metadatas)

    def search(self, query: str, k: int = 3) -> list[Tuple[str, dict[str, Any], float]]:
        qv = self._embed([query])
        scores, idxs = self.index.search(qv, k=k)
        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i != -1:
                out.append((self.docs[i], self.metas[i], float(score)))
        return out


# Fake data
FAKE_ORDERS = {
    "ORD12345": {
        "order_id": "ORD12345",
        "status": "processing",
        "items": ["Wireless Headphones", "USB-C Cable"],
        "total": 79.99,
        "order_date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
    }
}

KNOWLEDGE_BASE = [
    {
        "article_id": "KB001",
        "title": "Shipping Information",
        "content": "Standard shipping takes 3-5 business days. Express shipping available for $15 (1-2 days). Free shipping on orders over $50.",
    },
    {
        "article_id": "KB002",
        "title": "Return Policy",
        "content": "Returns accepted within 30 days. Items must be in original condition. Refunds processed within 5-7 business days.",
    }
]

_kb_store = None

def get_kb_store():
    global _kb_store
    if _kb_store is None:
        _kb_store = FaissVectorStore()
        texts = [f"{a[\\'title\\']}: {a[\\'content\\']}" for a in KNOWLEDGE_BASE]
        _kb_store.add_texts(texts, KNOWLEDGE_BASE)
    return _kb_store


# ==================== TOOLS ====================

class OrderInfo(TypedDict):
    order_id: str


@function_tool
async def check_order_status(order_info: OrderInfo) -> str:
    """Check order status."""
    order_id = order_info["order_id"]
    if order_id not in FAKE_ORDERS:
        return f"Order {order_id} not found."
    
    order = FAKE_ORDERS[order_id]
    return f"Order {order_id}: {order[\\'status\\']}. Items: {\\', \\'.join(order[\\'items\\'])}. Total: ${order[\\'total\\']:.2f}"


@function_tool
def search_knowledge_base(ctx, query: str, k: int = 2) -> str:
    """Search knowledge base."""
    store = get_kb_store()
    hits = store.search(query, k=k)
    
    if not hits:
        return "No articles found."
    
    result = f"Found {len(hits)} articles:\\n\\n"
    for i, (doc, meta, score) in enumerate(hits, 1):
        result += f"{i}. **{meta[\\'title\\']}**\\n   {meta[\\'content\\']}\\n\\n"
    return result


# ==================== AGENT ====================

class CustomerServiceMLflowAgent(ResponsesAgent):
    """MLflow ResponsesAgent for customer service."""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.agent = Agent(
            name="CustomerServiceAgent",
            model=model,
            instructions="""You are a helpful customer service representative.
            Help with orders, shipping, returns, and general questions.
            Be polite and professional.""",
            tools=[check_order_status, search_knowledge_base],
        )
    
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Process request and return response."""
        # Extract messages
        input_messages = []
        for msg in request.input:
            msg_dict = msg.model_dump() if hasattr(msg, \\'model_dump\\') else msg
            input_messages.append(msg_dict)
        
        # Simple case: single user message
        if len(input_messages) == 1 and input_messages[0].get("role") == "user":
            query = input_messages[0].get("content", "")
        else:
            query = input_messages
        
        # Run agent
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(Runner.run(self.agent, query))
        
        # Create response
        output_item = self.create_text_output_item(
            text=result.final_output,
            id=f"msg_{id(result)}"
        )
        
        return ResponsesAgentResponse(output=[output_item])
    
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream response."""
        response = self.predict(request)
        item_id = f"msg_{id(response)}"
        text = response.output[0]["content"][0]["text"]
        
        # Stream in chunks
        chunk_size = 20
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=chunk, item_id=item_id)
            )
        
        # Done event
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=text, id=item_id)
        )


# Create agent instance
agent = CustomerServiceMLflowAgent(model="gpt-4o")
set_model(agent)
'''
    
    # Write to file
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(agent_code)


def interactive_chat(loaded_model):
    """Interactive chat with the loaded model."""
    print("\n" + "=" * 70)
    print("Interactive Chat")
    print("Type 'quit', 'exit', or 'bye' to end")
    print("=" * 70 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            request = {
                "input": [{"role": "user", "content": user_input}]
            }
            
            result = loaded_model.predict(request)
            
            # Extract response text
            if isinstance(result, dict) and "output" in result:
                output_items = result["output"]
                for item in output_items:
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if content.get("type") == "output_text":
                                print(f"\nAgent: {content.get('text')}\n")
            else:
                print(f"\nAgent: {result}\n")
            
        except KeyboardInterrupt:
            print("\\n\\nGoodbye!")
            break
        except Exception as e:
            print(f"\\nError: {e}\\n")


if __name__ == "__main__":
    main()
