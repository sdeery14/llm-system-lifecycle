"""
MLflow Agent Lifecycle Script

This script demonstrates the complete lifecycle of logging, registering, promoting,
and serving the customer service agent using MLflow following the ResponsesAgent pattern.

Based on: https://mlflow.org/docs/3.4.0/genai/serving/responses-agent/#logging-and-serving
"""

import os
import sys
import asyncio
from typing import Generator
from pathlib import Path

import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from dotenv import load_dotenv

# Add the src directory to the path so we can import the customer service agent
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dev_agents.customer_service_agent import CustomerServiceAgent
from agents import Runner

# Load environment variables
load_dotenv()


class CustomerServiceResponsesAgent(ResponsesAgent):
    """
    MLflow ResponsesAgent wrapper for the CustomerServiceAgent.
    
    This class adapts the OpenAI Agents SDK-based customer service agent
    to the MLflow ResponsesAgent interface for logging and serving.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the customer service responses agent.
        
        Args:
            model: The OpenAI model to use for the agent (default: gpt-4o).
        """
        self.model = model
        self.customer_service_agent = CustomerServiceAgent(model=model)
        self.agent = self.customer_service_agent.get_agent()
    
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Process a request and return a structured response.
        
        Args:
            request: The request containing input messages and context.
        
        Returns:
            A ResponsesAgentResponse with the agent's output.
        """
        # Convert input messages to the format expected by OpenAI Agents SDK
        input_messages = []
        for msg in request.input:
            msg_dict = msg.model_dump() if hasattr(msg, 'model_dump') else msg
            input_messages.append(msg_dict)
        
        # For the first message, just pass the content as a string
        if len(input_messages) == 1 and input_messages[0].get("role") == "user":
            query = input_messages[0].get("content", "")
        else:
            # For multi-turn conversations, we need to format properly
            query = input_messages
        
        # Run the agent synchronously using asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(Runner.run(self.agent, query))
        
        # Extract the final output
        final_output = result.final_output
        
        # Create a text output item
        output_item = self.create_text_output_item(
            text=final_output,
            id=f"msg_{id(result)}"
        )
        
        # Return the response with custom outputs if available
        return ResponsesAgentResponse(
            output=[output_item],
            custom_outputs=request.custom_inputs if hasattr(request, 'custom_inputs') else None
        )
    
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Process a request and stream the response.
        
        Args:
            request: The request containing input messages and context.
        
        Yields:
            ResponsesAgentStreamEvent objects as the response is generated.
        """
        # For simplicity, we'll convert the non-streaming response to a stream
        response = self.predict(request)
        
        # Stream the output text in chunks
        item_id = f"msg_{id(response)}"
        text = response.output[0]["content"][0]["text"]
        
        # Split text into chunks for streaming
        chunk_size = 20
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            yield ResponsesAgentStreamEvent(
                **self.create_text_delta(delta=chunk, item_id=item_id)
            )
        
        # Yield the final done event
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=text, id=item_id)
        )


def save_agent_to_file():
    """
    Save the agent class to a separate file for models-from-code logging.
    
    This is required by MLflow's models-from-code approach.
    """
    agent_code = '''"""
Customer Service ResponsesAgent for MLflow serving.
"""

import sys
import asyncio
from typing import Generator
from pathlib import Path

import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from dotenv import load_dotenv

# Add the src directory to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dev_agents.customer_service_agent import CustomerServiceAgent
from agents import Runner

# Load environment variables
load_dotenv()


class CustomerServiceResponsesAgent(ResponsesAgent):
    """MLflow ResponsesAgent wrapper for the CustomerServiceAgent."""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.customer_service_agent = CustomerServiceAgent(model=model)
        self.agent = self.customer_service_agent.get_agent()
    
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Process a request and return a structured response."""
        # Convert input messages
        input_messages = []
        for msg in request.input:
            msg_dict = msg.model_dump() if hasattr(msg, 'model_dump') else msg
            input_messages.append(msg_dict)
        
        # For the first message, just pass the content as a string
        if len(input_messages) == 1 and input_messages[0].get("role") == "user":
            query = input_messages[0].get("content", "")
        else:
            query = input_messages
        
        # Run the agent synchronously
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(Runner.run(self.agent, query))
        final_output = result.final_output
        
        # Create output
        output_item = self.create_text_output_item(
            text=final_output,
            id=f"msg_{id(result)}"
        )
        
        return ResponsesAgentResponse(
            output=[output_item],
            custom_outputs=request.custom_inputs if hasattr(request, 'custom_inputs') else None
        )
    
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Process a request and stream the response."""
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
        
        # Final done event
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=text, id=item_id)
        )


# Create the agent instance for MLflow to use
from mlflow.models import set_model

agent = CustomerServiceResponsesAgent(model="gpt-4o")
set_model(agent)
'''
    
    # Save to scripts directory
    agent_file_path = project_root / "scripts" / "customer_service_responses_agent.py"
    with open(agent_file_path, 'w') as f:
        f.write(agent_code)
    
    return agent_file_path


def log_agent():
    """
    Log the customer service agent to MLflow using models-from-code approach.
    
    Returns:
        The logged model info.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Logging Agent to MLflow")
    print("=" * 70)
    
    # Save agent to file first
    agent_file = save_agent_to_file()
    print(f"✓ Saved agent code to: {agent_file}")
    
    # Print MLflow tracking URI
    print(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Create or set experiment
    experiment_name = "customer-service-agent"
    mlflow.set_experiment(experiment_name)
    print(f"✓ Using experiment: {experiment_name}")
    
    # Start a run and log the model
    with mlflow.start_run(run_name="customer_service_agent_v1") as run:
        print(f"✓ Started run: {run.info.run_id}")
        
        # Log the agent using models-from-code
        logged_agent_info = mlflow.pyfunc.log_model(
            python_model="scripts/customer_service_responses_agent.py",
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
            metadata={"task": "customer_service", "version": "1.0.0"},
        )
        
        print(f"✓ Logged agent to: {logged_agent_info.model_uri}")
        
        # Log additional metadata
        mlflow.log_param("model_type", "gpt-4o")
        mlflow.log_param("agent_type", "customer_service")
        mlflow.log_param("num_tools", 5)
        
        print("✓ Logged parameters")
    
    return logged_agent_info


def register_model(model_uri: str, model_name: str = "customer-service-agent"):
    """
    Register the logged model in the MLflow Model Registry.
    
    Args:
        model_uri: The URI of the logged model.
        model_name: The name to register the model under.
    
    Returns:
        The registered model version.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Registering Model in Model Registry")
    print("=" * 70)
    
    # Register the model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={"task": "customer_service", "framework": "openai-agents"}
    )
    
    print(f"✓ Registered model: {model_name}")
    print(f"✓ Version: {model_version.version}")
    print(f"✓ Current stage: {model_version.current_stage}")
    
    return model_version


def promote_to_production(model_name: str, version: int):
    """
    Promote the model version to production.
    
    Args:
        model_name: The name of the registered model.
        version: The version number to promote.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Promoting Model to Production")
    print("=" * 70)
    
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    
    # Transition the model to production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"✓ Promoted {model_name} version {version} to Production")
    print(f"✓ Previous production versions have been archived")


def load_production_model(model_name: str):
    """
    Load the production version of the model from MLflow.
    
    Args:
        model_name: The name of the registered model.
    
    Returns:
        The loaded model.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Loading Production Model from MLflow")
    print("=" * 70)
    
    # Load the production model
    model_uri = f"models:/{model_name}/Production"
    print(f"Loading from: {model_uri}")
    
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print(f"✓ Successfully loaded production model")
    
    return loaded_model


def test_loaded_model(loaded_model):
    """
    Test the loaded model with a sample query.
    
    Args:
        loaded_model: The loaded MLflow model.
    """
    print("\n" + "=" * 70)
    print("STEP 5: Testing Loaded Model")
    print("=" * 70)
    
    # Test queries
    test_queries = [
        {
            "input": [{"role": "user", "content": "Can you check the status of order ORD12345?"}],
            "context": {"conversation_id": "test-001", "user_id": "test-user"}
        },
        {
            "input": [{"role": "user", "content": "How long does shipping take?"}],
            "context": {"conversation_id": "test-002", "user_id": "test-user"}
        }
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Input: {query['input'][0]['content']}")
        
        try:
            result = loaded_model.predict(query)
            print(f"✓ Response received")
            print(f"Output: {result}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("-" * 70)


def interactive_chat(loaded_model):
    """
    Run an interactive chat session with the loaded model.
    
    Args:
        loaded_model: The loaded MLflow model.
    """
    print("\n" + "=" * 70)
    print("STEP 6: Interactive Chat with Loaded Model")
    print("=" * 70)
    print("Type your questions or requests below.")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("=" * 70 + "\n")
    
    conversation_id = "interactive-session"
    user_id = "demo-user"
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nThank you for using the customer service agent. Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Create the request
            request = {
                "input": [{"role": "user", "content": user_input}],
                "context": {"conversation_id": conversation_id, "user_id": user_id}
            }
            
            # Get the response from the model
            result = loaded_model.predict(request)
            
            # Extract and display the response
            # The result is a ResponsesAgentResponse
            if isinstance(result, dict) and "output" in result:
                # Extract text from the output
                output_items = result["output"]
                for item in output_items:
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if content.get("type") == "output_text":
                                print(f"\nAgent: {content.get('text')}\n")
            else:
                print(f"\nAgent: {result}\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Thank you for using the customer service agent. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            continue


def main():
    """
    Main function that orchestrates the complete MLflow agent lifecycle.
    """
    print("\n" + "=" * 70)
    print("MLflow Agent Lifecycle Demo")
    print("Customer Service Agent - Log, Register, Promote, Load, Chat")
    print("=" * 70)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        return
    
    try:
        # Step 1: Log the agent
        logged_info = log_agent()
        
        # Step 2: Register the model
        model_name = "customer-service-agent"
        model_version = register_model(logged_info.model_uri, model_name)
        
        # Step 3: Promote to production
        promote_to_production(model_name, model_version.version)
        
        # Step 4: Load the production model
        loaded_model = load_production_model(model_name)
        
        # Step 5: Test the loaded model
        test_loaded_model(loaded_model)
        
        # Step 6: Interactive chat
        print("\nWould you like to start an interactive chat? (y/n): ", end="")
        response = input().strip().lower()
        if response == 'y':
            interactive_chat(loaded_model)
        
        print("\n" + "=" * 70)
        print("✓ MLflow Agent Lifecycle Demo Completed Successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error during lifecycle demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
