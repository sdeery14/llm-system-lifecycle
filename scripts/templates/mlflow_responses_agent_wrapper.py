"""
MLflow ResponsesAgent Wrapper Template

This template wraps an OpenAI Agents SDK agent for MLflow serving.
The actual agent class will be injected during model logging.
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
from agents import Runner

# Add the src directory to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load environment variables
load_dotenv()


class MLflowResponsesAgentWrapper(ResponsesAgent):
    """
    Generic MLflow ResponsesAgent wrapper for OpenAI Agents SDK agents.
    
    This wrapper adapts any OpenAI Agents SDK agent to the MLflow
    ResponsesAgent interface for logging and serving.
    """
    
    def __init__(self, agent_class, model: str = "gpt-4o", **agent_kwargs):
        """
        Initialize the wrapper with an agent class.
        
        Args:
            agent_class: The agent class to wrap (e.g., CustomerServiceAgent).
            model: The OpenAI model to use (default: gpt-4o).
            **agent_kwargs: Additional keyword arguments for the agent class.
        """
        self.model = model
        self.agent_instance = agent_class(model=model, **agent_kwargs)
        
        # Get the underlying Agent from the agent instance
        if hasattr(self.agent_instance, 'get_agent'):
            self.agent = self.agent_instance.get_agent()
        elif hasattr(self.agent_instance, 'agent'):
            self.agent = self.agent_instance.agent
        else:
            # Assume the instance itself is the agent
            self.agent = self.agent_instance
    
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
            # For multi-turn conversations, pass the full message list
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
        # For simplicity, convert the non-streaming response to a stream
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
