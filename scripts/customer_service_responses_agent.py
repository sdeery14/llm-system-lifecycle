"""
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
