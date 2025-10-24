from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
 
OpenAIAgentsInstrumentor().instrument()

from langfuse import get_client
 
langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

import asyncio
from agents import Agent, Runner
 
async def main():
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
    )
 
    result = await Runner.run(agent, "Tell me about recursion in programming.")
    print(result.final_output)
 

asyncio.run(main())