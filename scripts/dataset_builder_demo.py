"""
Dataset Builder Demo Script

This script demonstrates how to use the DatasetBuilderAgent to create
high-quality evaluation datasets for LLM-based agents.

The script shows:
1. Logging a target agent in MLflow (or using a previously logged one)
2. Analyzing the target agent's capabilities
3. Creating a dataset plan with categories
4. Generating test cases in batches
5. Storing the final dataset in MLflow

The DatasetBuilderAgent uses advanced features like:
- Parallel generation with worker agents
- Automatic checkpointing for resumable creation
- Diversity checking to avoid duplicate patterns
- Incremental MLflow logging
"""

import os
import sys
import asyncio
from pathlib import Path

import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up paths
project_root = Path(__file__).parent.parent

# Add src to path
sys.path.insert(0, str(project_root / "src"))

from app_agents.dataset_builder import DatasetBuilderAgent


async def automated_demo():
    """
    Run an automated demo of the dataset builder.
    
    This demonstrates the full workflow programmatically by building a real dataset
    for the CustomerServiceAgent with 3 test cases per category.
    """
    print("\n" + "=" * 70)
    print("Dataset Builder - Automated Demo")
    print("=" * 70)
    print("\nThis demo will automatically build a dataset for the CustomerServiceAgent")
    print("with 3 test cases per category.")
    print("=" * 70)
    
    # Import Runner for agent execution
    from agents import Runner
    
    # Create the dataset builder agent
    builder = DatasetBuilderAgent(model="gpt-4o")
    agent = builder.get_agent()
    
    print("\n✓ Dataset Builder Agent initialized.")
    
    # Step 1: Log the target agent
    print("\n" + "=" * 70)
    print("STEP 1: Logging Target Agent in MLflow")
    print("=" * 70)
    
    initial_message = """I want to create a dataset for the CustomerServiceAgent.
    
Please log the target agent with the following details:
- Agent file path: src/dev_agents/customer_service_agent.py
- Agent class name: CustomerServiceAgent
- Agent description: A customer service agent that handles orders, refunds, account queries, and general inquiries
"""
    
    print(f"\nUser: {initial_message.strip()}")
    result = await Runner.run(agent, initial_message)
    print(f"\nAgent: {result.final_output}\n")
    
    # Step 2: Analyze the agent and create a plan
    print("\n" + "=" * 70)
    print("STEP 2: Analyzing Agent and Creating Dataset Plan")
    print("=" * 70)
    
    plan_message = """Great! Now please analyze the agent and create a dataset plan.

I want a small demo dataset with the following categories (3 test cases each):
1. order_status - Testing order status lookups
2. refunds - Testing refund processing
3. account_info - Testing account balance and info retrieval
4. knowledge_base - Testing general knowledge queries
5. contact_updates - Testing customer contact information updates

Total: 15 test cases (3 per category)
Dataset name: customer_service_demo_dataset

For each category:
- Create diverse, realistic test cases
- Include edge cases where appropriate
- Set clear expectations for agent behavior

Please proceed with creating this plan."""
    
    print(f"\nUser: {plan_message.strip()}")
    conversation_input = result.to_input_list() + [{"role": "user", "content": plan_message}]
    result = await Runner.run(agent, conversation_input)
    print(f"\nAgent: {result.final_output}\n")
    
    # Step 3: Approve the plan
    print("\n" + "=" * 70)
    print("STEP 3: Approving Dataset Plan")
    print("=" * 70)
    
    approve_message = """Perfect! I approve this plan. Please proceed with generating the test cases."""
    
    print(f"\nUser: {approve_message}")
    conversation_input = result.to_input_list() + [{"role": "user", "content": approve_message}]
    result = await Runner.run(agent, conversation_input)
    print(f"\nAgent: {result.final_output}\n")
    
    # Step 4: Check if generation is complete (agent may have already generated all cases during approval)
    output_lower = result.final_output.lower()
    
    # Check for completion phrases
    completion_phrases = [
        "15/15",
        "100%",
        "all test cases",
        "already generated",
        "successfully generated",
        "dataset is complete",
        "finalize and store",
        "would you like to finalize"
    ]
    
    is_complete = any(phrase in output_lower for phrase in completion_phrases)
    
    if not is_complete:
        # Need to continue generation
        print("\n(Continuing test case generation...)")
        max_generation_turns = 10  # Safety limit
        generation_turn = 0
        
        while generation_turn < max_generation_turns:
            # Check if we should continue
            output_lower = result.final_output.lower()
            
            # Check for completion
            is_complete = any(phrase in output_lower for phrase in completion_phrases)
            
            if is_complete:
                # Generation complete
                print("\n✓ All test cases generated!")
                break
            elif "error" in output_lower and "complete" not in output_lower:
                # Error occurred
                print("\n✗ Error during generation. Stopping.")
                break
            else:
                # Continue generation
                continue_message = "Please continue generating the remaining test cases."
                print(f"\nUser: {continue_message}")
                conversation_input = result.to_input_list() + [{"role": "user", "content": continue_message}]
                result = await Runner.run(agent, conversation_input)
                print(f"\nAgent: {result.final_output}\n")
                generation_turn += 1
    else:
        print("\n✓ All test cases already generated!")
    
    # Step 5: Finalize and store the dataset (if not already done)
    print("\n" + "=" * 70)
    print("STEP 5: Finalizing and Storing Dataset in MLflow")
    print("=" * 70)
    
    # Check if already finalized
    if "finalized" in result.final_output.lower() and "mlflow" in result.final_output.lower():
        print("✓ Dataset already finalized and stored in MLflow!")
    else:
        finalize_message = """Excellent! Please finalize and store the dataset in MLflow now."""
        
        print(f"\nUser: {finalize_message}")
        conversation_input = result.to_input_list() + [{"role": "user", "content": finalize_message}]
        result = await Runner.run(agent, conversation_input)
        print(f"\nAgent: {result.final_output}\n")
    
    # Summary
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    
    print("\nWhat just happened:")
    print("1. ✓ Logged CustomerServiceAgent in MLflow")
    print("2. ✓ Analyzed agent capabilities and created dataset plan")
    print("3. ✓ Generated 15 test cases (3 per category) using worker agents")
    print("4. ✓ Applied diversity checking to ensure unique test cases")
    print("5. ✓ Stored the dataset in MLflow under 'evaluation-datasets' experiment")
    
    print("\nAdvanced Features Used:")
    print("- Parallel Generation: Worker agents generated batches in parallel")
    print("- Diversity Checking: Prevented duplicate test case patterns")
    print("- Incremental Logging: Batches logged to MLflow as created")
    
    print("\nNext Steps:")
    print("- View the dataset in MLflow UI: http://localhost:5000")
    print("- Use the dataset for agent evaluation")
    print("- Run interactive mode: python src/app_agents/dataset_builder.py")
    
    print("\n" + "=" * 70 + "\n")


async def interactive_demo():
    """
    Run an interactive demo where the user can chat with the agent.
    
    This is similar to running the dataset_builder.py directly.
    """
    print("\n" + "=" * 70)
    print("Dataset Builder - Interactive Demo")
    print("=" * 70)
    print("\nThis agent will help you create evaluation datasets for your agents.")
    print("It will log your target agent, discuss requirements, and create datasets.")
    print("\nType 'quit', 'exit', or 'bye' to end the conversation.")
    print("=" * 70 + "\n")
    
    # Import the interactive chat function
    from app_agents.dataset_builder import interactive_chat
    
    # Run the interactive chat
    await interactive_chat()


async def quick_start_example():
    """
    A quick-start example showing the typical workflow.
    """
    print("\n" + "=" * 70)
    print("Quick Start Example")
    print("=" * 70)
    
    print("\nScenario: Creating a dataset for CustomerServiceAgent")
    print("\nStep-by-step guide:")
    
    print("\n1. First, ensure MLflow is running:")
    print("   mlflow server --host 127.0.0.1 --port 5000")
    
    print("\n2. Run the dataset builder:")
    print("   python src/app_agents/dataset_builder.py")
    
    print("\n3. Choose to use a previously logged agent or log a new one:")
    print("   - Option 1: Use a previously logged agent from MLflow")
    print("   - Option 2: Log a new agent from source file")
    
    print("\n4. If logging a new agent, provide:")
    print("   - Agent file path: src/dev_agents/customer_service_agent.py")
    print("   - Agent class name: CustomerServiceAgent")
    print("   - Agent description: Handles customer service requests")
    
    print("\n5. Review the agent analysis and suggested test categories")
    
    print("\n6. Discuss and refine the dataset requirements:")
    print("   - What scenarios to test")
    print("   - How many test cases per category")
    print("   - Expected outputs and behaviors")
    
    print("\n7. Approve the dataset plan")
    
    print("\n8. The agent will generate test cases in batches")
    print("   - Progress is tracked automatically")
    print("   - Checkpoints saved for resumability")
    print("   - Diversity checks prevent duplicates")
    
    print("\n9. Dataset is stored in MLflow under 'evaluation-datasets' experiment")
    
    print("\n" + "=" * 70)
    print("Configuration Options")
    print("=" * 70)
    
    print("\nEnvironment variables (set in .env):")
    print("  MAX_DATASET_INSTANCES=100    # Maximum dataset size")
    print("  OPENAI_API_KEY=sk-...        # Your OpenAI API key")
    
    print("\nDatasetBuilderConfig options:")
    print("  max_dataset_size: Maximum number of dataset instances allowed")
    print("  batch_size: Create datasets in batches of this size (default: 20)")
    print("  model: LLM model for main agent (default: gpt-4o)")
    print("  worker_model: LLM model for worker agents (default: gpt-4o-mini)")
    print("  enable_parallel_generation: Enable parallel batch generation (default: True)")
    print("  enable_checkpointing: Enable checkpointing (default: True)")
    print("  checkpoint_interval: Save checkpoint every N cases (default: 50)")
    print("  enable_diversity_check: Enable diversity checks (default: True)")
    
    print("\n" + "=" * 70)


async def show_examples():
    """
    Show example dataset plans for different use cases.
    """
    print("\n" + "=" * 70)
    print("Example Dataset Plans")
    print("=" * 70)
    
    print("\n--- Example 1: Customer Service Agent ---")
    print("""
Dataset Plan:
  Name: customer_service_regression_v1
  Total Instances: 100
  
Categories:
  1. order_status (25 cases)
     Description: Test order status lookup functionality
     Example Inputs: "What's the status of order ORD12345?"
     Expectations: Returns correct order status
  
  2. refunds (20 cases)
     Description: Test refund request handling
     Example Inputs: "I want to return my order"
     Expectations: Initiates refund process correctly
  
  3. account_management (15 cases)
     Description: Test account info updates
     Example Inputs: "Update my email address"
     Expectations: Updates user information
  
  4. general_knowledge (25 cases)
     Description: Test general customer service knowledge
     Example Inputs: "What's your return policy?"
     Expectations: Provides accurate information
  
  5. edge_cases (15 cases)
     Description: Test error handling and edge cases
     Example Inputs: Invalid order numbers, malformed requests
     Expectations: Handles errors gracefully
""")
    
    print("\n--- Example 2: RAG Agent ---")
    print("""
Dataset Plan:
  Name: rag_agent_evaluation_v1
  Total Instances: 80
  
Categories:
  1. factual_queries (30 cases)
     Description: Questions requiring factual retrieval
     Example Inputs: "What is the capital of France?"
     Expectations: Correct factual answer with citations
  
  2. multi_hop_reasoning (20 cases)
     Description: Questions requiring multiple retrieval steps
     Example Inputs: "Compare X and Y based on the documents"
     Expectations: Synthesizes information from multiple sources
  
  3. no_answer_cases (15 cases)
     Description: Questions with no answer in knowledge base
     Example Inputs: Questions about non-existent topics
     Expectations: Admits when information is not available
  
  4. ambiguous_queries (15 cases)
     Description: Ambiguous or unclear questions
     Example Inputs: Vague or ambiguous phrasing
     Expectations: Asks for clarification or provides best answer
""")
    
    print("\n--- Example 3: Code Assistant Agent ---")
    print("""
Dataset Plan:
  Name: code_assistant_tests_v1
  Total Instances: 60
  
Categories:
  1. code_generation (20 cases)
     Description: Generate code based on requirements
     Example Inputs: "Write a function to sort a list"
     Expectations: Correct, idiomatic code
  
  2. debugging (15 cases)
     Description: Find and fix bugs in code
     Example Inputs: Code snippets with bugs
     Expectations: Identifies and fixes issues
  
  3. code_explanation (15 cases)
     Description: Explain code functionality
     Example Inputs: "Explain this code snippet"
     Expectations: Clear, accurate explanations
  
  4. best_practices (10 cases)
     Description: Code review and best practices
     Example Inputs: "Review this code for improvements"
     Expectations: Suggests improvements and best practices
""")


async def main():
    """
    Main function to run the demo.
    """
    print("\n" + "=" * 70)
    print("Dataset Builder Demo")
    print("Create high-quality evaluation datasets for your agents")
    print("=" * 70)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        return
    
    # Show MLflow tracking URI
    print(f"\nMLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Ask user which demo to run
    print("\n" + "=" * 70)
    print("Choose a demo mode:")
    print("=" * 70)
    print("\n1. Automated Demo (shows capabilities and workflow)")
    print("2. Interactive Demo (chat with the agent)")
    print("3. Quick Start Guide (step-by-step instructions)")
    print("4. Example Dataset Plans (see example plans)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        await automated_demo()
    elif choice == "2":
        await interactive_demo()
    elif choice == "3":
        await quick_start_example()
    elif choice == "4":
        await show_examples()
    elif choice == "5":
        print("\nGoodbye!")
        return
    else:
        print("\nInvalid choice. Please run the script again.")
        return


if __name__ == "__main__":
    asyncio.run(main())
