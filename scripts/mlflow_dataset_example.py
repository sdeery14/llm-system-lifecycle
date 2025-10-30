import mlflow
from mlflow.genai.datasets import create_dataset
from dotenv import load_dotenv
load_dotenv()

# Create dataset with manual test cases
dataset = create_dataset(
    name="regression_test_suite",
    experiment_id=["0"],
    tags={"type": "regression", "priority": "critical"},
)

# Define test cases with expected outputs
test_cases = [
    {
        "inputs": {
            "question": "How do I reset my password?",
            "context": "user_support",
        },
        "expectations": {
            "answer": "To reset your password, click 'Forgot Password' on the login page, enter your email, and follow the link sent to your inbox",
            "contains_steps": True,
            "tone": "helpful",
            "max_response_time": 2.0,
        },
    },
    {
        "inputs": {
            "question": "What are your refund policies?",
            "context": "customer_service",
        },
        "expectations": {
            "includes_timeframe": True,
            "mentions_exceptions": True,
            "accuracy": 1.0,
        },
    },
]

dataset.merge_records(test_cases)