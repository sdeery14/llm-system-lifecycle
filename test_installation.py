"""
Test script to verify package installation and basic functionality.

Run this after installing the package to ensure everything works correctly.
"""

import sys
import os


def test_imports():
    """Test that all necessary modules can be imported."""
    print("Testing imports...")
    
    try:
        import mlflow_eval_tools
        print(f"✓ mlflow_eval_tools imported successfully (v{mlflow_eval_tools.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import mlflow_eval_tools: {e}")
        return False
    
    try:
        from mlflow_eval_tools import cli
        print("✓ mlflow_eval_tools.cli imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import mlflow_eval_tools.cli: {e}")
        return False
    
    try:
        from app_agents import dataset_builder
        print("✓ app_agents.dataset_builder imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import app_agents.dataset_builder: {e}")
        return False
    
    try:
        from app_agents import agent_analysis
        print("✓ app_agents.agent_analysis imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import app_agents.agent_analysis: {e}")
        return False
    
    return True


def test_cli_available():
    """Test that CLI commands are available."""
    print("\nTesting CLI availability...")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ["mlflow-eval-tools", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print(f"✓ CLI available: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ CLI returned error code {result.returncode}")
            return False
    except FileNotFoundError:
        print("✗ CLI command 'mlflow-eval-tools' not found in PATH")
        print("  Make sure the package is installed and your virtual environment is activated")
        return False
    except Exception as e:
        print(f"✗ Error testing CLI: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are available."""
    print("\nTesting dependencies...")
    
    dependencies = [
        "mlflow",
        "pydantic",
        "agents",  # openai-agents
        "click",
        "dotenv",
        "openai",
    ]
    
    all_present = True
    for dep in dependencies:
        try:
            if dep == "agents":
                __import__("agents")
            elif dep == "dotenv":
                __import__("dotenv")
            else:
                __import__(dep)
            print(f"✓ {dep} available")
        except ImportError:
            print(f"✗ {dep} not available")
            all_present = False
    
    return all_present


def test_environment():
    """Test environment configuration."""
    print("\nTesting environment...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✓ OPENAI_API_KEY is set")
    else:
        print("⚠ OPENAI_API_KEY is not set (required for actual usage)")
        print("  Set it in .env file or export OPENAI_API_KEY=your-key")
    
    max_instances = os.getenv("MAX_DATASET_INSTANCES", "100")
    print(f"✓ MAX_DATASET_INSTANCES: {max_instances}")
    
    return True


def test_cli_commands():
    """Test that CLI commands are accessible."""
    print("\nTesting CLI commands...")
    
    import subprocess
    
    commands = [
        ["mlflow-eval-tools", "--help"],
        ["mlflow-eval-tools", "dataset-builder", "--help"],
        ["mlflow-eval-tools", "agent-analysis", "--help"],
        ["mlflow-eval-tools", "info"],
    ]
    
    all_passed = True
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print(f"✓ {' '.join(cmd)}")
            else:
                print(f"✗ {' '.join(cmd)} failed with code {result.returncode}")
                all_passed = False
        except Exception as e:
            print(f"✗ {' '.join(cmd)} error: {e}")
            all_passed = False
    
    return all_passed


def main():
    """Run all tests."""
    print("=" * 70)
    print("mlflow-eval-tools Installation Test")
    print("=" * 70)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("CLI Available", test_cli_available()))
    results.append(("Dependencies", test_dependencies()))
    results.append(("Environment", test_environment()))
    results.append(("CLI Commands", test_cli_commands()))
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ All tests passed! Package is ready to use.")
        print("\nNext steps:")
        print("  1. Set OPENAI_API_KEY in .env file")
        print("  2. Run: mlflow-eval-tools dataset-builder")
        print("  3. See QUICK_START_TEAMS.md for detailed guide")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  - Ensure package is installed: pip install mlflow-eval-tools")
        print("  - Activate your virtual environment")
        print("  - Check that all dependencies are installed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
