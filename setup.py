#!/usr/bin/env python
"""
Setup script to prepare the experiment environment.
"""
import os
import sys
import subprocess
import argparse

# Import our setup utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from utils.setup import setup_environment

def main():
    """Main entry point for setup."""
    parser = argparse.ArgumentParser(description="Set up the environment for interpretability experiments")
    parser.add_argument("--venv", type=str, default=".venv",
                       help="Path to create/use the virtual environment")
    parser.add_argument("--requirements", type=str, default="requirements.txt",
                       help="Path to the requirements file")
    args = parser.parse_args()
    
    print(f"Setting up environment at {args.venv} using requirements from {args.requirements}")
    success = setup_environment(args.venv, args.requirements)
    
    if success:
        print("\nSetup completed successfully!")
        print("\nNext steps:")
        print(f"1. Activate the environment: source {args.venv}/bin/activate (Linux/Mac) or {args.venv}\\Scripts\\activate (Windows)")
        print("2. Run an experiment: python run_experiment.py")
        print("   - Use --help to see available options")
        print("   - Use --collect-data to collect data from a model")
        print("   - Use --train-probes to train probes on collected data")
        print("   - Use --analyze-results to visualize and analyze results")
        print("\nExample (full pipeline with default model):")
        print("python run_experiment.py --samples 50")
        print("\nExample (specific model with GPU):")
        print("python run_experiment.py --model gpt2 --device cuda --samples 100")
    else:
        print("\nSetup failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())