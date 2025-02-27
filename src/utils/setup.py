"""
Setup utilities for the interpretability experiments.
"""
import os
import subprocess
import sys
from pathlib import Path

def create_venv(venv_path=".venv"):
    """
    Create a virtual environment.
    
    Args:
        venv_path: Path to create the virtual environment at
    """
    if os.path.exists(venv_path):
        print(f"Virtual environment already exists at {venv_path}")
        return
    
    try:
        # Try to use python -m venv first
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        print(f"Created virtual environment at {venv_path}")
    except subprocess.CalledProcessError:
        # Fall back to virtualenv if venv fails
        try:
            subprocess.run(["virtualenv", venv_path], check=True)
            print(f"Created virtual environment at {venv_path} using virtualenv")
        except subprocess.CalledProcessError:
            print("Failed to create virtual environment. Please install virtualenv or ensure Python venv module is available.")
            sys.exit(1)

def install_requirements(venv_path=".venv", requirements_path="requirements.txt"):
    """
    Install requirements into the virtual environment.
    
    Args:
        venv_path: Path to the virtual environment
        requirements_path: Path to the requirements file
    """
    if not os.path.exists(requirements_path):
        print(f"Requirements file not found at {requirements_path}")
        return False
    
    # Determine the activate script based on platform
    if sys.platform == "win32":
        activate_script = os.path.join(venv_path, "Scripts", "activate")
        pip_path = os.path.join(venv_path, "Scripts", "pip")
    else:
        activate_script = os.path.join(venv_path, "bin", "activate")
        pip_path = os.path.join(venv_path, "bin", "pip")
    
    # Install requirements using the virtual environment's pip
    if os.path.exists(pip_path):
        # First upgrade pip
        upgrade_cmd = [pip_path, "install", "--upgrade", "pip"]
        subprocess.run(upgrade_cmd, check=True)
        
        # Then install requirements
        install_cmd = [pip_path, "install", "-r", requirements_path]
        subprocess.run(install_cmd, check=True)
        print(f"Installed requirements from {requirements_path}")
        return True
    else:
        print(f"Failed to find pip in virtual environment at {venv_path}")
        return False

def setup_environment(venv_path=".venv", requirements_path="requirements.txt"):
    """
    Set up the environment for the experiments.
    
    Args:
        venv_path: Path to create/use the virtual environment
        requirements_path: Path to the requirements file
    
    Returns:
        True if setup was successful, False otherwise
    """
    try:
        create_venv(venv_path)
        success = install_requirements(venv_path, requirements_path)
        
        if success:
            # Create directories if they don't exist
            for dir_name in ["data", "results", "logs"]:
                os.makedirs(dir_name, exist_ok=True)
                
            print("Environment setup complete!")
            
            # Print activation instructions
            if sys.platform == "win32":
                print(f"\nActivate the environment with: {venv_path}\\Scripts\\activate")
            else:
                print(f"\nActivate the environment with: source {venv_path}/bin/activate")
                
            return True
        else:
            print("Environment setup failed during requirements installation.")
            return False
    except Exception as e:
        print(f"Error during environment setup: {e}")
        return False

if __name__ == "__main__":
    setup_environment()