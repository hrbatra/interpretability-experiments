# Development Guide for Interpretability Experiments

## Commands
- Run experiment: `python run_experiment.py [OPTIONS]`
- Install dependencies: `pip install -r requirements.txt`
- Setup environment: `python setup.py`
- Run specific phase: 
  - `python run_experiment.py --collect-data`
  - `python run_experiment.py --train-probes`
  - `python run_experiment.py --analyze-results`

## Code Style Guidelines
- **Imports:** Group standard library, third-party, and local imports separately
- **Formatting:** Follow PEP 8 conventions, line length ≤ 100 characters
- **Types:** Use type annotations for function parameters and return values
- **Naming:**
  - Variables/functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
- **Documentation:** Add docstrings to describe functions, classes, and modules
- **Error handling:** Use try/except blocks with specific exceptions