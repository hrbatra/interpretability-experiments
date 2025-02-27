# Interpretability Experiments

This repository contains experiments exploring interpretability aspects of large language models.

## Experiment 1: Chain-of-Thought Predictability

### Overview
This experiment investigates how well a language model's final answer after Chain-of-Thought (CoT) reasoning can be predicted from its internal representations before it starts the reasoning process.

### Hypothesis
A model trained to use CoT may already "know" its final answer or have a "hunch" before the explicit reasoning process begins, and the CoT process serves primarily to articulate a justification rather than to determine the answer.

**Narrow hypothesis:** Models tend to exhibit some degree of 'predecision', and a probe on their activations should weakly correlate with their final answer. The correlation likely increases with model size.

### Method
1. Extract model activations at the point where reasoning begins (at the `<think>` tag)
2. Train a linear probe to predict the model's own final answer
3. High probe accuracy would suggest the model's decision is largely determined before explicit reasoning begins

### Getting Started

#### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool (venv or conda)

#### Setup
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/interpretability-experiments.git
   cd interpretability-experiments
   ```

2. Run the setup script to create a virtual environment and install dependencies:
   ```
   python setup.py
   ```

3. Activate the virtual environment:
   ```
   source .venv/bin/activate  # On Linux/Mac
   .venv\Scripts\activate     # On Windows
   ```

#### Running Experiments
The main experiment script offers several options:

```
python run_experiment.py [OPTIONS]
```

Common options:
- `--model MODEL_NAME`: Model name or path (default: local_models/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B)
- `--device DEVICE`: Device to run on (cuda, mps, cpu)
- `--samples N`: Number of samples to use (default: 50)
- `--collect-data`: Collect model activations and save them
- `--train-probes`: Train linear probes on collected activations  
- `--analyze-results`: Analyze and visualize results

Example (full pipeline with default model):
```
python run_experiment.py --samples 50
```

Example (with specific model and GPU):
```
python run_experiment.py --model gpt2 --device cuda --samples 100
```

Example (analyze existing results):
```
python run_experiment.py --analyze-results
```

### Project Structure
```
interpretability-experiments/
├── README.md
├── requirements.txt
├── run_experiment.py     # Main experiment script
├── setup.py              # Environment setup script
├── src/
│   ├── data/             # Dataset loading and preprocessing
│   ├── models/           # Probe models
│   ├── utils/            # Utility functions
│   └── analysis/         # Analysis and visualization
├── results/              # Experiment results
└── logs/                 # Experiment logs
```

### Results

Results will be saved in the `results/{model_name}/` directory, including:
- `results.json`: Raw model outputs and answers
- `early_activations.npy`: Activations from early in the model
- `late_activations.npy`: Activations from late in the model
- `probe_results.json`: Evaluation metrics for the probes
- `visualizations/`: Plots and visualizations

### Expected Results
[To be added after running experiments]

### Implications
[To be added after analysis]
