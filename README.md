# interpretability-experiments

# first experiment - 

Chain-of-Thought Predictability Experiment
What We're Testing
This experiment investigates whether a language model's final answer after Chain-of-Thought (CoT) reasoning can be predicted from its internal representations before it starts the reasoning process.
Hypothesis: A model trained to use CoT may already "know" its final answer or have a "hunch" before the explicit reasoning process begins, and the CoT process serves primarily to articulate a justification rather than to determine the answer.

*Narrow hypothesis - models tend to exhibit some degree of 'predecision', and the probe on its activations should weakly correlate with its final answer. Likely the correlation increases with model size.*

Method: We extract model activations at the point where reasoning begins (<think> tag), then train a linear probe to predict the model's own final answer. High probe accuracy would suggest the model's decision is largely determined before explicit reasoning.
