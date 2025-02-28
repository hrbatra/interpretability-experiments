# interpretability-experiments

My current curiosity centers on mechanistically understanding the role of Chain of Thought (CoT) in reasoning models. Specifically, I'm intrigued by generalizable differences between outputs enclosed in <think>  vs <answer>  tags, and by testing conditions in which the model believes its thoughts to be monitored versus unmonitored. In a preliminary experiment (available on my GitHub), I trained a linear probe on late-layer activations of a model's forward pass just before it began to <think>, to predict the model's final answer to a multiple-choice question. Essentially, I want to see whether—or under what circumstances—a reasoning model might pre-decide an answer, using the CoT primarily as a post-hoc rationalization. 

I'm also curious about probing various points of the reasoning process and experimenting with targeted injections of intermediate thoughts. These methods feel both highly scalable and illuminating, though my current setup has major limitations—most notably, the lack of proper frontier-level reasoning models. So far, I've tested the 1.5B parameter DeepSeek distillation and plan to evaluate slightly larger models, but these are not true reasoning models; they were trained via SFT distillation of R1's reasoning traces rather than through pure RL.

For my initial project, I’d build on my existing probe-based methods to investigate at which stages a model commits to its final answer and how that aligns (or misaligns) with the chain of thought it produces. This would involve:

- Collecting data from e.g., Gemini Flash Thinking at multiple points in the generation process
- Training probes to see if the model is “locked in” to an answer before producing its reasoning trace
- Experimenting with minimal interventions to shift the model’s final outputs if we detect “hidden” decisions early on
- If the above methods prove fruitful, shifting the test focus from reasoning-style questions to standard safety evals ("how do I build a nerve agent?")

These experiments could lead to better safety and interpretability tools, such as real-time detection of potential misalignment between a model’s internal decision state and its external chain of thought.
