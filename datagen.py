import os
import json
import numpy as np
import torch
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
MODEL_NAME = "DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_PATH = f'local_models/deepseek-ai_{MODEL_NAME}'
N_SAMPLES = 1100  # Full dataset size (20 for testing)
SAVE_DIR = "cot_predictability_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model and tokenizer
print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Setup activation hooks
hooked_activations = {
    "early_rep": None,
    "late_rep": None,
}

def early_hook_fn(module, input, output):
    # Store embedding layer output
    hooked_activations["early_rep"] = output.clone().detach()

def late_hook_fn(module, input, output):
    # Store late forward pass output (just before CoT begins)
    hooked_activations["late_rep"] = output.clone().detach()

# Register hooks
early_hook = model.model.embed_tokens.register_forward_hook(early_hook_fn)
late_hook = model.model.layers[-2].mlp.register_forward_hook(late_hook_fn)

# Load dataset
print("Loading dataset...")
dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
train_data = dataset["train"].select(range(N_SAMPLES))

# Function to extract the final answer from model's output
def extract_final_answer(text):
    # Look for boxed answer format
    boxed_match = re.search(r'\\boxed{([^}]+)}', text)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        return answer
    
    # Look for "answer" tag format
    answer_match = re.search(r'<answer>([^<]+)</answer>', text)
    if answer_match:
        answer = answer_match.group(1).strip()
        return answer
    
    # Look for T/F at the end
    final_lines = text.strip().split('\n')[-3:]  # Check last few lines
    for line in final_lines:
        if re.match(r'^[TF]$', line.strip()):
            return line.strip()
    
    # If all else fails, look for any standalone T or F
    tf_match = re.search(r'\b([TF])\b', text)
    if tf_match:
        return tf_match.group(1)
    
    return None

# Storage for data
all_data = []

# Process dataset
print(f"Processing {N_SAMPLES} questions...")
for idx, sample in tqdm(enumerate(train_data), total=len(train_data)):
    question = sample["question"]
    all_choices = sample["choices"]["text"]
    
    for choice_idx, choice_text in enumerate(all_choices):
        # Construct prompt for this question-choice pair
        prompt = f"Question: {question}\nPlease respond T or F based on the following statement: {choice_text}\n<think>\n"
        
        # Step 1: Run full generation to get model's answer after CoT
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        
        # Extract the CoT part and the final answer
        cot_part = full_output[len(prompt):]
        model_answer = extract_final_answer(cot_part)
        
        if model_answer not in ['T', 'F']:
            print(f"Warning: Couldn't extract clear T/F answer for question {idx}, choice {choice_idx}")
            print(f"Output: {cot_part}")
            continue
        
        # Step 2: Extract model representations at the CoT boundary
        with torch.no_grad():
            _ = model(**inputs)
        
        # Get the representations
        early_rep = hooked_activations["early_rep"].mean(dim=1).cpu().numpy().flatten()
        late_rep = hooked_activations["late_rep"].mean(dim=1).cpu().numpy().flatten()
        
        # Store the data
        data_point = {
            "question_id": idx,
            "choice_id": choice_idx,
            "question": question,
            "choice": choice_text,
            "model_answer": model_answer,
            "cot_text": cot_part,
            "early_rep_path": f"{SAVE_DIR}/early_rep_{idx}_{choice_idx}.npy",
            "late_rep_path": f"{SAVE_DIR}/late_rep_{idx}_{choice_idx}.npy",
        }
        
        all_data.append(data_point)
        
        # Save the representations
        np.save(f"{SAVE_DIR}/early_rep_{idx}_{choice_idx}.npy", early_rep)
        np.save(f"{SAVE_DIR}/late_rep_{idx}_{choice_idx}.npy", late_rep)
        
        # Periodically save progress
        if len(all_data) % 100 == 0:
            with open(f"{SAVE_DIR}/data_partial.json", "w") as f:
                json.dump(all_data, f, indent=2)

# Remove hooks
early_hook.remove()
late_hook.remove()

# Save final data
with open(f"{SAVE_DIR}/all_data.json", "w") as f:
    json.dump(all_data, f, indent=2)

print(f"Data collection complete. {len(all_data)} samples processed.")

# Create metadata file with experiment details
metadata = {
    "model": MODEL_NAME,
    "dataset": "ARC-Challenge",
    "num_samples": len(all_data),
    "hypothesis": "Testing if model's final answer after CoT can be predicted from pre-CoT representations",
    "representation_details": {
        "early": "Embedding layer output (averaged across sequence)",
        "late": "Second-to-last MLP layer output (averaged across sequence)"
    }
}

with open(f"{SAVE_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Done! Data saved to:", SAVE_DIR)
