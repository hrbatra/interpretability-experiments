import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuration
DATA_DIR = "cot_predictability_data"
RESULTS_DIR = "cot_predictability_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load the collected data
print("Loading data...")
with open(f"{DATA_DIR}/all_data.json", "r") as f:
    all_data = json.load(f)

# Extract features and labels
early_reps = []
late_reps = []
labels = []

print(f"Processing {len(all_data)} samples...")
for data_point in all_data:
    # Load the saved representations
    early_rep = np.load(data_point["early_rep_path"])
    late_rep = np.load(data_point["late_rep_path"])
    
    # Convert model's answer to binary label
    label = 1 if data_point["model_answer"] == "T" else 0
    
    early_reps.append(early_rep)
    late_reps.append(late_rep)
    labels.append(label)

# Convert to arrays
early_reps = np.array(early_reps)
late_reps = np.array(late_reps)
labels = np.array(labels)

print(f"Data shapes:")
print(f"Early representations: {early_reps.shape}")
print(f"Late representations: {late_reps.shape}")
print(f"Labels: {labels.shape}")

# Split data
X_train_early, X_test_early, y_train_early, y_test_early = train_test_split(
    early_reps, labels, test_size=0.2, random_state=42
)

X_train_late, X_test_late, y_train_late, y_test_late = train_test_split(
    late_reps, labels, test_size=0.2, random_state=42
)

# Train logistic regression probes
print("Training early representation probe...")
early_probe = LogisticRegression(max_iter=1000, class_weight='balanced')
early_probe.fit(X_train_early, y_train_early)

print("Training pre-CoT representation probe...")
late_probe = LogisticRegression(max_iter=1000, class_weight='balanced')
late_probe.fit(X_train_late, y_train_late)

# Evaluate probes
y_pred_early = early_probe.predict(X_test_early)
early_acc = accuracy_score(y_test_early, y_pred_early)

y_pred_late = late_probe.predict(X_test_late)
late_acc = accuracy_score(y_test_late, y_pred_late)

# Generate report
results = {
    "early_representation": {
        "accuracy": float(early_acc),
        "report": classification_report(y_test_early, y_pred_early, output_dict=True)
    },
    "late_representation": {
        "accuracy": float(late_acc),
        "report": classification_report(y_test_late, y_pred_late, output_dict=True)
    }
}

print("\nRESULTS:")
print(f"Early (Embedding) Representation Accuracy: {early_acc:.4f}")
print(f"Late (Pre-CoT) Representation Accuracy: {late_acc:.4f}")

# Save results
with open(f"{RESULTS_DIR}/probe_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Visualize results
plt.figure(figsize=(10, 6))
accuracies = [early_acc, late_acc]
layers = ["Embedding Layer", "Pre-CoT Layer"]
plt.bar(layers, accuracies, color=['blue', 'orange'])
plt.ylabel('Probe Accuracy')
plt.title('How well can we predict the model\'s final answer?')
plt.ylim(0, 1.0)
plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance')
plt.text(0, 0.51, 'Random Chance', color='red')

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center')

plt.savefig(f"{RESULTS_DIR}/probe_accuracy.png")
plt.close()

print(f"Results saved to {RESULTS_DIR}")

# Interpretation of results
interpretation = """
# CoT Predictability Experiment Results

## Interpretation

- **Early Representation Accuracy**: {early_acc:.2%}
  Measures how well the model's initial embedding representations predict its final answer.

- **Pre-CoT Representation Accuracy**: {late_acc:.2%}
  Measures how well the model's representations *just before* starting CoT predict its final answer.

## Conclusions

If the Pre-CoT accuracy is high (e.g., >80%):
- The model likely "knows" its answer before the reasoning process begins
- CoT serves more as justification/articulation rather than genuine reasoning

If the Pre-CoT accuracy is low (e.g., <60%):
- The model genuinely uses the CoT process to determine its answer
- The reasoning steps meaningfully influence the final output

The difference between early and pre-CoT accuracy indicates how much information 
about the final answer is accumulated during the forward pass before reasoning begins.
""".format(early_acc=early_acc, late_acc=late_acc)

with open(f"{RESULTS_DIR}/interpretation.md", "w") as f:
    f.write(interpretation)

print("Done!")
