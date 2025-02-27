#!/usr/bin/env python
"""
Main experiment runner script for interpretability experiments.
This script runs the full pipeline of data collection, model activation extraction,
probe training, and analysis.
"""
import os
import argparse
import json
import logging
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path

# Import our modules
from src.utils.model_utils import ModelWrapper
from src.data.dataset import DatasetManager
from src.models.probes import LinearProbe
from src.analysis.visualization import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_class_distribution,
    save_results_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'experiment.log'), mode='a')
    ]
)
logger = logging.getLogger("experiment")

# Create directories if they don't exist
for dir_name in ['data', 'results', 'logs', 'models']:
    os.makedirs(dir_name, exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run interpretability experiments")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="local_models/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Model name or path")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cuda, mps, cpu). If not specified, will use best available.")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="allenai/ai2_arc",
                        help="Dataset name")
    parser.add_argument("--split", type=str, default="ARC-Challenge",
                        help="Dataset split")
    
    # Experiment configuration
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of samples to use")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting index in the dataset")
    parser.add_argument("--save-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Actions
    parser.add_argument("--collect-data", action="store_true",
                        help="Collect model activations and save them")
    parser.add_argument("--train-probes", action="store_true",
                        help="Train linear probes on collected activations")
    parser.add_argument("--analyze-results", action="store_true",
                        help="Analyze and visualize results")
    
    # If no actions specified, do all
    args = parser.parse_args()
    if not (args.collect_data or args.train_probes or args.analyze_results):
        args.collect_data = True
        args.train_probes = True
        args.analyze_results = True
    
    return args

def collect_data(args):
    """Collect model activations and responses."""
    logger.info(f"Starting data collection with model {args.model}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize model and dataset
    model_wrapper = ModelWrapper(args.model, args.device)
    dataset_manager = DatasetManager(args.dataset, args.split)
    
    # Get samples
    samples = dataset_manager.get_sample(args.samples, args.start_idx)
    
    # Process each sample
    all_results = []
    early_activations = []
    late_activations = []
    labels = []
    
    logger.info(f"Processing {len(samples)} samples...")
    
    for i, item in enumerate(tqdm(samples)):
        # Build prompt
        prompt = dataset_manager.build_prompt(item)
        
        # Generate response and collect activations
        model_answer_text, activations = model_wrapper.generate(
            prompt, 
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # Parse the answer
        parsed_answer = dataset_manager.parse_answer(model_answer_text)
        
        # Save model activations
        if 'early_rep' in activations and 'late_rep' in activations:
            early_activations.append({'early_rep': activations['early_rep']})
            late_activations.append({'late_rep': activations['late_rep']})
            
            # Convert answer to numeric label if possible
            true_answer = item["answerKey"]
            model_answer = parsed_answer
            
            if true_answer in dataset_manager.label_map:
                true_label = dataset_manager.label_map[true_answer]
            else:
                true_label = None
                
            if model_answer in dataset_manager.label_map:
                model_label = dataset_manager.label_map[model_answer]
                labels.append(model_label)
            else:
                model_label = None
        
        # Save results
        result = {
            "id": i,
            "question": item["question"],
            "true_answer": true_answer,
            "true_label": true_label,
            "model_answer": model_answer,
            "model_label": model_label,
            "full_output": model_answer_text
        }
        
        all_results.append(result)
        
        # Log progress
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Processed {i+1}/{len(samples)} samples")
    
    # Save all data
    save_path = os.path.join(args.save_dir, f"{Path(args.model).name}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save results summary
    with open(os.path.join(save_path, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save activations and labels
    np.save(os.path.join(save_path, "early_activations.npy"), early_activations)
    np.save(os.path.join(save_path, "late_activations.npy"), late_activations)
    np.save(os.path.join(save_path, "labels.npy"), labels)
    
    logger.info(f"Data collection complete. Saved to {save_path}")
    
    # Clean up
    model_wrapper.cleanup()
    
    return save_path, early_activations, late_activations, labels, all_results

def train_probes(save_path, early_activations, late_activations, labels, args):
    """Train linear probes on the collected activations."""
    logger.info("Training linear probes...")
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Check if we have enough data
    if len(labels) < 10:
        logger.warning(f"Very small dataset ({len(labels)} examples) - results may be unreliable")
    
    # Split data into train/test (70/30 split)
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.3, random_state=args.seed, stratify=labels
    )
    
    # Train early probe
    early_probe = LinearProbe(activation_type="early_rep", random_state=args.seed)
    early_train_acts = [early_activations[i] for i in train_indices]
    early_test_acts = [early_activations[i] for i in test_indices]
    
    train_labels = [labels[i] for i in train_indices]
    test_labels = [labels[i] for i in test_indices]
    
    logger.info(f"Training early probe with {len(train_labels)} examples...")
    early_probe.fit(early_train_acts, train_labels)
    
    early_results = early_probe.evaluate(early_test_acts, test_labels)
    logger.info(f"Early probe accuracy: {early_results['accuracy']:.4f}")
    
    # Train late probe
    late_probe = LinearProbe(activation_type="late_rep", random_state=args.seed)
    late_train_acts = [late_activations[i] for i in train_indices]
    late_test_acts = [late_activations[i] for i in test_indices]
    
    logger.info(f"Training late probe with {len(train_labels)} examples...")
    late_probe.fit(late_train_acts, train_labels)
    
    late_results = late_probe.evaluate(late_test_acts, test_labels)
    logger.info(f"Late probe accuracy: {late_results['accuracy']:.4f}")
    
    # Save probes
    early_probe.save_model(os.path.join(save_path, "early_probe.pkl"))
    late_probe.save_model(os.path.join(save_path, "late_probe.pkl"))
    
    # Save evaluation results
    combined_results = {
        "early_probe": early_results,
        "late_probe": late_results,
        "train_size": len(train_labels),
        "test_size": len(test_labels),
        "class_distribution": {str(label): train_labels.count(label) for label in set(train_labels)}
    }
    
    with open(os.path.join(save_path, "probe_results.json"), "w") as f:
        json.dump(combined_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    logger.info(f"Probe training complete. Results saved to {save_path}")
    
    return early_probe, late_probe, early_results, late_results, combined_results

def analyze_results(save_path, combined_results, args):
    """Analyze and visualize the experiment results."""
    logger.info("Analyzing results...")
    
    # Create visualization directory
    viz_path = os.path.join(save_path, "visualizations")
    os.makedirs(viz_path, exist_ok=True)
    
    # Plot accuracy comparison
    accuracy_results = {
        "Early Probe": {"accuracy": combined_results["early_probe"]["accuracy"]},
        "Late Probe": {"accuracy": combined_results["late_probe"]["accuracy"]}
    }
    
    plot_accuracy_comparison(
        accuracy_results,
        title=f"Probe Accuracy Comparison - {Path(args.model).name}",
        save_path=os.path.join(viz_path, "accuracy_comparison.png")
    )
    
    # Plot confusion matrices
    early_cm = np.array(combined_results["early_probe"]["confusion_matrix"])
    late_cm = np.array(combined_results["late_probe"]["confusion_matrix"])
    
    early_probe = LinearProbe.load_model(os.path.join(save_path, "early_probe.pkl"))
    late_probe = LinearProbe.load_model(os.path.join(save_path, "late_probe.pkl"))
    
    early_probe.plot_confusion_matrix(
        early_cm,
        save_path=os.path.join(viz_path, "early_confusion_matrix.png")
    )
    
    late_probe.plot_confusion_matrix(
        late_cm,
        save_path=os.path.join(viz_path, "late_confusion_matrix.png")
    )
    
    # Plot class distribution
    class_counts = {int(k): v for k, v in combined_results["class_distribution"].items()}
    plot_class_distribution(
        class_counts,
        title="Class Distribution in Training Data",
        save_path=os.path.join(viz_path, "class_distribution.png")
    )
    
    logger.info(f"Analysis complete. Visualizations saved to {viz_path}")
    
    # Print summary
    logger.info("\nRESULTS SUMMARY:")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset} ({args.split})")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Early probe accuracy: {combined_results['early_probe']['accuracy']:.4f}")
    logger.info(f"Late probe accuracy: {combined_results['late_probe']['accuracy']:.4f}")
    logger.info(f"Random baseline (4 classes): 0.2500")
    
    # Compare early vs late
    early_acc = combined_results['early_probe']['accuracy']
    late_acc = combined_results['late_probe']['accuracy']
    
    if late_acc > early_acc:
        logger.info(f"FINDING: Late representations are more predictive than early ones (+{late_acc - early_acc:.4f})")
    elif early_acc > late_acc:
        logger.info(f"FINDING: Early representations are more predictive than late ones (+{early_acc - late_acc:.4f})")
    else:
        logger.info("FINDING: Early and late representations have equal predictive power")
    
    if early_acc > 0.25:
        logger.info(f"FINDING: Early representations contain information about the final answer (accuracy: {early_acc:.4f} vs random: 0.2500)")
    
    return viz_path

def main():
    """Main function to run the experiment."""
    args = parse_args()
    
    # Set up paths
    logger.info(f"Setting up experiment with model: {args.model}")
    
    save_path = None
    early_activations = None
    late_activations = None
    labels = None
    all_results = None
    combined_results = None
    
    # Collect data if requested
    if args.collect_data:
        save_path, early_activations, late_activations, labels, all_results = collect_data(args)
    else:
        # Try to load existing data
        model_name = Path(args.model).name
        save_path = os.path.join(args.save_dir, model_name)
        
        if os.path.exists(save_path):
            try:
                with open(os.path.join(save_path, "results.json"), "r") as f:
                    all_results = json.load(f)
                
                early_activations = np.load(os.path.join(save_path, "early_activations.npy"), allow_pickle=True)
                late_activations = np.load(os.path.join(save_path, "late_activations.npy"), allow_pickle=True)
                labels = np.load(os.path.join(save_path, "labels.npy"), allow_pickle=True)
                
                logger.info(f"Loaded existing data from {save_path}")
            except FileNotFoundError:
                logger.error(f"Could not find existing data at {save_path}")
                logger.info("Run with --collect-data to generate the data first")
                return
    
    # Train probes if requested
    if args.train_probes:
        if early_activations is None or late_activations is None or labels is None:
            logger.error("Cannot train probes without activations and labels")
            return
            
        early_probe, late_probe, early_results, late_results, combined_results = train_probes(
            save_path, early_activations, late_activations, labels, args
        )
    else:
        # Try to load existing probe results
        try:
            with open(os.path.join(save_path, "probe_results.json"), "r") as f:
                combined_results = json.load(f)
                logger.info(f"Loaded existing probe results from {save_path}")
        except FileNotFoundError:
            logger.error(f"Could not find existing probe results at {save_path}")
            logger.info("Run with --train-probes to train the probes first")
            if not args.analyze_results:
                return
    
    # Analyze results if requested
    if args.analyze_results:
        if combined_results is None:
            logger.error("Cannot analyze results without probe results")
            return
            
        viz_path = analyze_results(save_path, combined_results, args)
    
    logger.info("Experiment completed successfully!")

if __name__ == "__main__":
    main()