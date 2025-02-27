"""
Visualization utilities for analyzing experiment results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

def plot_accuracy_comparison(results: Dict[str, Dict[str, float]], title: str = "Probe Accuracy Comparison",
                            save_path: Optional[str] = None) -> None:
    """
    Plot accuracy comparison between different probes or models.
    
    Args:
        results: Dictionary mapping model/probe names to dictionaries with 'accuracy' keys
        title: Plot title
        save_path: Path to save the plot to (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Extract names and accuracies
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    
    # Create the bar plot
    sns.barplot(x=names, y=accuracies)
    
    # Add text labels on top of the bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    # Add title and labels
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)  # Set y-axis limit with some headroom for labels
    
    # Add random guess baseline
    if len(accuracies) > 0:
        plt.axhline(y=0.25, color='r', linestyle='--', label='Random Guess (4 choices)')
    
    plt.legend()
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Accuracy comparison plot saved to {save_path}")
    
    plt.close()

def plot_generalization_matrix(matrix: np.ndarray, row_labels: List[str], col_labels: List[str],
                              title: str = "Generalization Performance", save_path: Optional[str] = None) -> None:
    """
    Plot a heatmap showing generalization performance across models or tasks.
    
    Args:
        matrix: 2D array of accuracy or performance values
        row_labels: Labels for the rows (e.g., train models)
        col_labels: Labels for the columns (e.g., test models)
        title: Plot title
        save_path: Path to save the plot to (optional)
    """
    plt.figure(figsize=(10, 8))
    
    # Create the heatmap
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=col_labels, yticklabels=row_labels)
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('Test Model/Dataset')
    plt.ylabel('Train Model/Dataset')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Generalization matrix plot saved to {save_path}")
    
    plt.close()

def plot_class_distribution(class_counts: Dict[int, int], title: str = "Class Distribution",
                           save_path: Optional[str] = None) -> None:
    """
    Plot the distribution of classes in a dataset.
    
    Args:
        class_counts: Dictionary mapping class indices to counts
        title: Plot title
        save_path: Path to save the plot to (optional)
    """
    plt.figure(figsize=(8, 6))
    
    # Extract classes and counts
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Map numeric classes to letters if needed
    class_labels = [chr(65 + c) if isinstance(c, int) and 0 <= c <= 25 else str(c) for c in classes]
    
    # Create the bar plot
    sns.barplot(x=class_labels, y=counts)
    
    # Add text labels on top of the bars
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    
    plt.close()

def plot_results_over_time(results: List[Dict[str, Any]], metric: str = 'accuracy',
                         title: str = "Performance Over Time", save_path: Optional[str] = None) -> None:
    """
    Plot how a metric changes over time or with increasing data.
    
    Args:
        results: List of result dictionaries, each with the metric and a 'step' or 'size' field
        metric: Name of the metric to plot
        title: Plot title
        save_path: Path to save the plot to (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Extract data
    steps = [r.get('step', i) for i, r in enumerate(results)]
    values = [r.get(metric, 0) for r in results]
    
    # Create the line plot
    plt.plot(steps, values, 'o-', linewidth=2, markersize=8)
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('Step / Dataset Size')
    plt.ylabel(metric.capitalize())
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Results over time plot saved to {save_path}")
    
    plt.close()

def save_results_summary(results: Dict[str, Any], filepath: str) -> None:
    """
    Save a summary of results to a JSON file.
    
    Args:
        results: Dictionary of results to save
        filepath: Path to save the results to
    """
    # Make sure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    # Process results
    processed_results = convert_for_json(results)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(processed_results, f, indent=2)
    
    logger.info(f"Results summary saved to {filepath}")