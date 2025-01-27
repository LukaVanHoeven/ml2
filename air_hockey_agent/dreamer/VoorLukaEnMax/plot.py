import json
import matplotlib.pyplot as plt
import os
import numpy as np
def load_metrics_from_file(file_path):
    """
    Load metrics from a JSONL file and return organized data structures.
    
    Args:
        file_path (str): Path to the JSONL metrics file
        
    Returns:
        dict: Dictionary containing lists of metrics data
    """
    eval_return, eval_length, eval_episodes = [], [], []
    train_return, train_length, train_episodes = [], [], []
    eval_steps, train_steps = [], []
    
    with open(file_path, "r") as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                if "eval_return" in data:
                    eval_return.append(data["eval_return"])
                    eval_length.append(data.get("eval_length", 0))
                    eval_episodes.append(data.get("eval_episodes", 0))
                    eval_steps.append(data.get("step", 0))
                if "train_return" in data:
                    train_return.append(data["train_return"])
                    train_length.append(data.get("train_length", 0))
                    train_episodes.append(data.get("train_episodes", 0))
                    train_steps.append(data.get("step", 0))
            except json.JSONDecodeError:
                print(f"Skipped invalid JSON line: {line}")
    
    return {
        "eval_return": eval_return,
        "eval_length": eval_length,
        "eval_episodes": eval_episodes,
        "eval_steps": eval_steps,
        "train_return": train_return,
        "train_length": train_length,
        "train_episodes": train_episodes,
        "train_steps": train_steps
    }

def plot_metrics(metrics_data, title_prefix=""):
    """
    Create plots from the metrics data.
    
    Args:
        metrics_data (dict): Dictionary containing metrics data lists
        title_prefix (str): Optional prefix for plot titles
    """
    # Plot evaluation and training returns
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_data["train_steps"], metrics_data["train_return"], 
             label="Train Return", color="blue")
    plt.plot(metrics_data["eval_steps"], metrics_data["eval_return"], 
             label="Eval Return", color="green")
    plt.xlabel("Steps")
    plt.ylabel("Return")
    plt.title(f"{title_prefix}Train vs Eval Return Over Time")
    plt.legend()
    plt.grid()
    
def plot_comparison(normal_metrics, sparse_metrics):
    """
    Create a comparison plot of normal and sparse training returns.
    
    Args:
        normal_metrics (dict): Metrics data from normal training
        sparse_metrics (dict): Metrics data from sparse training
    """
    plt.figure(figsize=(12, 6))
    plt.plot(normal_metrics["train_steps"], normal_metrics["train_return"], 
             label="Normal Train Return", color="blue")
    plt.plot(sparse_metrics["train_steps"], sparse_metrics["train_return"], 
             label="Sparse Train Return", color="red")
    plt.xlabel("Steps")
    plt.ylabel("Return")
    plt.title("Comparison of Normal vs Sparse Training Returns")
    plt.legend()
    plt.grid()
    
def plot_smoothed_comparison(normal_metrics, sparse_metrics):
    """
    Create comparison plots with enhanced error handling
    """
    plt.figure(figsize=(12, 8))
    
    # Smoothing parameters
    window_size = 50  # Reduced window size for smaller datasets
    
    # Plot normal metrics
    if len(normal_metrics["train_return"]) > 0:
        normal_data = np.array(normal_metrics["train_return"])
        normal_steps = np.array(normal_metrics["train_steps"])
        
        # Simple moving average for normal data
        normal_smooth = np.convolve(normal_data, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
        # Adjust steps to align with the smoothed data
        normal_steps_smooth = normal_steps[window_size//2: -window_size//2 + 1]
        
        plt.plot(normal_steps_smooth, normal_smooth, 
                 label="Normal Training", color="blue", linewidth=2)
    
    # Plot sparse metrics
    if len(sparse_metrics["train_return"]) > 0:
        sparse_data = np.array(sparse_metrics["train_return"])
        sparse_steps = np.array(sparse_metrics["train_steps"])
        
        # Simple moving average for sparse data
        sparse_smooth = np.convolve(sparse_data, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
        # Adjust steps to align with the smoothed data
        sparse_steps_smooth = sparse_steps[window_size//2: -window_size//2 + 1]
        
        plt.plot(sparse_steps_smooth, sparse_smooth, 
                 label="Sparse Training", color="red", linewidth=2)
    
    # Add legend and labels
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Return")
    plt.title("Smoothed Comparison of Normal vs Sparse Training")
    plt.grid()
    plt.show()

    


def main():
    # Define file paths
    base_dir = os.path.dirname(__file__)
    normal_log_file = os.path.join(base_dir, "normaleMetrics.jsonl")
    sparse_log_file = os.path.join(base_dir, "sparsemetrics.jsonl")
    
    # Load and plot normal metrics
    normal_metrics = load_metrics_from_file(normal_log_file)
    plot_metrics(normal_metrics, "Normal Metrics - ")
    
    # Load and plot sparse metrics
    sparse_metrics = load_metrics_from_file(sparse_log_file)
    plot_metrics(sparse_metrics, "Sparse Metrics - ")
    
    plot_comparison(normal_metrics, sparse_metrics)
    
    plot_smoothed_comparison(normal_metrics, sparse_metrics)
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()