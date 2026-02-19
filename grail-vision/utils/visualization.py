import os
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

def load_all_experiment_data(experiment_dir):
    """
    Load and merge data from all experiment directories.

    Iterates through subdirectories in the given experiment directory,
    reads 'experiment_summary.csv', parses parameters from the directory name,
    and combines everything into a single DataFrame.

    Args:
        experiment_dir (str): Path to the root directory containing experiment logs.

    Returns:
        pd.DataFrame: Combined data from all experiments.
    """
    all_data = []
    
    # Traverse all experiment directories
    experiment_dirs = glob.glob(os.path.join(experiment_dir, "*"))
    
    for exp_dir in experiment_dirs:
        if os.path.isdir(exp_dir):
            csv_file = os.path.join(exp_dir, "experiment_summary.csv")
            if os.path.exists(csv_file):
                try:
                    # Read CSV file
                    df = pd.read_csv(csv_file)
                    
                    # Extract experiment parameters from directory name
                    exp_name = os.path.basename(exp_dir)
                    df['experiment_name'] = exp_name
                    
                    # Parse experiment parameters and add to DataFrame
                    params = parse_experiment_params(exp_name)
                    for key, value in params.items():
                        df[key] = value
                    
                    all_data.append(df)
                    
                except Exception as e:
                    print(f"Failed to load {exp_name}: {e}")
    
    if not all_data:
        raise ValueError("No experiment data found")
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Loaded data from {len(all_data)} experiments, total {len(combined_data)} rows")
    
    return combined_data


def parse_experiment_params(exp_name):
    """
    Parse parameters from the experiment directory name.

    Extracts information like dataset, architecture, optimizer, learning rate,
    batch size, etc., from the directory name string.

    Args:
        exp_name (str): Experiment directory name.

    Returns:
        dict: Dictionary containing parsed parameters.
    """
    params = {}
    
    # Parse dataset
    if 'dataset=cifar10' in exp_name:
        params['dataset'] = 'cifar10'
    
    # Parse architecture
    if 'arch=resnet18' in exp_name:
        params['architecture'] = 'resnet18'
    
    # Parse optimizer
    if 'opt=sgd' in exp_name:
        params['optimizer'] = 'sgd'
    
    # Parse learning rate
    lr_match = re.search(r'lr=([\d.]+)', exp_name)
    if lr_match:
        params['learning_rate'] = float(lr_match.group(1))
    
    # Parse batch size
    bs_match = re.search(r'batch_size=(\d+)', exp_name)
    if bs_match:
        params['batch_size'] = int(bs_match.group(1))
    
    # Parse momentum
    momentum_match = re.search(r'momentum=([\d.]+)', exp_name)
    if momentum_match:
        params['momentum'] = float(momentum_match.group(1))
    
    # Parse weight decay
    wd_match = re.search(r'wd=([\d.]+)', exp_name)
    if wd_match:
        params['weight_decay'] = float(wd_match.group(1))
    
    # Parse L1 regularization
    l1_match = re.search(r'l1=([\d.e-]+)', exp_name)
    if l1_match:
        params['l1_regularization'] = float(l1_match.group(1))
    
    # Parse L2 regularization
    l2_match = re.search(r'l2=([\d.]+)', exp_name)
    if l2_match:
        params['l2_regularization'] = float(l2_match.group(1))
    
    # Parse SAM (Sharpness-Aware Minimization)
    if 'sam=True' in exp_name:
        params['sam'] = True
        sam_rho_match = re.search(r'sam_rho=([\d.]+)', exp_name)
        if sam_rho_match:
            params['sam_rho'] = float(sam_rho_match.group(1))
    else:
        params['sam'] = False
    
    # Parse data augmentation
    if 'rand_aug=True' in exp_name:
        params['rand_aug'] = True
    else:
        params['rand_aug'] = False
    
    # Parse learning rate schedule
    if 'lr_schedule=True' in exp_name:
        params['lr_schedule'] = True
    else:
        params['lr_schedule'] = False
    
    return params


def plot_compressed_vs_compensated_scatter(data, model, method, save_path=None, name=['compressed_accuracy', 'compressed_compensated_accuracy']):
    """
    Generate a scatter plot comparing compressed accuracy vs. compensated accuracy.

    Args:
        data (pd.DataFrame): The experiment data containing accuracy metrics.
        model (str): Name of the model (e.g., 'vit', 'resnet').
        method (str): Name of the compression/compensation method.
        save_path (str, optional): Path to save the generated plot.
        name (list): List of two column names to plot [x_axis_col, y_axis_col].
    """

    # Ensure data is DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
        
    # Filter data for ViT models: exclude points where initial accuracy < 75
    if 'vit' in model.lower():
        title = 'ViT'
        filtered_data = data[data['original_accuracy'] > 75]
        print(f"ViT model detected: filtered from {len(data)} to {len(filtered_data)} data points (removed {len(data) - len(filtered_data)} points with initial accuracy <= 75%)")
    elif 'resnet' in model.lower():
        title = 'ResNet18'
        filtered_data = data
        print(f"Non-ViT model detected: using all {len(data)} data points")
    else:
        title = model
        filtered_data = data
        print(f"Model {model} detected: using all {len(data)} data points")
    
    # Check if columns exist
    if name[0] not in data.columns or name[1] not in data.columns:
        print(f"Skipping plot: Columns {name} not found in data")
        return
        
    # Drop NaNs for the plotting columns
    filtered_data = filtered_data.dropna(subset=name)
    
    # Check if we have any data left after filtering
    if len(filtered_data) == 0:
        print("Warning: No data points remaining after filtering. Skipping plot.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create scatter plot using compression ratio as color
    scatter = ax.scatter(
        filtered_data[name[0]],
        filtered_data[name[1]],
        c=filtered_data['compression_ratio'],
        cmap='viridis',
        alpha=0.7,
        s=50,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Compression Ratio', fontsize=20)
    
    # Add diagonal reference line (y=x)
    min_acc = min(filtered_data[name[0]].min(), filtered_data[name[1]].min())
    max_acc = max(filtered_data[name[0]].max(), filtered_data[name[1]].max())
    ax.plot([min_acc, max_acc], [min_acc, max_acc], 'r--', alpha=0.7, linewidth=2, label='y=x (No compensation effect)')
    
    # Calculate statistics
    improvement = filtered_data[name[1]] - filtered_data[name[0]]
    avg_improvement = improvement.mean()
    correlation = filtered_data[name[0]].corr(filtered_data[name[1]])
    
    # Add statistics text box
    # stats_text = f"""Statistics:
    #     Average Improvement: {avg_improvement:.2f}%
    #     Correlation: {correlation:.3f}
    #     Total Data Points: {len(data)}
    #     Number of Experiments: {data['experiment_name'].nunique()}"""
    
    # ax.text(0.98, 0.05, stats_text, transform=ax.transAxes, fontsize=15,
    #         ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set axis labels and title
    ax.set_xlabel(f'{method.upper()} Test Accuracy (%)', fontsize=20)
    ax.set_ylabel(f'{method.upper()} after Compensation Test Accuracy (%)', fontsize=20)
    #ax.set_title(f'{name[0].replace("_", " ")} vs {name[1].replace("_", " ")} Scatter Plot\n{title, method.upper()}', fontsize=20, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(min_acc - 2, max_acc + 2)
    ax.set_ylim(min_acc - 2, max_acc + 2)
    
    # Set tick label size
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    plt.show()
    
    # Print detailed statistics
    print(f"\nDetailed Statistics:")
    print(f"Pruned Accuracy Range: {data[name[0]].min():.2f}% - {data[name[0]].max():.2f}%")
    print(f"Compensated Accuracy Range: {data[name[1]].min():.2f}% - {data[name[1]].max():.2f}%")
    print(f"Average Improvement: {avg_improvement:.2f}%")
    print(f"Correlation: {correlation:.3f}")
    print(f"Total Data Points: {len(data)}")
    print(f"Number of Experiments: {data['experiment_name'].nunique()}")


def plot_by_parameter_groups(data, parameter, save_dir=None, name=['compressed_accuracy', 'compressed_compensated_accuracy']):
    """
    Plot scatter plots grouped by a specified parameter (e.g., learning rate).

    Args:
        data (pd.DataFrame): Experiment data.
        parameter (str): The column name to group by.
        save_dir (str, optional): Directory to save the plot.
        name (list): List of two column names to plot [x_axis_col, y_axis_col].
    """
    if parameter not in data.columns:
        print(f"Parameter '{parameter}' not found in data")
        return
    
    # Filter data for ViT models
    if 'vit' in str(data.get('model', '')).lower():
        filtered_data = data[data['original_accuracy'] >= 75]
        print(f"ViT model detected in grouped plot: filtered from {len(data)} to {len(filtered_data)} data points")
    else:
        filtered_data = data
    
    unique_values = filtered_data[parameter].unique()
    
    # Create subplots
    fig, axes = plt.subplots(1, len(unique_values), figsize=(5*len(unique_values), 5))
    if len(unique_values) == 1:
        axes = [axes]
    
    for i, value in enumerate(unique_values):
        subset = filtered_data[filtered_data[parameter] == value]
        subset = subset.dropna(subset=name)
        
        if len(subset) == 0:
            continue
            
        scatter = axes[i].scatter(
            subset[name[0]],
            subset[name[1]],
            c=subset['compression_ratio'],
            cmap='viridis',
            alpha=0.7,
            s=40
        )
        
        # Add diagonal line
        min_acc = min(subset[name[0]].min(), subset[name[1]].min())
        max_acc = max(subset[name[0]].max(), subset[name[1]].max())
        axes[i].plot([min_acc, max_acc], [min_acc, max_acc], 'r--', alpha=0.7)
        
        axes[i].set_xlabel('Compressed Accuracy (%)')
        axes[i].set_ylabel('Compensated Accuracy (%)')
        axes[i].set_title(f'{parameter} = {value}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f'{name[0]}_vs_{name[1]}_grouped_by_{parameter}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grouped image saved to: {save_path}")
    
    plt.show()


def plot_compression_ratio_hist(data, name, path):
    """
    Plot a boxplot of accuracy difference vs. compression ratio.

    Args:
        data (pd.DataFrame): Experiment data.
        name (list): List of two column names [baseline_col, compensated_col].
                     The difference is calculated as compensated - baseline.
        path (str): Path to save the plot.
    """
    data['diff'] = data[name[1]] - data[name[0]]

    boxplot_df = data[["compression_ratio", "diff"]].copy()
    boxplot_df = boxplot_df.dropna()
    
    if len(boxplot_df) == 0:
        print(f"Skipping boxplot: No valid data for {name}")
        return

    # Get sorted unique sparsities for color palette
    unique_sparsities = sorted(boxplot_df["compression_ratio"].unique())
    
    # Plotting
    fig = plt.figure(figsize=(6, 6))
    sns.boxplot(
        data=boxplot_df,
        x="compression_ratio",
        y="diff",
        hue="compression_ratio",
        palette="rainbow",
        dodge=False,
        legend=False
    )

    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Layer-wise Compression Ratio", fontsize=15)
    plt.ylabel(f"Δ Accuracy [%] (with compensation - w/o compensation)", fontsize=15)

    plt.xticks(rotation=45, ha="right", fontsize=15)
    plt.yticks(fontsize=15)

    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    fig.savefig(path, dpi=800)


def get_averaged_data_for_line_plot(experiment_dir):
    """
    Load and average data from multiple checkpoints for line plots.
    
    Filters checkpoints based on specific criteria (l1=0, wd=0, original_accuracy >= 90).

    Args:
        experiment_dir (str): Path to experiment logs directory.

    Returns:
        dict: Averaged results containing sparsity and various accuracy metrics.
    """
    # Find all experiment data files
    json_files = glob.glob(os.path.join(experiment_dir, "*", "experiment_data.json"))
    
    if not json_files:
        raise ValueError(f"No experiment data files found in {experiment_dir}")
    
    print(f"Found {len(json_files)} experiment data files")
    
    all_results = []
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                exp_data = json.load(f)
                
            # Check if this checkpoint meets the criteria
            dir_name = os.path.basename(os.path.dirname(json_file))
            
            # 1. Check if directory name contains l1=0 and wd=0
            if ('l1=0' not in dir_name or 'wd=0' not in dir_name) and "resnet18" in dir_name:
                # print(f"Skipping {dir_name}: does not meet l1=0 and wd=0 criteria")
                continue
                
            overall_summary = exp_data.get("overall_summary", [])
            if not overall_summary:
                continue
                
            first_entry = overall_summary[0]
            original_accuracy = first_entry.get("original_accuracy", 0)
            # 2. Check if original accuracy > 90 for ResNet18
            if original_accuracy < 90 and "resnet18" in dir_name:
                # print(f"Skipping {dir_name}: original accuracy {original_accuracy} < 90")
                continue

            # 2. Check if original accuracy > 75 for ViT    
            if original_accuracy < 83 and "vit" in dir_name:
                # print(f"Skipping {dir_name}: original accuracy {original_accuracy} < 75")
                continue
                
            print(f"Including {dir_name}: original_accuracy={original_accuracy}")
            
            # Extract data
            results = {
                "sparsity": [],
                "compressed": [],
                "compressed_compensated": [],
                "compressed_compensated_repaired": [],
                "compressed_repaired": [],
                "compressed_repaired_compensated_accuracy": [],
                "compressed_repaired_compensated_repaired_accuracy": []
            }
            
            for summary in exp_data.get("overall_summary", []):
                results["sparsity"].append(summary.get("compression_rate"))
                results["compressed"].append(summary.get("compressed_accuracy"))
                results["compressed_repaired"].append(summary.get("compressed_repaired_accuracy"))
                results["compressed_compensated"].append(summary.get("compressed_compensated_accuracy"))
                results["compressed_compensated_repaired"].append(summary.get("compressed_compensated_repaired_accuracy"))
                results["compressed_repaired_compensated_accuracy"].append(summary.get("compressed_repaired_compensated_accuracy"))
                results["compressed_repaired_compensated_repaired_accuracy"].append(summary.get("compressed_repaired_compensated_repaired_accuracy"))
            
            all_results.append(results)
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    if not all_results:
        raise ValueError("No valid checkpoints found that meet the criteria")
    
    print(f"Successfully loaded {len(all_results)} valid checkpoints")
    
    # Average the results
    averaged_results = {}
    keys = list(all_results[0].keys())
    
    for key in keys:
        if key == "sparsity":
            # Sparsity should be the same across all experiments
            averaged_results[key] = all_results[0][key]
        else:
            # Average across all checkpoints, filtering out None values
            all_values = [result[key] for result in all_results]
            
            # Transpose to iterate by index (compression level)
            # all_values is [exp1_list, exp2_list, ...]
            # We want to average exp1_list[i], exp2_list[i], ...
            
            num_points = len(all_values[0])
            averaged_metric = []
            
            for i in range(num_points):
                values_at_i = [exp[i] for exp in all_values if exp[i] is not None]
                if values_at_i:
                    averaged_metric.append(np.mean(values_at_i))
                else:
                    averaged_metric.append(None)
            
            averaged_results[key] = averaged_metric
    
    return averaged_results


def line_plot(experiment_dir):
    """
    Plot line plot using averaged data from multiple checkpoints.

    Args:
        experiment_dir (str): Path to experiment logs directory.
    """
    # Determine model type and repair method for labeling
    if "resnet" in experiment_dir:
        repair = "REPAIR"
    else:
        repair = "Retune_Layernorm"
    
    method = experiment_dir.split("/")[-1] 
    subpath = method + "_vis"
    
    # Get averaged results
    print(f"Loading and averaging data from {experiment_dir}")
    results = get_averaged_data_for_line_plot(experiment_dir)
    
    sparsity = results["sparsity"]
    compressed_accuracy = results["compressed"]
    compressed_repaired_accuracy = results["compressed_repaired"]
    compressed_compensated_repaired_accuracy = results["compressed_compensated_repaired"]
    
    # Create plot
    plt.figure(figsize=(7,4))

    plt.plot(sparsity, compressed_accuracy, 'o-', label=f'{method.replace("-","").upper()}')
    plt.plot(sparsity, compressed_repaired_accuracy, 'o-', label=f'{method.replace("-","").upper()}-{repair}')
    plt.plot(sparsity, compressed_compensated_repaired_accuracy, 'o-', label=f'{method.replace("-","").upper()}-Compensation-{repair}')

    plt.xlabel("Sparsity")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()
    
    # Save plot
    save_path = Path(experiment_dir).parent / subpath / "sparsity_accuracy_averaged.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def scatter_plot(model):
    """
    Main function to execute the complete data loading and visualization pipeline for scatter plots.
    
    Iterates through defined methods and experiment pairs to generate scatter plots,
    grouped plots, and histograms.
    """
    # Configuration
    methods = ["mag-l1", "mag-l2", "wanda", "fold"]
    base_path = "/scratch/fs201038/wt93384/projects/compression_compensation/experiment_logs"
    
    for method in methods:
        # Define pairs of metrics to compare
        metric_pairs = [
            ['compressed_repaired_accuracy', 'compressed_compensated_repaired_accuracy'],
            ['compressed_accuracy', 'compressed_compensated_accuracy' ],
            ['compressed_repaired_accuracy', 'compressed_repaired_compensated_repaired_accuracy'],
            ['compressed_repaired_accuracy', 'compressed_repaired_compensated_accuracy'],
            ['compressed_accuracy', 'compressed_compensated_repaired_accuracy'],
            ['compressed_accuracy', 'compressed_repaired_accuracy'],
            ['compressed_repaired_accuracy', 'compressed_compensated_repaired_accuracy'],
            ['compressed_compensated_accuracy', 'compressed_repaired_accuracy']
        ]
        
        experiment_dir = os.path.join(base_path, model, method)
        vis_dir = os.path.join(base_path, model, f"{method}_vis")
        os.makedirs(vis_dir, exist_ok=True)
        
        for name in metric_pairs:
            try:
                # Load all experiment data
                print(f"\nProcessing {method}: {name[0]} vs {name[1]}")
                data = load_all_experiment_data(experiment_dir)
                
                # Plot main scatter plot
                print("Generating scatter plot...")
                plot_compressed_vs_compensated_scatter(
                    data, model, method,  
                    save_path=os.path.join(vis_dir, f"{name[0]}_vs_{name[1]}_scatter.png"),
                    name=name
                )
                
                # Optional: Plot by parameter groups
                if 'learning_rate' in data.columns:
                    print("Generating grouped scatter plots...")
                    plot_by_parameter_groups(
                        data, 
                        'learning_rate',
                        save_dir=vis_dir,
                        name=name
                    )
                
                # Plot compression rate vs accuracy difference
                print("Generating compression rate vs accuracy difference plots...")
                plot_compression_ratio_hist(
                    data, 
                    name, 
                    path=os.path.join(vis_dir, f"compression_ratio_hist_{name[1]}_vs_{name[0]}.png")
                )
                
                print("Visualization completed for this pair!")
                
            except Exception as e:
                print(f"Error during execution for {method} - {name}: {e}")
                # import traceback
                # traceback.print_exc()

def plot_all_methods_comparison(base_dir, model_name, methods=["wanda", "fold", "mag-l1", "mag-l2"]):
    """
    Plot comparison of different methods on the same graph.
    Plots both Compressed Accuracy (dashed) and Compressed+Compensated+Repaired Accuracy (solid)
    for each method.

    Args:
        base_dir (str): Base directory containing experiment logs (e.g., .../experiment_logs).
        model_name (str): Model directory name (e.g., 'resnet18_sgd' or 'vit').
        methods (list): List of method names to compare.
    """

    plt.figure(figsize=(10, 6))
    if "resnet18" in model_name:
        title = "ResNet18 SGD"
    elif "vit" in model_name:
        title = "ViT"
    
    if "resnet" in model_name:
        repair = "REPAIR"
    else:
        repair = "Retune_Layernorm"
    
    # Define colors for states (Red, Blue, Green)
    state_colors = {
        'compressed': '#d62728',  # Red
        'repaired': '#1f77b4',    # Blue
        'compensated': '#2ca02c'  # Green
    }
    
    # Define markers for methods
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, method in enumerate(methods):
        experiment_dir = os.path.join(base_dir, model_name, method)
        marker = markers[idx % len(markers)]
        
        if not os.path.exists(experiment_dir):
            print(f"Warning: Directory not found for {method}: {experiment_dir}")
            continue
            
        try:
            print(f"Loading data for {method}...")
            results = get_averaged_data_for_line_plot(experiment_dir)
            
            sparsity = results["sparsity"]
            compressed_acc = results["compressed"]
            compressed_compensated_acc = results["compressed_compensated"]
            #compressed_compensated_repaired = results["compressed_compensated_repaired"] 
            #compressed_repaired = results["compressed_repaired"]
            
            # Plot Compressed Accuracy (Red, Dotted)
            plt.plot(sparsity, compressed_acc, linestyle=':', marker=marker, markersize=6, 
                     color=state_colors['compressed'], alpha=0.6, 
                     label=f'{method.upper()} (w/o Compensation)')
            
            # Plot Repaired Accuracy (Blue, Dashed)
            # plt.plot(sparsity, compressed_compensated_acc, linestyle='--', marker=marker, markersize=6, 
            #          color=state_colors['repaired'], alpha=0.7, 
            #          label=f'{method.upper()}+{repair}')

            # Plot Compensated+Repaired Accuracy (Green, Solid)
            plt.plot(sparsity, compressed_compensated_acc, linestyle='-', marker=marker, markersize=8, 
                     color=state_colors['compensated'], linewidth=2, 
                     label=f'{method.upper()} (with Compensation)')
            
        except Exception as e:
            print(f"Error processing {method}: {e}")
            continue

    plt.xlabel("Sparsity", fontsize=15)
    plt.ylabel("Test Accuracy (%)", fontsize=15)
    #plt.title(f"Comparison of Compression Methods ({title})", fontsize=14)
    plt.ylim(0, 100)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left')
    plt.tight_layout()
    
    # Save plot
    save_dir = os.path.join(base_dir, model_name, "comparison_vis")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "methods_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.show()


def plot_for_llms():
    import matplotlib.pyplot as plt
    import numpy as np

    methods = ["SlimGPT", "Wanda_sp", "Wanda++_sp", "FLAP"]
    before = np.array([148.02, 19.10, 17.50, 8.73])
    after = np.array([16.37, 13.91, 15.90, 8.37])
    base = np.array([5.47, 5.47, 5.47, 5.47])

    improve = (after - before) / before * 100

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8,4))
    rects0 = ax.bar(x - width, base, width, label='Dense', hatch='\\')
    rects1 = ax.bar(x , before, width, label='w/o Compensation', hatch='//')
    rects2 = ax.bar(x + width, after, width, label='w/ Compensation', hatch='..')

    ax.set_ylabel('Perplexity')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    # annotate improvement percentage
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i, max(b, a) * 1.05, f"{improve[i]:.0f}%", 
                ha='center', va='bottom', fontsize=10)

    #ax.set_title('Perplexity Improvement @ 30% Sparsity (LLaMA-2-7B)')
    ax.set_ylim(0, max(before) * 1.3)

    plt.tight_layout()
    plt.show()
    plt.savefig("/scratch/fs201038/wt93384/projects/compression_compensation/experiment_logs/mix/llm_comparison.png", dpi=800, bbox_inches='tight')

def plot_methods_comparison_from_data(data, save_dir=None, methods=None):
    """
    Plot comparison of different methods using loaded DataFrame.
    
    Args:
        data (DataFrame): DataFrame containing experiment results with columns:
                         method, compression_ratio, compressed_accuracy, 
                         compressed_compensated_accuracy, compressed_repaired_accuracy, etc.
        save_dir (str): Directory to save the plot
        methods (list): List of methods to plot. If None, uses all unique methods in data.
    """
    # Ensure data is DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    if methods is None:
        methods = data['method'].unique()
    
    plt.figure(figsize=(12, 8))
    
    # Define colors for states
    state_colors = {
        'compressed': '#d62728',  # Red
        'repaired': '#1f77b4',    # Blue  
        'compensated': '#2ca02c'  # Green
    }
    
    # Define markers for methods
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, method in enumerate(methods):
        method_data = data[data['method'] == method]
        
        if len(method_data) == 0:
            print(f"Warning: No data found for method {method}")
            continue
        
        # Group by compression_ratio and calculate mean
        agg_dict = {
            'compressed_accuracy': 'mean',
            'compressed_compensated_accuracy': 'mean'
        }
        if 'compressed_repaired_accuracy' in method_data.columns:
            agg_dict['compressed_repaired_accuracy'] = 'mean'
        if 'compressed_compensated_repaired_accuracy' in method_data.columns:
            agg_dict['compressed_compensated_repaired_accuracy'] = 'mean'
            
        grouped = method_data.groupby('compression_ratio', as_index=False).agg(agg_dict)
        grouped = grouped.sort_values('compression_ratio')
        
        # Apply ratio mapping
        ratio_mapping = {
            0.0: 0.0,
            0.1: 0.037425,
            0.2: 0.104620,
            0.3: 0.185302,
            0.4: 0.260517,
            0.5: 0.316897,
            0.6: 0.350799,
            0.7: 0.366595,
            0.8: 0.372063,
            0.9: 0.373156
        }
        grouped['mapped_ratio'] = grouped['compression_ratio'].apply(
            lambda x: ratio_mapping.get(round(x, 1), x)
        )
        
        marker = markers[idx % len(markers)]
        
        # Plot Compressed Accuracy (Red, Dotted)
        plt.plot(grouped['mapped_ratio'], grouped['compressed_accuracy'], 
                 linestyle=':', marker=marker, markersize=6,
                 color=state_colors['compressed'], alpha=0.6,
                 label=f'{method.upper()}')
        
        # Plot Compensated Accuracy (Green, Solid)
        plt.plot(grouped['mapped_ratio'], grouped['compressed_compensated_accuracy'],
                 linestyle='-', marker=marker, markersize=8,
                 color=state_colors['compensated'], linewidth=2,
                 label=f'{method.upper()}+COMPENSATION')
        
        # Plot Repaired if available
        if 'compressed_repaired_accuracy' in agg_dict:
            plt.plot(grouped['mapped_ratio'], grouped['compressed_repaired_accuracy'],
                     linestyle='--', marker=marker, markersize=6,
                     color=state_colors['repaired'], alpha=0.7,
                     label=f'{method.upper()}+RETUNE')
    
    plt.xlabel("Sparsity", fontsize=15)
    plt.ylabel("Test Accuracy (%)", fontsize=15)
    #plt.title("CLIP ViT-B/32: Comparison of Compression Methods", fontsize=16)
    plt.ylim(0, 100)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "clip_methods_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CLIP methods comparison plot saved to: {save_path}")
    
    plt.show()


def load_data_from_log(log_path, project_root="/scratch/fs201038/wt93384/projects/compression_compensation"):
    """
    Parse log file to find saved experiment data and load it into a DataFrame.
    """
    print(f"Parsing log file: {log_path}")
    csv_paths = set()
    
    with open(log_path, 'r') as f:
        for line in f:
            if "Summary saved to:" in line:
                # Extract path
                path = line.split("Summary saved to:")[1].strip()
                if not os.path.isabs(path):
                    path = os.path.join(project_root, path)
                csv_paths.add(path)
    
    if not csv_paths:
        print("No experiment data found in log file.")
        return None
        
    print(f"Found {len(csv_paths)} unique experiment data files.")
    
    all_data = []
    for csv_file in csv_paths:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                
                # Extract experiment_name from directory name
                exp_dir = os.path.dirname(csv_file)
                exp_name = os.path.basename(exp_dir)
                df['experiment_name'] = exp_name
                
                # Extract method from parent directory of experiment directory
                # Path structure: .../experiment_logs/vit/{method}/{experiment_name}/...
                method = os.path.basename(os.path.dirname(exp_dir))
                df['method'] = method
                
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        else:
            print(f"File not found: {csv_file}")
            
    if not all_data:
        return None
        
    # combined_data = pd.concat(all_data, ignore_index=True)
    # print(f"Loaded total {len(combined_data)} rows of data.")
    # return combined_data
    
    # Return list of dicts to avoid pandas issues during concatenation
    combined_list = []
    for df in all_data:
        combined_list.extend(df.to_dict('records'))
        
    print(f"Loaded total {len(combined_list)} rows of data (as list).")
    return combined_list


def plot_accuracy_vs_ratio(data, save_dir=None):
    """
    Plot Accuracy vs Compression Ratio for different methods.
    """
    # Ensure data is DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
        
    plt.figure(figsize=(7, 5))
    
    # Group by method and compression_ratio
    # We want to plot the mean accuracy for each ratio per method
    
    if 'method' not in data.columns:
        print("Error: 'method' column not found in data.")
        return

    methods = data['method'].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, method in enumerate(methods):
        method_data = data[data['method'] == method]
        
        # Group by ratio and calculate mean
        agg_dict = {
            'compressed_accuracy': 'mean',
            'compressed_compensated_accuracy': 'mean'
        }
        if 'compressed_repaired_accuracy' in method_data.columns:
            agg_dict['compressed_repaired_accuracy'] = 'mean'
            
        print(f"Aggregating for method {method}. Columns: {method_data.columns.tolist()}")
        print(f"Data types: {method_data.dtypes}")
        
        try:
            grouped = method_data.groupby('compression_ratio', as_index=False).agg(agg_dict)
        except Exception as e:
            print(f"Error during groupby for {method}: {e}")
            continue
        
        
        # Apply mapping to compression_ratio
        ratio_mapping = {
            0.0: 0.0,
            0.1: 0.037425,
            0.2: 0.104620,
            0.3: 0.185302,
            0.4: 0.260517,
            0.5: 0.316897,
            0.6: 0.350799,
            0.7: 0.366595,
            0.8: 0.372063,
            0.9: 0.373156
        }
        
        # Map the ratio. Use round(x, 1) to handle floating point inaccuracies
        grouped['mapped_ratio'] = grouped['compression_ratio'].apply(lambda x: ratio_mapping.get(round(x, 1), x))
        
        marker = markers[idx % len(markers)]
        
        # Plot Compressed
        plt.plot(grouped['mapped_ratio'], grouped['compressed_accuracy'], 
                 linestyle=':', marker=marker, label=f'{method.upper()} (w/o Compensation)')
                 
        # Plot Compensated
        plt.plot(grouped['mapped_ratio'], grouped['compressed_compensated_accuracy'], 
                 linestyle='-', marker=marker, label=f'{method.upper()} (with Compensation)')
                 
    plt.xlabel("Sparsity", fontsize=15)
    plt.ylabel("Accuracy (%)", fontsize=15)
    #plt.title("Performance Comparison: Accuracy vs Sparsity", fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "performance_comparison_curve.png")
        plt.savefig(save_path, dpi=800)
        print(f"Performance curve saved to: {save_path}")
        
    plt.show()




if __name__ == "__main__":
    plot_for_llms()
    
