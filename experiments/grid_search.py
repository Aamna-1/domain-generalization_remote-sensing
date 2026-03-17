"""
Augmentation Grid Search Script.

Systematically evaluates different combinations of photometric (αp)
and geometric (αg) augmentation intensities for domain generalization.

Usage:
    python experiments/grid_search.py --method sora --config L3 --dataset isprs --scenario V2P
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import run_experiment, set_seed


# Default grid values
ALPHA_P_VALUES = [0.0, 0.2, 0.5, 0.8]
ALPHA_G_VALUES = [0.0, 15.0, 45.0, 90.0]


def run_grid_search(
    method: str,
    config: str,
    dataset: str,
    scenario: str,
    seeds: list,
    alpha_p_values: list = ALPHA_P_VALUES,
    alpha_g_values: list = ALPHA_G_VALUES,
    epochs: int = 10,
    data_root: str = "./data",
    output_dir: str = "./outputs"
):
    """
    Run augmentation grid search.
    
    Args:
        method: 'sora' or 'reins'
        config: Configuration name
        dataset: 'isprs' or 'loveda'
        scenario: DG scenario
        seeds: List of random seeds
        alpha_p_values: List of photometric intensities
        alpha_g_values: List of geometric intensities
        epochs: Training epochs
        data_root: Dataset root directory
        output_dir: Output directory
        
    Returns:
        Dictionary with grid results
    """
    print("=" * 70)
    print("AUGMENTATION GRID SEARCH")
    print("=" * 70)
    print(f"Method:   {method.upper()} ({config})")
    print(f"Dataset:  {dataset.upper()} ({scenario})")
    print(f"Seeds:    {seeds}")
    print(f"Grid:     αp={alpha_p_values} × αg={alpha_g_values}")
    print(f"Total:    {len(alpha_p_values) * len(alpha_g_values)} configurations")
    print("=" * 70)
    
    grid_results = {}
    
    for alpha_p in alpha_p_values:
        for alpha_g in alpha_g_values:
            key = (alpha_p, alpha_g)
            print(f"\n{'#'*70}")
            print(f"# GRID CELL: αp={alpha_p}, αg={alpha_g}")
            print(f"{'#'*70}")
            
            result = run_experiment(
                method=method,
                config=config,
                dataset=dataset,
                scenario=scenario,
                seeds=seeds,
                alpha_p=alpha_p,
                alpha_g=alpha_g,
                epochs=epochs,
                data_root=data_root
            )
            
            grid_results[key] = {
                "alpha_p": alpha_p,
                "alpha_g": alpha_g,
                "test_miou_mean": result["test_miou_mean"],
                "test_miou_std": result["test_miou_std"],
                "val_miou_mean": result["val_miou_mean"],
                "val_miou_std": result["val_miou_std"],
                "test_oa_mean": result["test_oa_mean"],
                "test_oa_std": result["test_oa_std"],
                "full_result": result
            }
            
            print(f"\n→ αp={alpha_p}, αg={alpha_g}: "
                  f"Test mIoU = {result['test_miou_mean']:.4f} ± {result['test_miou_std']:.4f}")
    
    # Find best configuration
    best_key = max(grid_results.keys(), key=lambda k: grid_results[k]["test_miou_mean"])
    best_result = grid_results[best_key]
    
    print("\n" + "=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)
    print(f"\n{'αp':>6}  {'αg(°)':>7}  {'Val mIoU':>14}  {'Test mIoU':>14}")
    print("-" * 52)
    
    for (ap, ag), res in sorted(grid_results.items()):
        marker = " ← BEST" if (ap, ag) == best_key else ""
        print(f"{ap:>6.1f}  {ag:>7.1f}  "
              f"{res['val_miou_mean']:>7.4f}±{res['val_miou_std']:.4f}  "
              f"{res['test_miou_mean']:>7.4f}±{res['test_miou_std']:.4f}{marker}")
    
    print(f"\nBest: αp={best_key[0]}, αg={best_key[1]}° → "
          f"Test mIoU = {best_result['test_miou_mean']:.4f}")
    
    return {
        "method": method,
        "config": config,
        "dataset": dataset,
        "scenario": scenario,
        "seeds": seeds,
        "alpha_p_values": alpha_p_values,
        "alpha_g_values": alpha_g_values,
        "grid_results": grid_results,
        "best_key": best_key,
        "best_result": best_result
    }


def visualize_grid(
    grid_results: dict,
    alpha_p_values: list,
    alpha_g_values: list,
    output_path: str = None,
    title: str = "Grid Search Results"
):
    """
    Visualize grid search results as a heatmap.
    
    Args:
        grid_results: Dictionary with grid results
        alpha_p_values: List of αp values
        alpha_g_values: List of αg values
        output_path: Path to save figure (optional)
        title: Plot title
    """
    # Build matrices
    mean = np.zeros((len(alpha_p_values), len(alpha_g_values)))
    std = np.zeros((len(alpha_p_values), len(alpha_g_values)))
    
    for i, ap in enumerate(alpha_p_values):
        for j, ag in enumerate(alpha_g_values):
            res = grid_results[(ap, ag)]
            mean[i, j] = res["test_miou_mean"] * 100
            std[i, j] = res["test_miou_std"] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mean, cmap='Oranges', aspect='auto')
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('mIoU (%)', fontsize=12)
    
    # Labels
    ax.set_xticks(np.arange(len(alpha_g_values)))
    ax.set_yticks(np.arange(len(alpha_p_values)))
    ax.set_xticklabels([f"{int(ag)}°" for ag in alpha_g_values])
    ax.set_yticklabels([f"{ap:.1f}" for ap in alpha_p_values])
    
    # Annotations
    for i in range(len(alpha_p_values)):
        for j in range(len(alpha_g_values)):
            text = f'{mean[i, j]:.2f}\n±{std[i, j]:.2f}'
            ax.text(j, i, text, ha='center', va='center', 
                   color='black', fontsize=10)
    
    ax.set_xlabel(r'$\alpha_g$ (degrees)', fontsize=12)
    ax.set_ylabel(r'$\alpha_p$', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Augmentation Grid Search")
    parser.add_argument("--method", type=str, default="sora", choices=["sora", "reins"])
    parser.add_argument("--config", type=str, default="L3")
    parser.add_argument("--dataset", type=str, default="isprs", choices=["isprs", "loveda"])
    parser.add_argument("--scenario", type=str, default="V2P")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--alpha_p", type=float, nargs="+", default=ALPHA_P_VALUES)
    parser.add_argument("--alpha_g", type=float, nargs="+", default=ALPHA_G_VALUES)
    
    args = parser.parse_args()
    
    # Run grid search
    results = run_grid_search(
        method=args.method,
        config=args.config,
        dataset=args.dataset,
        scenario=args.scenario,
        seeds=args.seeds,
        alpha_p_values=args.alpha_p,
        alpha_g_values=args.alpha_g,
        epochs=args.epochs,
        data_root=args.data_root,
        output_dir=args.output_dir
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as pickle
    output_file = os.path.join(
        args.output_dir,
        f"grid_{args.method}_{args.config}_{args.dataset}_{args.scenario}_{timestamp}.pt"
    )
    torch.save(results, output_file)
    print(f"\nResults saved to: {output_file}")
    
    # Create visualization
    fig_path = os.path.join(
        args.output_dir,
        f"grid_{args.method}_{args.config}_{args.dataset}_{args.scenario}_{timestamp}.png"
    )
    visualize_grid(
        results["grid_results"],
        args.alpha_p,
        args.alpha_g,
        output_path=fig_path,
        title=f"{args.method.upper()} {args.config} - {args.dataset.upper()} {args.scenario}"
    )


if __name__ == "__main__":
    main()
