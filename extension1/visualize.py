"""
Visualization for firewall evaluation results.
Generates safety-utility frontiers and comparison plots.
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List


sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


class FirewallVisualizer:
    """Visualizes firewall evaluation results."""

    def __init__(self, results_path: str):
        """
        Args:
            results_path: Path to pickled results
        """
        with open(results_path, 'rb') as f:
            self.results = pickle.load(f)

        self.detector_names = list(self.results.keys())

    def plot_safety_utility_frontier(self, save_path: str = None, figsize=(10, 6)):
        """
        Plot Blocked-Leakage (BL) vs False-Positive Rate (FPR) for all detectors.

        Args:
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = {'ROUGE-L': '#e74c3c', 'MinHash-Jaccard': '#3498db', 'TF-IDF-Cosine': '#2ecc71'}
        markers = {'ROUGE-L': 'o', 'MinHash-Jaccard': 's', 'TF-IDF-Cosine': '^'}

        for detector_name in self.detector_names:
            results = self.results[detector_name]
            df = pd.DataFrame(results)

            fpr = df['false_positive_rate'].values
            bl = df['blocked_leakage'].values
            thresholds = df['threshold'].values

            # Plot main line
            ax.plot(fpr, bl, label=detector_name,
                   color=colors.get(detector_name, 'gray'),
                   marker=markers.get(detector_name, 'o'),
                   markersize=6, linewidth=2, alpha=0.8)

            # Add confidence intervals
            if 'blocked_leakage_ci' in df.columns and 'false_positive_rate_ci' in df.columns:
                bl_ci = np.array([ci for ci in df['blocked_leakage_ci']])
                fpr_ci = np.array([ci for ci in df['false_positive_rate_ci']])

                # Shade CI region
                ax.fill_between(fpr, bl_ci[:, 0], bl_ci[:, 1],
                               alpha=0.15, color=colors.get(detector_name, 'gray'))

        # Highlight operating point region (BL >= 0.7, FPR <= 0.1)
        ax.axhline(0.7, color='red', linestyle='--', alpha=0.3, label='BL ≥ 0.7 target')
        ax.axvline(0.1, color='orange', linestyle='--', alpha=0.3, label='FPR ≤ 0.1 target')

        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Blocked Leakage (BL)', fontsize=12, fontweight='bold')
        ax.set_title('Safety-Utility Frontier: Blocked Leakage vs False Positives',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set axis limits
        ax.set_xlim(-0.02, max(1.0, ax.get_xlim()[1]))
        ax.set_ylim(-0.02, 1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        plt.show()

    def plot_trigger_rates(self, save_path: str = None, figsize=(10, 6)):
        """
        Plot trigger rates vs thresholds for all detectors.

        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = {'ROUGE-L': '#e74c3c', 'MinHash-Jaccard': '#3498db', 'TF-IDF-Cosine': '#2ecc71'}

        for detector_name in self.detector_names:
            results = self.results[detector_name]
            df = pd.DataFrame(results)

            thresholds = df['threshold'].values
            tr = df['trigger_rate'].values

            ax.plot(thresholds, tr, label=detector_name,
                   color=colors.get(detector_name, 'gray'),
                   marker='o', linewidth=2, alpha=0.8)

        ax.set_xlabel('Threshold (τ)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Trigger Rate (TR)', fontsize=12, fontweight='bold')
        ax.set_title('Trigger Rate vs Threshold', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        plt.show()

    def find_operating_points(self, bl_threshold: float = 0.7, fpr_threshold: float = 0.1) -> pd.DataFrame:
        """
        Find operating points that meet BL and FPR constraints.

        Args:
            bl_threshold: Minimum blocked leakage
            fpr_threshold: Maximum false positive rate

        Returns:
            DataFrame of operating points
        """
        operating_points = []

        for detector_name in self.detector_names:
            results = self.results[detector_name]
            df = pd.DataFrame(results)

            # Find points that meet criteria
            valid = (df['blocked_leakage'] >= bl_threshold) & (df['false_positive_rate'] <= fpr_threshold)

            if valid.any():
                # Get best point (highest BL, lowest FPR)
                valid_df = df[valid].copy()
                valid_df['score'] = valid_df['blocked_leakage'] - valid_df['false_positive_rate']
                best_idx = valid_df['score'].idxmax()
                best = valid_df.loc[best_idx]

                operating_points.append({
                    'Detector': detector_name,
                    'Threshold': best['threshold'],
                    'Blocked_Leakage': best['blocked_leakage'],
                    'FPR': best['false_positive_rate'],
                    'Trigger_Rate': best['trigger_rate']
                })
            else:
                operating_points.append({
                    'Detector': detector_name,
                    'Threshold': None,
                    'Blocked_Leakage': None,
                    'FPR': None,
                    'Trigger_Rate': None
                })

        return pd.DataFrame(operating_points)

    def print_summary(self):
        """Print summary statistics for all detectors."""
        print("\n" + "="*80)
        print("FIREWALL EVALUATION SUMMARY")
        print("="*80)

        for detector_name in self.detector_names:
            results = self.results[detector_name]
            df = pd.DataFrame(results)

            print(f"\n{detector_name}:")
            print(f"  Threshold range: {df['threshold'].min():.2f} - {df['threshold'].max():.2f}")
            print(f"  Max Blocked Leakage: {df['blocked_leakage'].max():.2%}")
            print(f"  Min FPR (at max BL): {df.loc[df['blocked_leakage'].idxmax(), 'false_positive_rate']:.2%}")

        print("\n" + "="*80)
        print("OPERATING POINTS (BL ≥ 0.7, FPR ≤ 0.1)")
        print("="*80)

        op_df = self.find_operating_points()
        print(op_df.to_string(index=False))

        print("="*80)


def main():
    """Main entry point for visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize firewall evaluation results")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to pickled results file")
    parser.add_argument("--output-dir", type=str, default="extension1/figures",
                       help="Output directory for figures")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizer
    viz = FirewallVisualizer(args.results)

    # Print summary
    viz.print_summary()

    # Generate plots
    model_name = Path(args.results).stem.replace('_firewall_results', '')

    viz.plot_safety_utility_frontier(
        save_path=output_dir / f"{model_name}_safety_utility_frontier.png"
    )

    viz.plot_trigger_rates(
        save_path=output_dir / f"{model_name}_trigger_rates.png"
    )


if __name__ == "__main__":
    main()
