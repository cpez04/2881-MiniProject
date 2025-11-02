"""
Cross-model comparison and analysis.
Compares firewall effectiveness across different LLMs.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import argparse


def load_all_results(results_dir: str) -> Dict[str, Dict]:
    """Load results for all models."""
    results_path = Path(results_dir)
    all_results = {}

    for pkl_file in results_path.glob("*_firewall_results.pkl"):
        model_name = pkl_file.stem.replace("_firewall_results", "")
        with open(pkl_file, 'rb') as f:
            all_results[model_name] = pickle.load(f)
        print(f"Loaded results for {model_name}")

    return all_results


def plot_cross_model_comparison(all_results: Dict[str, Dict], output_dir: str):
    """Plot safety-utility frontiers for all models on one graph."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Color palette for models
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    # Create one plot per detector showing all models
    detectors = ['ROUGE-L', 'MinHash-Jaccard', 'TF-IDF-Cosine']

    for detector in detectors:
        fig, ax = plt.subplots(figsize=(10, 7))

        for (model_name, results), color in zip(all_results.items(), model_colors):
            if detector not in results:
                continue

            df = pd.DataFrame(results[detector])

            # Plot with confidence intervals
            ax.plot(df['false_positive_rate'], df['blocked_leakage'],
                   marker='o', label=model_name, color=color, linewidth=2, markersize=5)

            # Add shaded CI region
            fpr_ci_lower = [ci[0] for ci in df['false_positive_rate_ci']]
            fpr_ci_upper = [ci[1] for ci in df['false_positive_rate_ci']]
            bl_ci_lower = [ci[0] for ci in df['blocked_leakage_ci']]
            bl_ci_upper = [ci[1] for ci in df['blocked_leakage_ci']]

            ax.fill_betweenx(df['blocked_leakage'], fpr_ci_lower, fpr_ci_upper,
                            alpha=0.1, color=color)
            ax.fill_between(df['false_positive_rate'], bl_ci_lower, bl_ci_upper,
                           alpha=0.1, color=color)

        # Target lines
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Target BL ≥ 70%')
        ax.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Target FPR ≤ 10%')

        # Target box
        ax.fill_between([0, 0.1], 0.7, 1.0, alpha=0.1, color='green', label='Ideal Region')

        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
        ax.set_ylabel('Blocked Leakage (BL)', fontsize=12)
        ax.set_title(f'{detector}: Cross-Model Safety-Utility Frontier', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_path / f"cross_model_{detector.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved cross-model comparison for {detector}")


def plot_leakage_rates(all_results: Dict[str, Dict], output_dir: str):
    """Compare baseline leakage rates across models."""
    output_path = Path(output_dir)

    # Extract leakage rates from each model's results
    leakage_data = []

    for model_name, results in all_results.items():
        # Get leakage rate from any detector (should be same across all)
        detector_name = list(results.keys())[0]
        df = pd.DataFrame(results[detector_name])
        leakage_rate = df['leakage_rate'].iloc[0]  # Same for all thresholds

        leakage_data.append({
            'Model': model_name,
            'Leakage Rate': leakage_rate
        })

    leakage_df = pd.DataFrame(leakage_data).sort_values('Leakage Rate', ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(leakage_df['Model'], leakage_df['Leakage Rate'], color='steelblue')

    # Add percentage labels
    for i, (idx, row) in enumerate(leakage_df.iterrows()):
        ax.text(row['Leakage Rate'] + 0.01, i, f"{row['Leakage Rate']:.1%}",
               va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Leakage Rate (% of outputs with ≥20 token overlap)', fontsize=12)
    ax.set_title('Data Leakage Rates Across Models', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(leakage_df['Leakage Rate']) * 1.15)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "cross_model_leakage_rates.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved leakage rate comparison")


def generate_summary_table(all_results: Dict[str, Dict], output_dir: str):
    """Generate summary table comparing best operating points across models."""
    output_path = Path(output_dir)

    summary_rows = []

    for model_name, results in all_results.items():
        for detector_name, detector_results in results.items():
            df = pd.DataFrame(detector_results)

            # Find best balanced point
            df['balance'] = df['blocked_leakage'] - df['false_positive_rate']
            best_idx = df['balance'].idxmax()
            best = df.loc[best_idx]

            # Find best operating point (BL >= 0.7, FPR <= 0.1)
            valid = (df['blocked_leakage'] >= 0.7) & (df['false_positive_rate'] <= 0.1)

            if valid.any():
                valid_df = df[valid].copy()
                valid_df['score'] = valid_df['blocked_leakage'] - valid_df['false_positive_rate']
                op_idx = valid_df['score'].idxmax()
                op = valid_df.loc[op_idx]
                op_bl = f"{op['blocked_leakage']:.1%}"
                op_fpr = f"{op['false_positive_rate']:.1%}"
                op_threshold = f"{op['threshold']:.2f}"
            else:
                op_bl = "N/A"
                op_fpr = "N/A"
                op_threshold = "N/A"

            summary_rows.append({
                'Model': model_name,
                'Detector': detector_name,
                'Best_Threshold': f"{best['threshold']:.2f}",
                'Best_BL': f"{best['blocked_leakage']:.1%}",
                'Best_FPR': f"{best['false_positive_rate']:.1%}",
                'Best_TR': f"{best['trigger_rate']:.1%}",
                'Op_Threshold': op_threshold,
                'Op_BL': op_bl,
                'Op_FPR': op_fpr,
                'Leakage_Rate': f"{best['leakage_rate']:.1%}"
            })

    summary_df = pd.DataFrame(summary_rows)

    # Save CSV
    summary_df.to_csv(output_path / "cross_model_summary.csv", index=False)
    print("Saved summary table to cross_model_summary.csv")

    # Print formatted table
    print("\n" + "="*120)
    print("CROSS-MODEL SUMMARY")
    print("="*120)
    print("\nBest Balanced Operating Points (max BL - FPR):")
    print("-"*120)
    print(summary_df[['Model', 'Detector', 'Best_Threshold', 'Best_BL', 'Best_FPR', 'Best_TR']].to_string(index=False))

    print("\n\nTarget Operating Points (BL ≥ 70%, FPR ≤ 10%):")
    print("-"*120)
    print(summary_df[['Model', 'Detector', 'Op_Threshold', 'Op_BL', 'Op_FPR']].to_string(index=False))

    print("\n\nLeakage Rates by Model:")
    print("-"*120)
    leakage_by_model = summary_df.groupby('Model')['Leakage_Rate'].first().reset_index()
    print(leakage_by_model.to_string(index=False))
    print("="*120)

    return summary_df


def plot_detector_comparison_across_models(all_results: Dict[str, Dict], output_dir: str):
    """Compare detector performance across models."""
    output_path = Path(output_dir)

    # For each model, find the best detector
    best_detectors = []

    for model_name, results in all_results.items():
        best_score = -np.inf
        best_detector = None
        best_metrics = None

        for detector_name, detector_results in results.items():
            df = pd.DataFrame(detector_results)
            df['balance'] = df['blocked_leakage'] - df['false_positive_rate']
            max_balance = df['balance'].max()

            if max_balance > best_score:
                best_score = max_balance
                best_detector = detector_name
                best_idx = df['balance'].idxmax()
                best_metrics = df.loc[best_idx]

        best_detectors.append({
            'Model': model_name,
            'Best_Detector': best_detector,
            'BL': best_metrics['blocked_leakage'],
            'FPR': best_metrics['false_positive_rate'],
            'Balance': best_score
        })

    bd_df = pd.DataFrame(best_detectors)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Best detector per model
    detector_counts = bd_df['Best_Detector'].value_counts()
    axes[0].bar(range(len(detector_counts)), detector_counts.values, color='steelblue')
    axes[0].set_xticks(range(len(detector_counts)))
    axes[0].set_xticklabels(detector_counts.index, rotation=15, ha='right')
    axes[0].set_ylabel('Number of Models', fontsize=12)
    axes[0].set_title('Best Detector Across Models', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Subplot 2: Balance scores
    x = np.arange(len(bd_df))
    axes[1].barh(x, bd_df['Balance'], color='coral')
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(bd_df['Model'], fontsize=9)
    axes[1].set_xlabel('Balance Score (BL - FPR)', fontsize=12)
    axes[1].set_title('Best Balance Score per Model', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    # Add detector labels
    for i, (idx, row) in enumerate(bd_df.iterrows()):
        axes[1].text(row['Balance'] + 0.01, i, row['Best_Detector'],
                    va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path / "detector_comparison_across_models.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved detector comparison plot")


def main():
    parser = argparse.ArgumentParser(description="Cross-model firewall comparison")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory containing model results")
    parser.add_argument("--output-dir", type=str, default="figures",
                       help="Output directory for figures")

    args = parser.parse_args()

    print("Loading results from all models...")
    all_results = load_all_results(args.results_dir)

    if len(all_results) == 0:
        print("ERROR: No results found in", args.results_dir)
        return

    print(f"\nFound {len(all_results)} models: {list(all_results.keys())}")
    print()

    # Generate comparisons
    print("Generating cross-model visualizations...")
    plot_cross_model_comparison(all_results, args.output_dir)
    plot_leakage_rates(all_results, args.output_dir)
    plot_detector_comparison_across_models(all_results, args.output_dir)

    print("\nGenerating summary table...")
    summary_df = generate_summary_table(all_results, args.output_dir)

    print("\nCross-model comparison complete!")


if __name__ == "__main__":
    main()
