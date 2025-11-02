"""
Quick analysis and summary of firewall evaluation results.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_results(results_path):
    """Load pickled results."""
    with open(results_path, 'rb') as f:
        return pickle.load(f)


def print_summary_table(results):
    """Print formatted summary table."""
    print("\n" + "="*100)
    print("FIREWALL EVALUATION SUMMARY")
    print("="*100)
    print()

    for detector_name, detector_results in results.items():
        df = pd.DataFrame(detector_results)

        print(f"\n{detector_name}")
        print("-" * 100)

        # Key statistics
        max_bl_idx = df['blocked_leakage'].idxmax()
        max_bl = df.loc[max_bl_idx]

        min_fpr_idx = df['false_positive_rate'].idxmin()
        min_fpr = df.loc[min_fpr_idx]

        # Best balanced point (maximize BL - FPR)
        df['balance'] = df['blocked_leakage'] - df['false_positive_rate']
        best_idx = df['balance'].idxmax()
        best = df.loc[best_idx]

        print(f"  Best Blocked Leakage (BL):")
        print(f"    Threshold: {max_bl['threshold']:.2f}")
        print(f"    BL: {max_bl['blocked_leakage']:.1%} (CI: {max_bl['blocked_leakage_ci'][0]:.1%} - {max_bl['blocked_leakage_ci'][1]:.1%})")
        print(f"    FPR: {max_bl['false_positive_rate']:.1%}")
        print(f"    TR: {max_bl['trigger_rate']:.1%}")

        print(f"\n  Lowest False Positive Rate (FPR):")
        print(f"    Threshold: {min_fpr['threshold']:.2f}")
        print(f"    FPR: {min_fpr['false_positive_rate']:.1%}")
        print(f"    BL: {min_fpr['blocked_leakage']:.1%}")
        print(f"    TR: {min_fpr['trigger_rate']:.1%}")

        print(f"\n  Best Balanced Point (max BL - FPR):")
        print(f"    Threshold: {best['threshold']:.2f}")
        print(f"    BL: {best['blocked_leakage']:.1%}")
        print(f"    FPR: {best['false_positive_rate']:.1%}")
        print(f"    TR: {best['trigger_rate']:.1%}")
        print(f"    Balance: {best['balance']:.3f}")

    print("\n" + "="*100)


def find_operating_points(results, bl_threshold=0.7, fpr_threshold=0.1):
    """Find thresholds meeting specified criteria."""
    print(f"\nOPERATING POINTS (BL ≥ {bl_threshold:.0%}, FPR ≤ {fpr_threshold:.0%})")
    print("="*100)

    operating_points = []

    for detector_name, detector_results in results.items():
        df = pd.DataFrame(detector_results)

        # Find valid operating points
        valid = (df['blocked_leakage'] >= bl_threshold) & (df['false_positive_rate'] <= fpr_threshold)

        if valid.any():
            valid_df = df[valid].copy()
            # Pick best: maximize BL, minimize FPR
            valid_df['score'] = valid_df['blocked_leakage'] - valid_df['false_positive_rate']
            best_idx = valid_df['score'].idxmax()
            best = valid_df.loc[best_idx]

            operating_points.append({
                'Detector': detector_name,
                'Threshold': f"{best['threshold']:.2f}",
                'BL': f"{best['blocked_leakage']:.1%}",
                'BL_CI': f"({best['blocked_leakage_ci'][0]:.1%}, {best['blocked_leakage_ci'][1]:.1%})",
                'FPR': f"{best['false_positive_rate']:.1%}",
                'FPR_CI': f"({best['false_positive_rate_ci'][0]:.1%}, {best['false_positive_rate_ci'][1]:.1%})",
                'TR': f"{best['trigger_rate']:.1%}",
            })
        else:
            operating_points.append({
                'Detector': detector_name,
                'Threshold': 'N/A',
                'BL': 'N/A',
                'BL_CI': 'N/A',
                'FPR': 'N/A',
                'FPR_CI': 'N/A',
                'TR': 'N/A',
            })

    op_df = pd.DataFrame(operating_points)
    print(op_df.to_string(index=False))
    print("="*100)

    return op_df


def compare_detectors_at_threshold(results, thresholds):
    """Compare detectors at specific thresholds."""
    print(f"\nDETECTOR COMPARISON AT SPECIFIED THRESHOLDS")
    print("="*100)

    comparison = []

    for detector_name, threshold in thresholds.items():
        if detector_name not in results:
            continue

        detector_results = results[detector_name]
        df = pd.DataFrame(detector_results)

        # Find closest threshold
        closest_idx = (df['threshold'] - threshold).abs().idxmin()
        row = df.loc[closest_idx]

        comparison.append({
            'Detector': detector_name,
            'Threshold': f"{row['threshold']:.2f}",
            'BL': f"{row['blocked_leakage']:.1%}",
            'FPR': f"{row['false_positive_rate']:.1%}",
            'TR': f"{row['trigger_rate']:.1%}",
        })

    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    print("="*100)

    return comp_df


def export_latex_table(results, output_path):
    """Export results as LaTeX table."""
    # Create a summary table suitable for paper
    rows = []

    for detector_name, detector_results in results.items():
        df = pd.DataFrame(detector_results)

        # Get best balanced point
        df['balance'] = df['blocked_leakage'] - df['false_positive_rate']
        best_idx = df['balance'].idxmax()
        best = df.loc[best_idx]

        rows.append({
            'Detector': detector_name,
            'τ': f"{best['threshold']:.2f}",
            'BL': f"{best['blocked_leakage']*100:.1f}",
            'FPR': f"{best['false_positive_rate']*100:.1f}",
            'TR': f"{best['trigger_rate']*100:.1f}",
        })

    table_df = pd.DataFrame(rows)
    latex = table_df.to_latex(index=False, escape=False,
                              column_format='lcccc',
                              caption='Firewall Detector Performance',
                              label='tab:firewall')

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"\nLaTeX table exported to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze firewall evaluation results")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to pickled results file")
    parser.add_argument("--bl-threshold", type=float, default=0.7,
                       help="Minimum blocked leakage for operating points")
    parser.add_argument("--fpr-threshold", type=float, default=0.1,
                       help="Maximum FPR for operating points")
    parser.add_argument("--export-latex", type=str, default=None,
                       help="Export LaTeX table to file")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results}")
    results = load_results(args.results)

    # Print summary
    print_summary_table(results)

    # Find operating points
    op_df = find_operating_points(results, args.bl_threshold, args.fpr_threshold)

    # Compare at matched thresholds
    thresholds = {
        'ROUGE-L': 0.70,
        'MinHash-Jaccard': 0.70,
        'TF-IDF-Cosine': 0.60
    }
    comp_df = compare_detectors_at_threshold(results, thresholds)

    # Export LaTeX if requested
    if args.export_latex:
        export_latex_table(results, args.export_latex)


if __name__ == "__main__":
    main()
