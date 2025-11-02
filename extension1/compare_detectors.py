"""
Paired comparison of detectors using McNemar's test.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from metrics import mcnemar_test
from itertools import combinations


def load_detector_decisions(results_path: str, threshold_map: dict):
    """
    Load detector blocking decisions at specified thresholds.

    Args:
        results_path: Path to pickled results
        threshold_map: Dict mapping detector name to threshold

    Returns:
        Dict mapping detector name to boolean array of blocking decisions
    """
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    # Need to also load similarities
    # For now, we'll need to recompute or load from somewhere
    # This is a simplified version that works with the results structure

    decisions = {}
    for detector_name, threshold in threshold_map.items():
        detector_results = results[detector_name]
        df = pd.DataFrame(detector_results)

        # Find closest threshold
        thresh_idx = (df['threshold'] - threshold).abs().idxmin()
        # We need the actual blocking decisions, not just the metrics
        # This requires access to the original similarities

    return decisions


def compare_all_detectors(similarities_dict: dict, is_leaky: np.ndarray,
                          threshold_map: dict):
    """
    Compare all detector pairs using McNemar's test.

    Args:
        similarities_dict: Dict mapping detector name to similarity scores
        is_leaky: Boolean array of leakage labels
        threshold_map: Dict mapping detector name to threshold

    Returns:
        DataFrame of pairwise comparisons
    """
    detector_names = list(similarities_dict.keys())
    comparisons = []

    # Get blocking decisions for each detector
    decisions = {}
    for detector_name, threshold in threshold_map.items():
        sims = similarities_dict[detector_name]
        decisions[detector_name] = sims >= threshold

    # Pairwise comparisons
    for det1, det2 in combinations(detector_names, 2):
        result = mcnemar_test(
            decisions[det1],
            decisions[det2],
            is_leaky
        )

        # Compute metrics for both
        d1_bl = np.mean(decisions[det1][is_leaky])
        d2_bl = np.mean(decisions[det2][is_leaky])
        d1_fpr = np.mean(decisions[det1][~is_leaky])
        d2_fpr = np.mean(decisions[det2][~is_leaky])

        comparisons.append({
            'Detector_1': det1,
            'Detector_2': det2,
            'Threshold_1': threshold_map[det1],
            'Threshold_2': threshold_map[det2],
            'BL_1': d1_bl,
            'BL_2': d2_bl,
            'BL_Diff': d1_bl - d2_bl,
            'FPR_1': d1_fpr,
            'FPR_2': d2_fpr,
            'FPR_Diff': d1_fpr - d2_fpr,
            'McNemar_Stat': result['statistic'],
            'p_value': result['p_value'],
            'Significant': result['p_value'] < 0.05
        })

    return pd.DataFrame(comparisons)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare detectors pairwise")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to pickled results")
    parser.add_argument("--similarities", type=str, required=True,
                       help="Path to pickled similarities dict")
    parser.add_argument("--leakage", type=str, required=True,
                       help="Path to pickled leakage labels")
    parser.add_argument("--thresholds", type=str, default="0.7,0.7,0.6",
                       help="Comma-separated thresholds for ROUGE,Jaccard,Cosine")

    args = parser.parse_args()

    # Parse thresholds
    thresh_values = [float(x) for x in args.thresholds.split(',')]
    threshold_map = {
        'ROUGE-L': thresh_values[0],
        'MinHash-Jaccard': thresh_values[1],
        'TF-IDF-Cosine': thresh_values[2]
    }

    # Load data
    with open(args.similarities, 'rb') as f:
        similarities_dict = pickle.load(f)

    with open(args.leakage, 'rb') as f:
        is_leaky = pickle.load(f)

    # Compare
    comparisons = compare_all_detectors(similarities_dict, is_leaky, threshold_map)

    print("\n" + "="*80)
    print("PAIRWISE DETECTOR COMPARISONS (McNemar's Test)")
    print("="*80)
    print(comparisons.to_string(index=False))
    print("="*80)

    # Save
    output_path = Path(args.results).parent / "detector_comparisons.csv"
    comparisons.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
