"""
Main firewall evaluation pipeline.
Loads data, runs detectors, sweeps thresholds, and computes metrics.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import pickle

from ground_truth import LeakageLabeler
from detectors import ROUGELDetector, MinHashJaccardDetector, TFIDFCosineDetector
from metrics import compute_all_metrics, mcnemar_test


class FirewallEvaluator:
    """Main evaluation pipeline for firewall detectors."""

    def __init__(self, model_name: str = "Llama-2-7b-chat-hf",
                 tokenizer_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 min_overlap: int = 20):
        """
        Args:
            model_name: Model to evaluate (for loading results)
            tokenizer_name: Tokenizer for detectors
            min_overlap: Minimum token overlap for leakage labeling
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.min_overlap = min_overlap

        # Initialize components
        self.labeler = LeakageLabeler(tokenizer_name, min_overlap)
        self.rouge_detector = ROUGELDetector(tokenizer_name)
        self.minhash_detector = MinHashJaccardDetector(tokenizer_name)
        self.tfidf_detector = TFIDFCosineDetector(tokenizer_name)

        # Data
        self.queries = []
        self.outputs = []
        self.chunks_list = []
        self.is_leaky = None

        # Similarity scores
        self.rouge_scores = None
        self.jaccard_scores = None
        self.cosine_scores = None

    def load_data(self, results_path: str, retrieved_contexts_path: str):
        """
        Load query results and retrieved contexts.

        Args:
            results_path: Path to model results JSONL
            retrieved_contexts_path: Path to retrieved contexts JSONL
        """
        print(f"Loading data for {self.model_name}...")

        # Load results
        with open(results_path, 'r') as f:
            results = [json.loads(line) for line in f]

        # Load retrieved contexts
        with open(retrieved_contexts_path, 'r') as f:
            contexts = [json.loads(line) for line in f]

        # Create ID mapping for contexts
        context_dict = {item['query_id']: item for item in contexts}

        # Match results with contexts
        for result in results:
            query_id = result.get('query_id') or result.get('id')
            if query_id in context_dict:
                self.queries.append(result['query'])
                self.outputs.append(result['output'])

                # Get retrieved chunks
                context_item = context_dict[query_id]
                chunks = [context_item['retrieved_context']]  # May be single or list
                if isinstance(chunks[0], list):
                    chunks = chunks[0]
                self.chunks_list.append(chunks)

        print(f"Loaded {len(self.queries)} query-output pairs")

    def label_leakage(self):
        """Label outputs as leaky or not."""
        print(f"Labeling leakage (min_overlap={self.min_overlap})...")
        self.is_leaky = np.array(self.labeler.label_dataset(self.outputs, self.chunks_list))
        leakage_rate = np.mean(self.is_leaky)
        print(f"Leakage rate: {leakage_rate:.2%} ({np.sum(self.is_leaky)}/{len(self.is_leaky)} outputs)")

    def compute_similarities(self):
        """Compute similarity scores for all detectors."""
        print("Computing ROUGE-L similarities...")
        self.rouge_scores = np.array([
            self.rouge_detector.compute_max_similarity(output, chunks)
            for output, chunks in tqdm(zip(self.outputs, self.chunks_list), total=len(self.outputs))
        ])

        print("Computing MinHash-Jaccard similarities...")
        self.jaccard_scores = np.array([
            self.minhash_detector.compute_max_similarity(output, chunks)
            for output, chunks in tqdm(zip(self.outputs, self.chunks_list), total=len(self.outputs))
        ])

        print("Computing TF-IDF Cosine similarities...")
        # First fit the vectorizer
        all_texts = self.outputs + [chunk for chunks in self.chunks_list for chunk in chunks]
        self.tfidf_detector.fit(all_texts)

        self.cosine_scores = np.array([
            self.tfidf_detector.compute_max_similarity(output, chunks)
            for output, chunks in tqdm(zip(self.outputs, self.chunks_list), total=len(self.outputs))
        ])

        print("Similarities computed.")

    def sweep_thresholds(self, n_bootstrap: int = 1000) -> Dict:
        """
        Sweep thresholds for all detectors and compute metrics.

        Args:
            n_bootstrap: Number of bootstrap samples for CIs

        Returns:
            Dictionary of results for each detector
        """
        results = {}

        # Define threshold grids
        detectors = {
            'ROUGE-L': {
                'scores': self.rouge_scores,
                'thresholds': np.arange(0.50, 0.96, 0.05)
            },
            'MinHash-Jaccard': {
                'scores': self.jaccard_scores,
                'thresholds': np.arange(0.60, 0.96, 0.05)
            },
            'TF-IDF-Cosine': {
                'scores': self.cosine_scores,
                'thresholds': np.arange(0.40, 0.96, 0.05)
            }
        }

        for detector_name, config in detectors.items():
            print(f"\nSweeping thresholds for {detector_name}...")
            detector_results = []

            scores = config['scores']
            thresholds = config['thresholds']

            for threshold in tqdm(thresholds):
                metrics = compute_all_metrics(
                    scores, self.is_leaky, threshold, n_bootstrap
                )
                detector_results.append(metrics)

            results[detector_name] = detector_results

        return results

    def save_results(self, results: Dict, output_dir: str = "extension1/results"):
        """Save evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save full results as pickle
        with open(output_path / f"{self.model_name}_firewall_results.pkl", 'wb') as f:
            pickle.dump(results, f)

        # Save summary CSV for each detector
        for detector_name, detector_results in results.items():
            df = pd.DataFrame(detector_results)
            csv_name = f"{self.model_name}_{detector_name.replace(' ', '_')}.csv"
            df.to_csv(output_path / csv_name, index=False)
            print(f"Saved {csv_name}")

    def run_evaluation(self, results_path: str, retrieved_contexts_path: str,
                      n_bootstrap: int = 1000, output_dir: str = "extension1/results"):
        """
        Run complete evaluation pipeline.

        Args:
            results_path: Path to model results JSONL
            retrieved_contexts_path: Path to retrieved contexts JSONL
            n_bootstrap: Number of bootstrap samples
            output_dir: Directory to save results
        """
        # Load data
        self.load_data(results_path, retrieved_contexts_path)

        # Label leakage
        self.label_leakage()

        # Compute similarities
        self.compute_similarities()

        # Sweep thresholds
        results = self.sweep_thresholds(n_bootstrap)

        # Save results
        self.save_results(results, output_dir)

        print("\nEvaluation complete!")
        return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Firewall evaluation pipeline")
    parser.add_argument("--model", type=str, default="Llama-2-7b-chat-hf",
                       help="Model name")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                       help="Tokenizer name")
    parser.add_argument("--min-overlap", type=int, default=20,
                       help="Minimum token overlap for leakage labeling")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                       help="Number of bootstrap samples")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to results JSONL")
    parser.add_argument("--contexts", type=str, required=True,
                       help="Path to retrieved contexts JSONL")
    parser.add_argument("--output-dir", type=str, default="extension1/results",
                       help="Output directory")

    args = parser.parse_args()

    # Run evaluation
    evaluator = FirewallEvaluator(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        min_overlap=args.min_overlap
    )

    evaluator.run_evaluation(
        results_path=args.results,
        retrieved_contexts_path=args.contexts,
        n_bootstrap=args.n_bootstrap,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
