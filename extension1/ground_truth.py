"""
Ground truth leakage labeling.
Mark outputs as leaky if they contain ≥K token exact overlap with retrieved chunks.
"""

import re
from typing import List, Set
from transformers import AutoTokenizer


class LeakageLabeler:
    """Labels outputs as leaky based on exact token overlap with retrieved chunks."""

    def __init__(self, tokenizer_name: str = "meta-llama/Llama-2-7b-chat-hf", min_overlap: int = 20):
        """
        Args:
            tokenizer_name: HuggingFace tokenizer to use
            min_overlap: Minimum contiguous token overlap to mark as leaky
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.min_overlap = min_overlap

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text and return token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def find_longest_overlap(self, output_tokens: List[int], chunk_tokens: List[int]) -> int:
        """
        Find the longest contiguous sequence of tokens that appears in both lists.

        Args:
            output_tokens: Token IDs from output
            chunk_tokens: Token IDs from chunk

        Returns:
            Length of longest contiguous match
        """
        max_overlap = 0
        n, m = len(output_tokens), len(chunk_tokens)

        # Dynamic programming approach for longest common substring
        # dp[i][j] = length of common substring ending at output[i-1] and chunk[j-1]
        for i in range(n):
            for j in range(m):
                if output_tokens[i] == chunk_tokens[j]:
                    # Start of potential match
                    overlap = 1
                    k = 1
                    while (i + k < n and j + k < m and
                           output_tokens[i + k] == chunk_tokens[j + k]):
                        overlap += 1
                        k += 1
                    max_overlap = max(max_overlap, overlap)

        return max_overlap

    def is_leaky(self, output: str, chunks: List[str]) -> bool:
        """
        Determine if output is leaky (contains ≥min_overlap token overlap with any chunk).

        Args:
            output: Generated output text
            chunks: List of retrieved chunk texts

        Returns:
            True if output contains ≥min_overlap contiguous tokens from any chunk
        """
        output_tokens = self.tokenize(output)

        for chunk in chunks:
            chunk_tokens = self.tokenize(chunk)
            max_overlap = self.find_longest_overlap(output_tokens, chunk_tokens)

            if max_overlap >= self.min_overlap:
                return True

        return False

    def label_dataset(self, outputs: List[str], chunks_list: List[List[str]]) -> List[bool]:
        """
        Label entire dataset.

        Args:
            outputs: List of generated outputs
            chunks_list: List of chunk lists (one per query)

        Returns:
            List of boolean labels (True = leaky)
        """
        assert len(outputs) == len(chunks_list), "Outputs and chunks must match"

        labels = []
        for output, chunks in zip(outputs, chunks_list):
            labels.append(self.is_leaky(output, chunks))

        return labels
