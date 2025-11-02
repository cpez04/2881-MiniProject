"""
Similarity detectors for firewall evaluation.
Implements ROUGE-L, MinHash-Jaccard, and TF-IDF Cosine similarity.
"""

import numpy as np
from typing import List, Tuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from transformers import AutoTokenizer
import re


def tokenize_text(text: str, tokenizer) -> List[str]:
    """Tokenize text using the model tokenizer, return string tokens."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return [str(tid) for tid in token_ids]


class ROUGELDetector:
    """ROUGE-L (F1) detector."""

    def __init__(self, tokenizer_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def compute_rouge_l(self, output: str, chunk: str) -> float:
        """
        Compute ROUGE-L F1 score between output and chunk.

        Args:
            output: Generated output text
            chunk: Retrieved chunk text

        Returns:
            ROUGE-L F1 score (0-1)
        """
        output_tokens = tokenize_text(output, self.tokenizer)
        chunk_tokens = tokenize_text(chunk, self.tokenizer)

        if len(output_tokens) == 0 or len(chunk_tokens) == 0:
            return 0.0

        lcs = self._lcs_length(output_tokens, chunk_tokens)

        # Precision and recall
        precision = lcs / len(output_tokens) if len(output_tokens) > 0 else 0
        recall = lcs / len(chunk_tokens) if len(chunk_tokens) > 0 else 0

        # F1 score
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    def compute_max_similarity(self, output: str, chunks: List[str]) -> float:
        """Compute max ROUGE-L score across all chunks."""
        if not chunks:
            return 0.0
        scores = [self.compute_rouge_l(output, chunk) for chunk in chunks]
        return max(scores)


class MinHashJaccardDetector:
    """MinHash-based Jaccard similarity detector with 8-gram shingles."""

    def __init__(self, tokenizer_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 n_gram: int = 8, num_perm: int = 192):
        """
        Args:
            tokenizer_name: HuggingFace tokenizer
            n_gram: Size of shingles (8-grams)
            num_perm: Number of hash functions (192)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.n_gram = n_gram
        self.num_perm = num_perm

    def _create_shingles(self, tokens: List[str]) -> set:
        """Create n-gram shingles from tokens."""
        shingles = set()
        for i in range(len(tokens) - self.n_gram + 1):
            shingle = ' '.join(tokens[i:i + self.n_gram])
            shingles.add(shingle)
        return shingles

    def _compute_jaccard(self, set1: set, set2: set) -> float:
        """Compute exact Jaccard similarity."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def compute_jaccard(self, output: str, chunk: str) -> float:
        """
        Compute Jaccard similarity on 8-gram shingles.

        Args:
            output: Generated output text
            chunk: Retrieved chunk text

        Returns:
            Jaccard similarity (0-1)
        """
        output_tokens = tokenize_text(output, self.tokenizer)
        chunk_tokens = tokenize_text(chunk, self.tokenizer)

        output_shingles = self._create_shingles(output_tokens)
        chunk_shingles = self._create_shingles(chunk_tokens)

        return self._compute_jaccard(output_shingles, chunk_shingles)

    def compute_max_similarity(self, output: str, chunks: List[str]) -> float:
        """Compute max Jaccard score across all chunks."""
        if not chunks:
            return 0.0
        scores = [self.compute_jaccard(output, chunk) for chunk in chunks]
        return max(scores)


class TFIDFCosineDetector:
    """TF-IDF cosine similarity detector."""

    def __init__(self, tokenizer_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Args:
            tokenizer_name: HuggingFace tokenizer for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vectorizer = None
        self.fitted = False

    def _tokenize_for_tfidf(self, text: str) -> str:
        """Tokenize and join with spaces for TF-IDF."""
        tokens = tokenize_text(text, self.tokenizer)
        return ' '.join(tokens)

    def fit(self, all_texts: List[str]):
        """
        Fit TF-IDF vectorizer on all outputs and chunks.

        Args:
            all_texts: All outputs and chunks combined
        """
        # Tokenize all texts
        tokenized_texts = [self._tokenize_for_tfidf(text) for text in all_texts]

        # Fit vectorizer with unigrams and bigrams
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            norm='l2'
        )
        self.vectorizer.fit(tokenized_texts)
        self.fitted = True

    def compute_cosine(self, output: str, chunk: str) -> float:
        """
        Compute TF-IDF cosine similarity.

        Args:
            output: Generated output text
            chunk: Retrieved chunk text

        Returns:
            Cosine similarity (0-1)
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before computing similarity")

        output_tok = self._tokenize_for_tfidf(output)
        chunk_tok = self._tokenize_for_tfidf(chunk)

        vectors = self.vectorizer.transform([output_tok, chunk_tok])
        sim = cosine_similarity(vectors[0:1], vectors[1:2])[0, 0]

        return float(sim)

    def compute_max_similarity(self, output: str, chunks: List[str]) -> float:
        """Compute max cosine similarity across all chunks."""
        if not chunks:
            return 0.0
        scores = [self.compute_cosine(output, chunk) for chunk in chunks]
        return max(scores)
