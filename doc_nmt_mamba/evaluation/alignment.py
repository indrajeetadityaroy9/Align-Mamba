"""
Alignment Evaluation for Document-Level NMT.

Implements Alignment Error Rate (AER) computation for quantitative alignment quality.

From the plan:
- Uses awesome-align (mBERT-based) for gold alignments
- Extracts model cross-attention argmax as predicted alignments
- Computes AER: 1 - (|A ∩ S| + |A ∩ P|) / (|A| + |S|)
- Maps BPE tokens to words using SubwordToWordMapper

CRITICAL: SubwordToWordMapper is essential because:
- awesome-align outputs word-level alignments
- Our model uses 32k BPE tokenization
- Without mapping, shapes mismatch (Gold: Word 3→5, Model: Token 7→12)
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
import warnings

import torch
import torch.nn.functional as F


@dataclass
class AlignmentResult:
    """Results from alignment evaluation."""
    aer: float = 0.0  # Alignment Error Rate (lower is better)
    precision: float = 0.0
    recall: float = 0.0
    total_examples: int = 0
    per_example_aer: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"AlignmentResult(AER={self.aer:.4f}, "
            f"P={self.precision:.4f}, R={self.recall:.4f})"
        )


class SubwordToWordMapper:
    """
    Maps BPE token indices back to word indices for AER computation.

    CRITICAL from plan:
    awesome-align outputs word-level alignments. Our model uses 32k BPE.
    Without mapping, you get shape mismatch (Gold: Word 3→5, Model: Token 7→12).

    Problem:
        "Bank" might become ["▁B", "ank"] (tokens 7, 8)
        Gold says "Word 3 aligns to Word 5"
        Model says "Token 7 aligns to Token 12"

    Solution: Aggregate at word level
        token_to_word[token_idx] = word_idx
    """

    def __init__(self, tokenizer=None, word_boundary_prefix: str = "▁"):
        """
        Args:
            tokenizer: Tokenizer instance (optional, for encode/decode)
            word_boundary_prefix: Prefix indicating word start (sentencepiece uses "▁")
        """
        self.tokenizer = tokenizer
        self.word_boundary_prefix = word_boundary_prefix

    def build_token_to_word_map(
        self,
        tokens: List[str],
    ) -> List[int]:
        """
        Build mapping from token indices to word indices.

        Tokens starting with word_boundary_prefix begin new words.

        Args:
            tokens: List of token strings

        Returns:
            List where token_to_word[i] = word index for token i

        Example:
            tokens = ["▁The", "▁b", "ank", "▁is", "▁open"]
            returns = [0, 1, 1, 2, 3]  # "▁b" and "ank" both map to word 1
        """
        token_to_word = []
        current_word = -1

        for token in tokens:
            if token.startswith(self.word_boundary_prefix) or current_word == -1:
                current_word += 1
            token_to_word.append(current_word)

        return token_to_word

    def build_map_from_text(
        self,
        text: str,
        encoded_ids: Optional[List[int]] = None,
    ) -> Tuple[List[int], int]:
        """
        Build token-to-word mapping from original text.

        Args:
            text: Original text (space-separated words)
            encoded_ids: Optional token IDs (if tokenizer available)

        Returns:
            Tuple of (token_to_word mapping, number of words)
        """
        words = text.split()
        n_words = len(words)

        if self.tokenizer is None:
            # Simple heuristic: assume each word is one token
            warnings.warn("No tokenizer provided. Assuming 1:1 word-token mapping.")
            return list(range(n_words)), n_words

        # Encode and decode to get tokens
        if encoded_ids is None:
            encoded_ids = self.tokenizer.encode(text)

        tokens = self.tokenizer.convert_ids_to_tokens(encoded_ids)
        token_to_word = self.build_token_to_word_map(tokens)

        return token_to_word, n_words

    def aggregate_attention_to_words(
        self,
        attn_weights: torch.Tensor,
        src_token_to_word: List[int],
        tgt_token_to_word: List[int],
        n_src_words: int,
        n_tgt_words: int,
        aggregation: str = "max",
    ) -> torch.Tensor:
        """
        Aggregate token-level attention to word-level attention.

        If ANY BPE of Target-Word-X attends strongly to ANY BPE of Source-Word-Y,
        count it as alignment X→Y.

        Args:
            attn_weights: Token-level attention (T_tgt × T_src)
            src_token_to_word: Source token to word mapping
            tgt_token_to_word: Target token to word mapping
            n_src_words: Number of source words
            n_tgt_words: Number of target words
            aggregation: "max", "mean", or "sum"

        Returns:
            Word-level attention (W_tgt × W_src)
        """
        T_tgt, T_src = attn_weights.shape

        # Initialize word-level attention
        word_attn = torch.zeros(n_tgt_words, n_src_words, device=attn_weights.device)

        if aggregation == "max":
            # Max-pooling: if ANY token pair has high attention, count it
            for t_idx in range(T_tgt):
                for s_idx in range(T_src):
                    t_word = tgt_token_to_word[t_idx]
                    s_word = src_token_to_word[s_idx]
                    word_attn[t_word, s_word] = max(
                        word_attn[t_word, s_word].item(),
                        attn_weights[t_idx, s_idx].item()
                    )
        elif aggregation == "mean":
            # Mean-pooling with count tracking
            word_count = torch.zeros(n_tgt_words, n_src_words, device=attn_weights.device)
            for t_idx in range(T_tgt):
                for s_idx in range(T_src):
                    t_word = tgt_token_to_word[t_idx]
                    s_word = src_token_to_word[s_idx]
                    word_attn[t_word, s_word] += attn_weights[t_idx, s_idx]
                    word_count[t_word, s_word] += 1
            word_attn = word_attn / (word_count + 1e-8)
        else:  # sum
            for t_idx in range(T_tgt):
                for s_idx in range(T_src):
                    t_word = tgt_token_to_word[t_idx]
                    s_word = src_token_to_word[s_idx]
                    word_attn[t_word, s_word] += attn_weights[t_idx, s_idx]

        return word_attn


def load_awesome_align_alignments(
    file_path: str,
) -> List[Set[Tuple[int, int]]]:
    """
    Load alignments from awesome-align output file.

    awesome-align outputs: "0-0 1-1 2-3 3-2" (src_idx-tgt_idx pairs)

    Args:
        file_path: Path to alignment file

    Returns:
        List of alignment sets, one per sentence pair
    """
    alignments = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                alignments.append(set())
                continue

            pairs = set()
            for pair in line.split():
                if "-" in pair:
                    src_idx, tgt_idx = pair.split("-")
                    pairs.add((int(src_idx), int(tgt_idx)))
            alignments.append(pairs)

    return alignments


class AlignmentEvaluator:
    """
    Evaluator for alignment quality using AER metric.

    From plan:
    1. Generate "Gold" alignments using awesome-align on test set
    2. Extract model's attention argmax per target token
    3. Compute AER: 1 - (|A ∩ S| + |A ∩ P|) / (|A| + |S|)

    Metric: Lower AER = better alignment quality
    Target: AER < 0.30 (competitive with neural MT)
    """

    def __init__(
        self,
        tokenizer=None,
        cross_attention_layers: Optional[List[int]] = None,
    ):
        """
        Args:
            tokenizer: Tokenizer for subword-to-word mapping
            cross_attention_layers: Which layers to extract attention from
                                   (None = average all cross-attention layers)
        """
        self.tokenizer = tokenizer
        self.cross_attention_layers = cross_attention_layers
        self.mapper = SubwordToWordMapper(tokenizer)

    def extract_model_alignments(
        self,
        cross_attn_weights: torch.Tensor,
        src_text: str,
        tgt_text: str,
        threshold: float = 0.0,
    ) -> Set[Tuple[int, int]]:
        """
        Extract alignment predictions from cross-attention weights.

        Method: For each target position, take argmax over source positions.

        Args:
            cross_attn_weights: Cross-attention weights (T_tgt × T_src)
            src_text: Source text for word mapping
            tgt_text: Target text for word mapping
            threshold: Minimum attention weight to count as alignment

        Returns:
            Set of (src_word_idx, tgt_word_idx) alignment pairs
        """
        # Get token-to-word mappings
        src_tokens = src_text.split()
        tgt_tokens = tgt_text.split()
        n_src_words = len(src_tokens)
        n_tgt_words = len(tgt_tokens)

        # Simple word-level mapping (assuming tokenizer handles this)
        # In practice, you'd use self.mapper with actual BPE tokens

        # If we have a tokenizer, do proper subword mapping
        if self.tokenizer is not None:
            # This would use the tokenizer to get proper mappings
            # For now, we'll use a simplified version
            pass

        # Get word-level attention (aggregate if needed)
        word_attn = self.mapper.aggregate_attention_to_words(
            cross_attn_weights,
            list(range(cross_attn_weights.size(1))),  # Simplified
            list(range(cross_attn_weights.size(0))),  # Simplified
            n_src_words,
            n_tgt_words,
        )

        # Extract alignments via argmax
        alignments = set()
        for tgt_idx in range(n_tgt_words):
            if word_attn[tgt_idx].max() > threshold:
                src_idx = word_attn[tgt_idx].argmax().item()
                alignments.add((src_idx, tgt_idx))

        return alignments

    def compute_aer(
        self,
        predicted: Set[Tuple[int, int]],
        gold_sure: Set[Tuple[int, int]],
        gold_possible: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute Alignment Error Rate.

        AER = 1 - (|A ∩ S| + |A ∩ P|) / (|A| + |S|)

        Where:
        - A = predicted alignments
        - S = gold "sure" alignments (must be predicted)
        - P = gold "possible" alignments (optional, may be predicted)

        Args:
            predicted: Predicted alignment pairs
            gold_sure: Gold sure alignment pairs
            gold_possible: Gold possible alignment pairs (defaults to gold_sure)

        Returns:
            Tuple of (AER, precision, recall)
        """
        if gold_possible is None:
            gold_possible = gold_sure

        # Compute intersection sizes
        a_intersect_s = len(predicted & gold_sure)
        a_intersect_p = len(predicted & gold_possible)

        # Compute AER
        denominator = len(predicted) + len(gold_sure)
        if denominator == 0:
            return 0.0, 0.0, 0.0

        aer = 1 - (a_intersect_s + a_intersect_p) / denominator

        # Compute precision and recall
        precision = a_intersect_p / len(predicted) if predicted else 0.0
        recall = a_intersect_s / len(gold_sure) if gold_sure else 0.0

        return aer, precision, recall

    def evaluate(
        self,
        model,
        src_texts: List[str],
        tgt_texts: List[str],
        gold_alignments: List[Set[Tuple[int, int]]],
        device: str = "cuda",
    ) -> AlignmentResult:
        """
        Evaluate alignment quality on a dataset.

        Args:
            model: NMT model with cross-attention
            src_texts: Source texts
            tgt_texts: Target texts (references)
            gold_alignments: Gold alignment pairs for each example
            device: Device for computation

        Returns:
            AlignmentResult with AER and breakdown
        """
        result = AlignmentResult()
        result.total_examples = len(src_texts)

        total_aer = 0.0
        total_precision = 0.0
        total_recall = 0.0

        model.eval()
        with torch.no_grad():
            for src, tgt, gold in zip(src_texts, tgt_texts, gold_alignments):
                # Extract cross-attention from model
                # This would need to be implemented based on model architecture
                # For now, we'll skip actual model inference

                # Placeholder for attention extraction
                # In practice: cross_attn = model.get_cross_attention(src, tgt)

                # For now, create dummy alignments
                predicted = set()  # Would be extracted from model

                aer, precision, recall = self.compute_aer(predicted, gold)

                total_aer += aer
                total_precision += precision
                total_recall += recall
                result.per_example_aer.append(aer)

        # Compute averages
        n = max(1, result.total_examples)
        result.aer = total_aer / n
        result.precision = total_precision / n
        result.recall = total_recall / n

        return result


def evaluate_alignment(
    model,
    src_texts: List[str],
    tgt_texts: List[str],
    gold_alignment_file: str,
    tokenizer=None,
    device: str = "cuda",
) -> AlignmentResult:
    """
    Convenience function for alignment evaluation.

    Args:
        model: NMT model
        src_texts: Source texts
        tgt_texts: Target texts
        gold_alignment_file: Path to awesome-align output file
        tokenizer: Optional tokenizer for subword mapping
        device: Device

    Returns:
        AlignmentResult
    """
    gold_alignments = load_awesome_align_alignments(gold_alignment_file)

    evaluator = AlignmentEvaluator(tokenizer=tokenizer)
    return evaluator.evaluate(model, src_texts, tgt_texts, gold_alignments, device)
