"""
========== Retrieval Evaluation Metrics ==========
Standard IR metrics for evaluating retrieval quality:
- MRR (Mean Reciprocal Rank)
- NDCG@k (Normalized Discounted Cumulative Gain)
- Hit Rate@k
- MAP@k (Mean Average Precision)

All metrics compare retrieved chunks against golden_context keywords 
to determine relevance without requiring ground-truth chunk IDs.
"""

import math
from typing import List


def _chunk_relevance(chunk_text: str, golden_terms: List[str]) -> float:
    """
    Compute relevance score for a chunk against golden context terms.
    
    Returns a score 0.0-1.0 based on how many golden terms appear.
    Uses substring matching (case-insensitive) to be robust against 
    minor text variations between chunk and golden answer.
    """
    if not golden_terms:
        return 0.0
    text_lower = chunk_text.lower()
    hits = sum(1 for term in golden_terms if term.lower() in text_lower)
    return hits / len(golden_terms)


def _binary_relevance(chunk_text: str, golden_terms: List[str], threshold: float = 0.3) -> int:
    """Binary relevance: 1 if enough golden terms match, else 0."""
    return 1 if _chunk_relevance(chunk_text, golden_terms) >= threshold else 0


def mrr(chunks: List[str], golden_terms: List[str]) -> float:
    """
    Mean Reciprocal Rank.
    
    Returns the reciprocal of the rank of the first relevant chunk.
    If no relevant chunk found, returns 0.
    
    Args:
        chunks: List of retrieved chunk texts in ranked order
        golden_terms: Keywords that should appear in relevant chunks
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    if not chunks or not golden_terms:
        return 0.0
    for i, chunk in enumerate(chunks):
        if _binary_relevance(chunk, golden_terms):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(chunks: List[str], golden_terms: List[str], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    
    Uses graded relevance (proportion of golden terms matched) 
    and log2(i+1) discount.
    
    Args:
        chunks: List of retrieved chunk texts in ranked order
        golden_terms: Keywords for relevance scoring
        k: Cutoff rank
        
    Returns:
        NDCG@k score (0.0 to 1.0)
    """
    if not chunks or not golden_terms:
        return 0.0
    
    k = min(k, len(chunks))
    # Compute DCG
    dcg = 0.0
    for i in range(k):
        rel = _chunk_relevance(chunks[i], golden_terms)
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0
    
    # Compute ideal DCG (ideal: all k chunks have relevance=1.0)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(k))
    
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(chunks: List[str], golden_terms: List[str], k: int = 5) -> float:
    """
    Hit Rate at k: did we retrieve at least one relevant chunk in top-k?
    
    Returns:
        1.0 if any chunk in top-k is relevant, else 0.0
    """
    if not chunks or not golden_terms:
        return 0.0
    k = min(k, len(chunks))
    for i in range(k):
        if _binary_relevance(chunks[i], golden_terms):
            return 1.0
    return 0.0


def map_at_k(chunks: List[str], golden_terms: List[str], k: int = 5) -> float:
    """
    Mean Average Precision at k.
    
    Computes precision at each rank where a relevant chunk is found,
    then averages over all relevant chunks.
    
    Returns:
        MAP@k score (0.0 to 1.0)
    """
    if not chunks or not golden_terms:
        return 0.0
    
    k = min(k, len(chunks))
    relevant_count = 0
    precision_sum = 0.0
    
    for i in range(k):
        if _binary_relevance(chunks[i], golden_terms):
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    
    # Normalize by min(k, total_relevant) - but since we don't know total_relevant,
    # use the count of relevant found as denominator (standard approximation)
    if relevant_count == 0:
        return 0.0
    return precision_sum / relevant_count


def compute_all_retrieval_metrics(
    chunks: List[str], 
    golden_terms: List[str], 
    k: int = 5
) -> dict:
    """
    Compute all retrieval metrics in one pass.
    
    Returns:
        dict with keys: mrr, ndcg_at_k, hit_rate, map_at_k
    """
    return {
        'mrr': round(mrr(chunks, golden_terms), 4),
        f'ndcg_at_{k}': round(ndcg_at_k(chunks, golden_terms, k), 4),
        f'hit_rate_at_{k}': round(hit_rate_at_k(chunks, golden_terms, k), 4),
        f'map_at_{k}': round(map_at_k(chunks, golden_terms, k), 4),
    }
