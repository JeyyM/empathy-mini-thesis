"""Length-adjusted summary evaluation module.

Provides summarize_evaluation(source_text, summary_text) returning metrics that adjust
topic coverage and semantic similarity for length, redundancy, and density.
Uses only standard library; replace get_embedding with a real embedding model for production.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Dict

def sentence_split(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]*\b", text.lower())

STOPWORDS = set("""
a an the and or if but while is are was were be been being to of in on for with as at by from this that these those it its
""".split())

def get_embedding(text: str) -> List[float]:
    vec = [0.0] * 64
    for token in re.findall(r"\w+", text.lower()):
        vec[hash(token) % 64] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def extract_concepts(text: str, max_concepts: int | None = None) -> List[str]:
    tokens = tokenize(text)
    content = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    freq = Counter(content)
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0])))
    if max_concepts is None:
        max_concepts = min(60, int(math.sqrt(len(freq)) + 8))
    return [w for w, _ in ranked[:max_concepts]]

def summarize_evaluation(source_text: str, summary_text: str) -> Dict[str, float]:
    source_sentences = sentence_split(source_text)
    summary_sentences = sentence_split(summary_text)
    source_tokens = tokenize(source_text)
    summary_tokens = tokenize(summary_text)

    if len(summary_tokens) < 12:
        return {"error": 1, "message": "Summary too short for reliable scoring.", "final_score": 0.0}

    concepts = extract_concepts(source_text)
    concept_embeddings = {c: get_embedding(c) for c in concepts}
    summary_sentence_embeddings = [get_embedding(s) for s in summary_sentences]

    covered = 0
    similarity_threshold = 0.7
    for emb in concept_embeddings.values():
        max_sim = 0.0
        for s_emb in summary_sentence_embeddings:
            sim = cosine(emb, s_emb)
            if sim > max_sim:
                max_sim = sim
        if max_sim >= similarity_threshold:
            covered += 1
    coverage_raw = covered / (len(concepts) or 1)

    compression_ratio_target = 0.15
    ideal_len = max(1, int(len(source_tokens) * compression_ratio_target))
    coverage_penalty = min(1.0, ideal_len / (len(summary_tokens) or 1))
    final_topic_coverage = coverage_raw * coverage_penalty

    source_sentence_embeddings = [get_embedding(s) for s in source_sentences]
    centroid_source = [sum(vals) / len(source_sentence_embeddings) for vals in zip(*source_sentence_embeddings)] if source_sentence_embeddings else get_embedding(source_text)
    centroid_summary = [sum(vals) / len(summary_sentence_embeddings) for vals in zip(*summary_sentence_embeddings)] if summary_sentence_embeddings else get_embedding(summary_text)
    global_sim = cosine(centroid_source, centroid_summary)

    redundancy_scores = []
    for i in range(len(summary_sentence_embeddings)):
        for j in range(i + 1, len(summary_sentence_embeddings)):
            redundancy_scores.append(cosine(summary_sentence_embeddings[i], summary_sentence_embeddings[j]))
    redundancy = sum(redundancy_scores) / len(redundancy_scores) if redundancy_scores else 0.0
    redundancy_floor, redundancy_ceiling = 0.3, 0.85
    if redundancy <= redundancy_floor:
        redundancy_penalty = 1.0
    elif redundancy >= redundancy_ceiling:
        redundancy_penalty = 0.0
    else:
        redundancy_penalty = 1 - (redundancy - redundancy_floor) / (redundancy_ceiling - redundancy_floor)

    info_density = covered / (len(summary_tokens) or 1)
    target_density = 0.015
    info_score = min(1.0, info_density / target_density)
    brevity_ratio = (len(summary_tokens) or 1) / ideal_len
    epsilon = 1e-6
    length_factor = math.exp(-abs(math.log(max(epsilon, brevity_ratio))))

    semantic_score = global_sim * redundancy_penalty * length_factor * (0.5 + 0.5 * info_score)

    w1, w2 = 0.55, 0.45
    final_score = w1 * final_topic_coverage + w2 * semantic_score

    return {
        "final_score": round(final_score, 4),
        "topic_coverage": round(final_topic_coverage, 4),
        "coverage_raw": round(coverage_raw, 4),
        "coverage_penalty": round(coverage_penalty, 4),
        "global_similarity": round(global_sim, 4),
        "redundancy": round(redundancy, 4),
        "redundancy_penalty": round(redundancy_penalty, 4),
        "info_density": round(info_density, 6),
        "info_score": round(info_score, 4),
        "length_factor": round(length_factor, 4),
        "summary_tokens": len(summary_tokens),
        "source_tokens": len(source_tokens),
        "ideal_summary_tokens": ideal_len,
        "concept_count": len(concepts),
        "concepts_covered": covered,
    }

__all__ = ["summarize_evaluation"]
