#!/usr/bin/env python3
"""
main.py -- Summary quality grader (single-file)

Usage examples:
  python main.py prompt.txt summary.txt
  python main.py original_article.txt my_summary.txt --json
  python main.py conversation.txt summary.txt --mode conversation --use-embeddings

Inputs:
  - Two text files: original_file summary_file

Outputs:
  - Pretty text to stdout by default
  - Use --json to emit machine-readable JSON
  - Use --out <path> to save JSON or CSV results

Notes:
  - Optional dependencies: nltk, scikit-learn, sentence-transformers, transformers, language_tool_python
  - The script tries to gracefully fall back when optional libs are missing.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import statistics
from collections import Counter
from functools import lru_cache
from typing import List, Dict, Tuple, Any

# ---------- Optional imports (graceful) ----------
try:
    import nltk

    nltk_available = True
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
except Exception:
    nltk_available = False

try:
    import numpy as np
    np_available = True
except Exception:
    np_available = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sklearn_available = True
except Exception:
    sklearn_available = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    st_available = True
except Exception:
    st_available = False

try:
    from transformers import pipeline
    transformers_available = True
except Exception:
    transformers_available = False

try:
    import language_tool_python
    langtool_available = True
except Exception:
    langtool_available = False

# ---------- Utilities ----------
def safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return a / (b + eps)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# ---------- Text preprocessing & tokenization ----------
class TextProcessor:
    def __init__(self, mode: str = "article"):
        assert mode in ("article", "conversation")
        self.mode = mode
        self._word_pattern = re.compile(r"\b[\w']+\b", flags=re.UNICODE)

    def preprocess(self, text: str) -> str:
        if text is None:
            return ""
        text = text.strip()
        
        # Clean ChatGPT conversation format artifacts
        text = self.clean_chatgpt_conversation(text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # If conversation, canonicalize speaker markers like "You:", "Assistant:", "User:"
        if self.mode == "conversation":
            text = re.sub(r"(^|\s)(You|User|Assistant|System|A|B|Person)\s*:", r" [\1\2]:", text)
            # Also replace common markers -> canonical tokens
            text = re.sub(r"\b(you|user|assistant|system)\b\s*:", lambda m: f" [{m.group(1).upper()}]:", text, flags=re.IGNORECASE)
        # Lowercasing left to tokenizers where necessary; keep original casing for NER or grammar checks
        return text

    def clean_chatgpt_conversation(self, text: str) -> str:
        """Remove ChatGPT conversation UI elements and markers"""
        # Remove common ChatGPT conversation artifacts
        patterns_to_remove = [
            r'Skip to content\s*',
            r'This is a copy of a conversation between ChatGPT & Anonymous\.\s*',
            r'Report conversation\s*',
            r'You said:\s*',
            r'ChatGPT said:\s*',
            r'Attach\s*',
            r'Search\s*',
            r'Study\s*',
            r'Voice\s*',
            r'No file chosen\s*',
            r'ChatGPT can make mistakes\. Check important info\.\s*',
            r'^\s*\n+',  # Leading whitespace and newlines
            r'\n+\s*$',  # Trailing whitespace and newlines
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up multiple newlines and whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double newlines
        text = re.sub(r'^\s+|\s+$', '', text)  # Strip leading/trailing whitespace
        
        return text

    def tokenize_sentences(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if nltk_available:
            return [s.strip() for s in nltk.tokenize.sent_tokenize(text) if s.strip()]
        # fallback
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_words(self, text: str) -> List[str]:
        if not text:
            return []
        # lowercase normalization for word matching
        words = [w.lower() for w in self._word_pattern.findall(text)]
        return words

# ---------- Scoring components ----------
class SummaryGrader:
    def __init__(self,
                 mode: str = "article",
                 use_embeddings: bool = False,
                 use_nli: bool = False,
                 top_k: int = 30,
                 ideal_min: float = 0.05,
                 ideal_max: float = 0.25,
                 soft_max: float = 0.4):
        self.processor = TextProcessor(mode=mode)
        self.mode = mode
        self.use_embeddings = use_embeddings and st_available
        self.use_nli = use_nli and transformers_available
        self.top_k = top_k
        self.ideal_min = ideal_min
        self.ideal_max = ideal_max
        self.soft_max = soft_max

        # cached models
        self._tfidf_vectorizer = None
        self._embedding_model = None
        self._nli_pipeline = None
        if self.use_embeddings:
            try:
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                # fallback to disabled
                self._embedding_model = None
                self.use_embeddings = False

        if self.use_nli:
            try:
                # use a distilled entailment model pipeline (may download model)
                self._nli_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1)
            except Exception:
                self._nli_pipeline = None
                self.use_nli = False

    # ---------- TF-IDF utilities ----------
    def _ensure_tfidf(self, corpus: List[str]):
        if not sklearn_available:
            raise RuntimeError("scikit-learn is required for TF-IDF fallback or content keyword extraction.")
        # If no vectorizer or corpus size changes significantly, re-fit for reliability in pairwise mode we fit per-pair but reuse config
        vect = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')
        self._tfidf_vectorizer = vect.fit(corpus)
        return self._tfidf_vectorizer

    # ---------- Metric implementations ----------
    def word_overlap_f1(self, original: str, summary: str) -> float:
        o_words = set(self.processor.tokenize_words(original))
        s_words = set(self.processor.tokenize_words(summary))
        if not o_words and not s_words:
            return 1.0
        if not o_words or not s_words:
            return 0.0
        overlap = len(o_words & s_words)
        precision = safe_div(overlap, len(s_words))
        recall = safe_div(overlap, len(o_words))
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return clamp01(f1)

    def semantic_similarity(self, original: str, summary: str) -> float:
        # Prefer embeddings if available
        if self.use_embeddings and self._embedding_model is not None:
            try:
                emb = self._embedding_model.encode([original, summary], convert_to_numpy=True, show_progress_bar=False)
                sim = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-9))
                return clamp01((sim + 1) / 2 if sim < -1 or sim > 1 else sim)  # sim likely in [-1,1] for cosine, but keep clamp
            except Exception:
                # fall back to TF-IDF below
                pass

        # TF-IDF fallback (pairwise fit)
        if sklearn_available:
            vect = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')
            tf = vect.fit_transform([original, summary])
            sim = cosine_similarity(tf[0:1], tf[1:2])[0][0]
            # cosine similarity in [0,1] for tfidf non-negative vectors
            return float(clamp01(sim))
        # last resort: word overlap
        return self.word_overlap_f1(original, summary)

    def content_coverage(self, original: str, summary: str, k: int = None) -> Dict[str, Any]:
        k = k or self.top_k
        if not sklearn_available:
            # fallback to frequency-based top-k
            words = self.processor.tokenize_words(original)
            freq = Counter(words)
            top = [w for w,_ in freq.most_common(k)]
        else:
            # use TF-IDF on original only to get top-k important words
            vect = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')
            tf = vect.fit_transform([original])
            # get feature names and scores
            try:
                feature_names = vect.get_feature_names_out()
            except Exception:
                feature_names = vect.get_feature_names()
            scores = tf.toarray()[0]
            pairs = sorted(list(zip(feature_names, scores)), key=lambda x: x[1], reverse=True)
            top = [w for w,s in pairs[:k] if w.strip()]
            # if top is empty, fallback to freq
            if not top:
                words = self.processor.tokenize_words(original)
                top = [w for w,_ in Counter(words).most_common(k)]
        # compute coverage
        summary_words = set(self.processor.tokenize_words(summary))
        covered = [w for w in top if w.lower() in summary_words]
        missed = [w for w in top if w.lower() not in summary_words]
        coverage_ratio = safe_div(len(covered), max(1, len(top)))
        return {"top_k": top, "covered": covered, "missed": missed, "coverage": float(coverage_ratio)}

    def length_appropriateness(self, original: str, summary: str,
                               ideal_min: float = None, ideal_max: float = None, soft_max: float = None) -> float:
        ideal_min = self.ideal_min if ideal_min is None else ideal_min
        ideal_max = self.ideal_max if ideal_max is None else ideal_max
        soft_max = self.soft_max if soft_max is None else soft_max
        o_len = max(1, len(self.processor.tokenize_words(original)))
        s_len = max(0, len(self.processor.tokenize_words(summary)))
        ratio = safe_div(s_len, o_len)
        # If within ideal range -> score 1.0
        if ideal_min <= ratio <= ideal_max:
            return 1.0
        # Smooth penalty: triangular style for short; linear up to soft_max for long
        if ratio < ideal_min:
            # relative shortness
            diff = (ideal_min - ratio) / max(ideal_min, 1e-9)
            score = clamp01(1.0 - diff)
            return score
        else:
            # ratio > ideal_max
            if ratio <= soft_max:
                # slight penalty
                scale = (ratio - ideal_max) / max((soft_max - ideal_max), 1e-9)
                score = clamp01(1.0 - 0.6 * scale)  # up to 40% penalty
                return score
            else:
                # very long -> stronger penalty
                diff = (ratio - soft_max) / max(soft_max, 1e-9)
                score = clamp01(1.0 - 0.6 - 0.4 * clamp01(diff))
                return score

    def coherence_score(self, summary: str) -> float:
        # compute adjacency similarity of sentences: if sentences have similar embeddings the summary flows; we use embeddings if available else tfidf on sentences
        sents = self.processor.tokenize_sentences(summary)
        if len(sents) <= 1:
            return 1.0 if sents else 0.0
        # embeddings preferred
        if self.use_embeddings and self._embedding_model is not None:
            try:
                emb = self._embedding_model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
                sims = []
                for i in range(len(emb)-1):
                    a, b = emb[i], emb[i+1]
                    sims.append(float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9)))
                # normalize cosine [-1,1] -> map to [0,1] and average
                mapped = [clamp01((x+1)/2) for x in sims]
                return float(statistics.mean(mapped))
            except Exception:
                pass
        # TF-IDF sentence vectors fallback
        if sklearn_available:
            vect = TfidfVectorizer(max_features=500, stop_words='english')
            tf = vect.fit_transform(sents)
            sims = []
            for i in range(tf.shape[0]-1):
                sim = cosine_similarity(tf[i:i+1], tf[i+1:i+2])[0][0]
                sims.append(float(sim))
            return float(clamp01(sum(sims)/len(sims))) if sims else 0.0
        # fallback: heuristic based on use of transitions
        transitions = ["however", "therefore", "then", "also", "meanwhile", "furthermore", "but", "and"]
        score = 0.0
        for t in transitions:
            if t in summary.lower():
                score += 0.1
        return clamp01(score)

    def grammar_issues(self, summary: str) -> Dict[str, Any]:
        # uses language_tool_python if available
        if not langtool_available:
            return {"available": False, "issues": [], "count": None}
        try:
            tool = language_tool_python.LanguageTool('en-US')
            matches = tool.check(summary)
            # map to simplified issues
            issues = []
            for m in matches[:50]:
                issues.append({"message": m.message, "offset": m.offset, "length": m.errorLength})
            return {"available": True, "issues": issues, "count": len(matches)}
        except Exception as e:
            return {"available": False, "error": str(e)}

    def nli_factuality(self, original: str, summary: str, top_n: int = 3) -> Dict[str, Any]:
        # Simple NLI: pick top_n important sentences from original (by TF-IDF), and check entailment vs summary
        if not self.use_nli or self._nli_pipeline is None:
            return {"available": False}
        try:
            orig_sents = self.processor.tokenize_sentences(original)
            if not orig_sents:
                return {"available": True, "judgments": [], "score": 1.0}
            # rank by manual heuristic: sentence length * number of content words
            scored = []
            for s in orig_sents:
                wn = len(self.processor.tokenize_words(s))
                scored.append((s, wn))
            top = [s for s,_ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]]
            judgments = []
            # each claim: is it entailed/contradicted by summary?
            for claim in top:
                # transformer pipeline expects hypothesis/premise mapping sometimes reversed; for NLI model we can pass text pair and interpret labels
                res = self._nli_pipeline(f"{claim} </s> {summary}", truncation=True)
                # result format may vary: label and score
                label = res[0].get('label')
                score = float(res[0].get('score', 0.0))
                judgments.append({"claim": claim, "label": label, "score": score})
            # compute a coarse factuality score: penalize contradictions
            contra = sum(1 for j in judgments if 'CONTRADI' in j['label'].upper() or 'CONTRADICTION' in j['label'].upper())
            entail = sum(1 for j in judgments if 'ENTAIL' in j['label'].upper())
            total = max(1, len(judgments))
            score = clamp01((entail - contra) / total * 0.5 + 0.5)  # map roughly into [0,1]
            return {"available": True, "judgments": judgments, "score": float(score)}
        except Exception as e:
            return {"available": False, "error": str(e)}

    # ---------- Aggregate grade ----------
    def grade_pair(self, original: str, summary: str) -> Dict[str, Any]:
        orig_proc = self.processor.preprocess(original)
        sum_proc = self.processor.preprocess(summary)

        # metrics
        overlap = float(self.word_overlap_f1(orig_proc, sum_proc))
        semsim = float(self.semantic_similarity(orig_proc, sum_proc))
        coverage_res = self.content_coverage(orig_proc, sum_proc, k=self.top_k)
        length_score = float(self.length_appropriateness(orig_proc, sum_proc))
        coherence = float(self.coherence_score(sum_proc))
        grammar = self.grammar_issues(sum_proc)
        nli = self.nli_factuality(orig_proc, sum_proc) if self.use_nli else {"available": False}

        # combine scores with weights (configurable later)
        # default weights:
        weights = {
            "overlap": 0.20,
            "semantic": 0.30,
            "coverage": 0.20,
            "length": 0.10,
            "coherence": 0.15,
            "factuality": 0.05
        }
        # factuality uses NLI score if available else 1.0
        factuality_score = float(nli.get("score", 1.0) if nli.get("available", False) else 1.0)

        combined = (
            weights["overlap"] * overlap +
            weights["semantic"] * semsim +
            weights["coverage"] * coverage_res["coverage"] +
            weights["length"] * length_score +
            weights["coherence"] * coherence +
            weights["factuality"] * factuality_score
        )
        combined = clamp01(combined)

        result = {
            "overlap_f1": overlap,
            "semantic_similarity": semsim,
            "coverage": coverage_res,
            "length_score": length_score,
            "coherence": coherence,
            "grammar": grammar,
            "nli_factuality": nli,
            "combined_score": combined,
            "explanation": {
                "weights": weights,
                "short": f"Score {combined:.3f} (combined).",
                "warnings": self._collect_warnings(orig_proc, sum_proc, coverage_res, grammar, nli)
            }
        }
        return result

    def _collect_warnings(self, original: str, summary: str, coverage_res: Dict[str, Any], grammar_res: Dict[str, Any], nli_res: Dict[str, Any]) -> List[str]:
        warns = []
        # too short or empty
        if len(self.processor.tokenize_words(summary)) < 3:
            warns.append("Summary is very short (<3 words).")
        # missed many top keywords
        miss_ratio = safe_div(len(coverage_res.get("missed", [])), max(1, len(coverage_res.get("top_k", []))))
        if miss_ratio > 0.7:
            warns.append(f"Summary missed {int(miss_ratio*100)}% of top-{len(coverage_res.get('top_k', []))} keywords.")
        # grammar issues
        if isinstance(grammar_res, dict) and grammar_res.get("available") and grammar_res.get("count", 0) > 5:
            warns.append(f"Grammar tool found {grammar_res.get('count')} issues.")
        # NLI contradictions
        if nli_res.get("available") and nli_res.get("judgments"):
            contra = sum(1 for j in nli_res["judgments"] if 'CONTRADI' in j.get("label","").upper())
            if contra > 0:
                warns.append(f"NLI detected {contra} possible contradictions in top claims.")
        return warns

# ---------- CLI & batch handling ----------
def read_batch_csv(path: str) -> List[Dict[str,str]]:
    rows = []
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        # Expect columns original, summary (case-insensitive)
        fieldnames = [c.lower() for c in reader.fieldnames] if reader.fieldnames else []
        orig_key = None
        sum_key = None
        for fn in reader.fieldnames:
            if fn.lower() == 'original':
                orig_key = fn
            if fn.lower() == 'summary':
                sum_key = fn
        if not orig_key or not sum_key:
            raise ValueError("Batch CSV must include columns named 'original' and 'summary'.")
        for r in reader:
            rows.append({"original": r[orig_key], "summary": r[sum_key]})
    return rows

def emit_pretty(result: Dict[str,Any], title: str = "Result"):
    print(f"=== {title} ===")
    print(f"Combined score: {result['combined_score']:.3f}")
    print(f" - Overlap F1: {result['overlap_f1']:.3f}")
    print(f" - Semantic sim: {result['semantic_similarity']:.3f}")
    print(f" - Coverage: {result['coverage']['coverage']:.3f} (covered {len(result['coverage']['covered'])}/{len(result['coverage']['top_k'])})")
    print(f" - Length score: {result['length_score']:.3f}")
    print(f" - Coherence: {result['coherence']:.3f}")
    if result['grammar'].get("available", False):
        print(f" - Grammar issues: {result['grammar'].get('count')}")
    if result['nli_factuality'].get('available', False):
        print(f" - NLI factuality score: {result['nli_factuality'].get('score'):.3f}")
    if result['explanation']['warnings']:
        print("Warnings:")
        for w in result['explanation']['warnings']:
            print("  -", w)
    # show missed keywords sample
    missed = result['coverage'].get('missed', [])
    if missed:
        print("Missed important keywords (sample):", ", ".join(missed[:10]))
    print()

def main(argv):
    parser = argparse.ArgumentParser(description="Summary quality grader - accepts two .txt files")
    parser.add_argument("original_file", type=str, help="Path to original text file (.txt)")
    parser.add_argument("summary_file", type=str, help="Path to summary text file (.txt)")
    parser.add_argument("--out", type=str, help="Save results to file (json or csv depending on extension)")
    parser.add_argument("--json", action="store_true", help="Print JSON result(s) to stdout")
    parser.add_argument("--use-embeddings", action="store_true", help="Use sentence-transformers embeddings if available")
    parser.add_argument("--use-nli", action="store_true", help="Use NLI (transformers) for factuality check if available (can be slow)")
    parser.add_argument("--mode", choices=["article", "conversation"], default="article", help="Preprocessing mode")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k keywords for coverage")
    parser.add_argument("--ideal-min", type=float, default=0.05, help="Ideal min summary/original length ratio")
    parser.add_argument("--ideal-max", type=float, default=0.25, help="Ideal max summary/original length ratio")
    parser.add_argument("--soft-max", type=float, default=0.4, help="Soft max ratio after which heavy penalty applies")
    args = parser.parse_args(argv)

    # Read text files
    try:
        with open(args.original_file, 'r', encoding='utf-8') as f:
            original_text = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Original file '{args.original_file}' not found.", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error reading original file '{args.original_file}': {e}", file=sys.stderr)
        sys.exit(2)
    
    try:
        with open(args.summary_file, 'r', encoding='utf-8') as f:
            summary_text = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Summary file '{args.summary_file}' not found.", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error reading summary file '{args.summary_file}': {e}", file=sys.stderr)
        sys.exit(2)

    # Create pairs list with the file contents
    pairs = [{"original": original_text, "summary": summary_text}]

    grader = SummaryGrader(mode=args.mode,
                           use_embeddings=args.use_embeddings,
                           use_nli=args.use_nli,
                           top_k=args.top_k,
                           ideal_min=args.ideal_min,
                           ideal_max=args.ideal_max,
                           soft_max=args.soft_max)

    results = []
    for i,p in enumerate(pairs):
        try:
            res = grader.grade_pair(p.get("original",""), p.get("summary",""))
            out = {"index": i, "original": p.get("original",""), "summary": p.get("summary",""), "result": res}
            results.append(out)
        except Exception as e:
            print(f"Error grading pair index {i}: {e}", file=sys.stderr)
            results.append({"index": i, "error": str(e)})

    # emit output
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for i,r in enumerate(results):
            if "error" in r:
                print(f"Pair {i}: ERROR: {r['error']}")
            else:
                emit_pretty(r["result"], title=f"Pair {i}")

    # optionally save out
    if args.out:
        try:
            if args.out.lower().endswith(".json"):
                with open(args.out, "w", encoding="utf-8") as fh:
                    json.dump(results, fh, indent=2, ensure_ascii=False)
            elif args.out.lower().endswith(".csv"):
                # flatten to simple CSV: index, combined_score, overlap, semantic, coverage, warnings
                with open(args.out, "w", newline='', encoding='utf-8') as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["index", "combined_score", "overlap_f1", "semantic_similarity", "coverage", "warnings"])
                    for r in results:
                        if "error" in r:
                            writer.writerow([r["index"], "", "", "", "", r["error"]])
                        else:
                            res = r["result"]
                            writer.writerow([r["index"], res["combined_score"], res["overlap_f1"], res["semantic_similarity"], res["coverage"]["coverage"], ";".join(res["explanation"]["warnings"])])
            else:
                # unknown extension: write JSON
                with open(args.out, "w", encoding='utf-8') as fh:
                    json.dump(results, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            print("Error saving output file:", e, file=sys.stderr)
            sys.exit(3)

    # exit with code 0 if all graded, else non-zero if any error
    if any("error" in r for r in results):
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])
