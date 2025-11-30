#!/usr/bin/env python3
"""
content_recall_grader.py -- Pure content recall focus
Simply: does the summary capture what the conversation was actually about?
Different wording is fine as long as the meaning is preserved.
"""

import re
import json
import argparse
from typing import Dict, Any
from collections import Counter
from pathlib import Path

from length_adjusted_eval import summarize_evaluation

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sklearn_available = True
except ImportError:
    sklearn_available = False

class ContentRecallGrader:
    def __init__(self):
        pass
    
    def clean_text(self, text: str) -> str:
        """Clean conversation artifacts"""
        patterns = [
            r'Skip to content\s*', r'This is a copy of a conversation.*?\n',
            r'Report conversation\s*', r'You said:\s*', r'ChatGPT said:\s*',
            r'Attach\s*', r'Search\s*', r'Study\s*', r'Voice\s*',
            r'No file chosen\s*', r'ChatGPT can make mistakes.*?\n'
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        return re.sub(r'\s+', ' ', text).strip()
    
    def calculate_semantic_similarity(self, original: str, summary: str) -> float:
        """
        Refactored: Focuses on 'Key Concept Recall' rather than whole-doc vector similarity.
        Checks if the summary captures the unique 'DNA words' of the debate.
        """
        # 1. EXPANDED STOP WORDS (Same list as Topic Coverage for consistency)
        stop_words = {
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 
            'she', 'her', 'it', 'its', 'they', 'them', 'their', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'a', 'an', 'the', 'and',
            'or', 'but', 'if', 'then', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now',
            'said', 'ask', 'answer', 'question', 'chatgpt', 'user', 'transcript',
            'source', 'think', 'thought', 'mean', 'like', 'yeah', 'okay', 'hey',
            'hello', 'right', 'know', 'actually', 'basically', 'going', 'gonna',
            'wanna', 'got', 'say', 'saying', 'tell', 'thing', 'things', 'point',
            'really', 'maybe', 'bit', 'lot', 'way', 'make', 'see', 'look', 'come',
            'coming', 'go', 'goes', 'went', 'take', 'took', 'talk', 'talking',
            'let', 'us', 'test', 'testing', 'shubidubidu', 'sorry', 'kind', 'feel',
            'believe', 'agree', 'disagree', 'perspective', 'stance', 'view', 'opinion',
            'response', 'reply', 'words', 'start', 'end'
        }

        # Helper to extract meaningful words
        orig_clean = self.clean_text(original).lower()
        summ_clean = self.clean_text(summary).lower()
        
        # Use helper to get clean lists without stopwords
        def get_tokens(text):
            return [w for w in re.findall(r'\b\w+\b', text) 
                    if w not in stop_words and len(w) > 2]

        orig_tokens_list = get_tokens(orig_clean)
        summ_tokens_set = set(get_tokens(summ_clean))

        if not orig_tokens_list or not summ_tokens_set:
            return 0.0

        if sklearn_available:
            try:
                # 2. EXTRACT "DNA WORDS" (TF-IDF on Original Only)
                # We want to know what makes the TRANSCRIPT unique.
                vectorizer = TfidfVectorizer(
                    stop_words=list(stop_words),
                    max_features=50, # Only look at the top 50 concepts
                    norm='l2'
                )
                
                # Fit ONLY on the original text to define the vocabulary of importance
                tfidf_matrix = vectorizer.fit_transform([" ".join(orig_tokens_list)])
                feature_names = vectorizer.get_feature_names_out()
                
                # Get the top 20 highest scoring words
                scores = zip(feature_names, tfidf_matrix.toarray()[0])
                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
                top_concepts = [w for w, score in sorted_scores[:20]]
                
                if not top_concepts: return 0.0

                # 3. CALCULATE RETENTION (Did the summary keep them?)
                hits = 0
                for concept in top_concepts:
                    # Direct Match
                    if concept in summ_tokens_set:
                        hits += 1
                    # Soft Match (e.g. 'laws' matches 'law')
                    elif any(concept in t or t in concept for t in summ_tokens_set): 
                        hits += 0.5
                
                # 4. GRADE ON A CURVE
                # A summary is NOT a copy. If you capture 80% of the top keywords, that is an "A".
                # We normalize so that 0.65 retention = 1.0 score.
                retention_rate = hits / len(top_concepts)
                final_score = min(1.0, retention_rate / 0.8)
                
                return final_score

            except Exception as e:
                print(f"Fallback due to error: {e}")
                return self._fallback_similarity(orig_tokens_list, summ_tokens_set)
        else:
            return self._fallback_similarity(orig_tokens_list, summ_tokens_set)

    def _fallback_similarity(self, orig_tokens_list, summ_tokens_set):
        """
        Fallback logic that doesn't punish conciseness.
        """
        orig_set = set(orig_tokens_list)
        if not orig_set: return 0.0
        
        intersection = orig_set.intersection(summ_tokens_set)
        
        # Precision: "Of the words you wrote, how many were relevant?"
        # (Avoids dividing by the massive original length)
        precision = len(intersection) / len(summ_tokens_set) if summ_tokens_set else 0
        
        # Recall: "Did you capture enough content?"
        # We expect a summary to be ~5% of the original size.
        # If you captured 5% of the unique original words, that's a perfect summary.
        target_count = int(len(orig_set) * 0.05)
        recall = len(intersection) / max(1, target_count)
        
        # Weighted Score
        return (precision * 0.4) + (min(1.0, recall) * 0.6)
    
    def assess_topic_coverage(self, original: str, summary: str) -> float:
        """
        Refactored: Enforces a 'Minimum Breadth' floor.
        Prevents short summaries from getting 100% just by hitting 4-5 common words.
        """
        # 1. EXPANDED STOP WORDS
        stop_words = {
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 
            'she', 'her', 'it', 'its', 'they', 'them', 'their', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'a', 'an', 'the', 'and',
            'or', 'but', 'if', 'then', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now',
            'people', 'time', 'year', 'years', 'day',
            'said', 'ask', 'answer', 'question', 'chatgpt', 'user', 'transcript',
            'source', 'think', 'thought', 'mean', 'like', 'yeah', 'okay', 'hey',
            'hello', 'right', 'know', 'actually', 'basically', 'going', 'gonna',
            'wanna', 'got', 'say', 'saying', 'tell', 'thing', 'things', 'point',
            'really', 'maybe', 'bit', 'lot', 'way', 'make', 'see', 'look', 'come',
            'coming', 'go', 'goes', 'went', 'take', 'took', 'talk', 'talking',
            'let', 'us', 'test', 'testing', 'shubidubidu', 'sorry', 'kind', 'feel',
            'believe', 'agree', 'disagree', 'perspective', 'stance', 'view', 'opinion',
            'response', 'reply', 'words', 'start', 'end'
        }

        # Helper to extract meaningful words
        orig_clean = self.clean_text(original).lower()
        summ_clean = self.clean_text(summary).lower()

        def get_tokens(text):
            return [w for w in re.findall(r'\b\w+\b', text) 
                    if w not in stop_words and len(w) > 2]

        orig_words = get_tokens(orig_clean)
        summ_words = set(get_tokens(summ_clean))
        
        if not orig_words: return 0.0

        # 2. IDENTIFY SIGNATURE TOPICS
        orig_freq = Counter(orig_words)
        # We increase the pool to Top 30 to ensure we catch enough specific nouns
        signature_topics = [word for word, count in orig_freq.most_common(30)]
        
        if not signature_topics: return 0.0

        # 3. DYNAMIC GOAL SETTING (With Stricter Floor)
        # "Minimum Breadth": Regardless of brevity, you must cover at least 10 key topics 
        # (or 50% of the signature list) to get a high score.
        summary_word_count = len(summ_words)
        
        # Base target: 1 topic per 4 summary words
        density_target = int(summary_word_count / 4)
        
        # The Floor: Require at least 20 topics (unless the transcript itself is tiny)
        min_floor = min(20, len(signature_topics))
        
        # The Target is the HIGHER of density or floor.
        # This prevents a 20-word summary (density target ~3) from getting an easy pass.
        # It must now hit 'min_floor' (20) to get 100%.
        target_topic_count = min(len(signature_topics), max(min_floor, density_target))
        
        # 4. CALCULATE HIT RATE
        hits = 0
        for topic in signature_topics:
            match_found = False
            if topic in summ_words:
                match_found = True
            elif len(topic) > 4:
                root = topic[:4]
                if any(w.startswith(root) for w in summ_words):
                    match_found = True
            
            if match_found:
                hits += 1

        raw_score = hits / target_topic_count
        return min(1.0, raw_score)
    
    def check_for_contradictions(self, original: str, summary: str) -> float:
        """Check for obvious factual contradictions in meaning"""
        orig_clean = self.clean_text(original).lower()
        summ_clean = summary.lower()
        
        # Start with perfect accuracy
        accuracy = 1.0
        
        # Check for contradictory sentiment/tone claims
        sentiment_pairs = [
            (['positive', 'good', 'supportive', 'helpful', 'respectful', 'balanced'], 
             ['negative', 'bad', 'hostile', 'harmful', 'disrespectful', 'one-sided']),
            (['agree', 'consensus', 'harmony', 'understanding'], 
             ['disagree', 'conflict', 'tension', 'misunderstanding']),
            (['successful', 'effective', 'working'], 
             ['failed', 'ineffective', 'broken'])
        ]
        
        for positive_words, negative_words in sentiment_pairs:
            # If summary claims one sentiment but original suggests opposite
            summ_positive = any(word in summ_clean for word in positive_words)
            summ_negative = any(word in summ_clean for word in negative_words)
            orig_positive = any(word in orig_clean for word in positive_words)
            orig_negative = any(word in orig_clean for word in negative_words)
            
            # Check for clear contradictions
            if summ_positive and not orig_positive and orig_negative:
                accuracy -= 0.2  # Summary claims positive when original is negative
            elif summ_negative and not orig_negative and orig_positive:
                accuracy -= 0.2  # Summary claims negative when original is positive
        
        return max(0.0, accuracy)
    
    def grade_summary(self, original_file: str, summary_file: str) -> Dict[str, Any]:
        """Grade summary focusing purely on content recall"""
        
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = f.read()
        except Exception as e:
            return {"error": f"Error reading files: {e}"}
        
        # Calculate the three main aspects of content recall
        semantic_similarity = self.calculate_semantic_similarity(original, summary)
        topic_coverage = self.assess_topic_coverage(original, summary)
        factual_accuracy = self.check_for_contradictions(original, summary)
        
        # Human summary adjustments - humans naturally paraphrase more than AI
        # Boost scores significantly to account for natural human summarization patterns
        human_adjusted_semantic = min(1.0, semantic_similarity * 2.4)  # Increased boost for paraphrasing
        human_adjusted_topics = min(1.0, topic_coverage * 1.6)         # Increased boost for theme coverage
        
        # Weights heavily favor topic coverage over exact semantic matching
        weights = {
            'semantic_similarity': 0.25,  # Minimal emphasis on exact semantic matching
            'topic_coverage': 0.55,       # Heavy emphasis on covering main themes  
            'factual_accuracy': 0.2      # No major contradictions?
        }
        
        scores = {
            'semantic_similarity': human_adjusted_semantic,
            'topic_coverage': human_adjusted_topics,
            'factual_accuracy': factual_accuracy
        }
        
        # Calculate weighted overall score
        overall = sum(scores[metric] * weights[metric] for metric in scores)
        percentage = overall * 100
        
        # Realistic grading scale for human content recall
        def get_grade(pct):
            if pct >= 75: return 'A'    # Excellent human content recall
            elif pct >= 60: return 'B'  # Good human content recall
            elif pct >= 45: return 'C'  # Adequate human content recall
            elif pct >= 30: return 'D'  # Poor human content recall
            else: return 'F'            # Very poor human content recall
        
        return {
            'overall_percentage': round(percentage, 1),
            'overall_score': round(overall, 3),
            'letter_grade': get_grade(percentage),
            'approach': 'Human-Adjusted Content Recall - Natural Paraphrasing Expected',
            'breakdown': {
                metric: {
                    'score': round(scores[metric], 3),
                    'weight': weights[metric],
                    'contribution': round(scores[metric] * weights[metric] * 100, 1)
                } for metric in scores
            },
            'summary_stats': {
                'original_words': len(self.clean_text(original).split()),
                'summary_words': len(summary.split()),
                'compression_ratio': round(len(summary.split()) / len(self.clean_text(original).split()), 3) if original.split() else 0,
                'sklearn_available': sklearn_available
            },
            'files': {
                'original_file': original_file,
                'summary_file': summary_file
            }
        }

def main():
    parser = argparse.ArgumentParser(description="Content Recall Grader + Length-Adjusted Evaluation")
    parser.add_argument("original_file", help="Original text file")
    parser.add_argument("summary_file", help="Summary text file")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--advanced-eval", action="store_true", help="Run length-adjusted evaluation")
    
    args = parser.parse_args()
    
    grader = ContentRecallGrader()
    result = grader.grade_summary(args.original_file, args.summary_file)
    advanced_result = None
    if args.advanced_eval and 'error' not in result:
        try:
            original_text = Path(args.original_file).read_text(encoding='utf-8')
            summary_text = Path(args.summary_file).read_text(encoding='utf-8')
            advanced_result = summarize_evaluation(original_text, summary_text)
        except Exception as e:
            advanced_result = {"error": f"Advanced evaluation failed: {e}", "final_score": 0.0}
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    if args.json and args.advanced_eval:
        print(json.dumps({'content_recall': result, 'length_adjusted': advanced_result}, indent=2))
        return
    if args.json:
        print(json.dumps(result, indent=2))
        return
    else:
        print("=" * 60)
        print("HUMAN-ADJUSTED CONTENT RECALL ASSESSMENT")
        print("=" * 60)
        print(f"Overall Score: {result['overall_percentage']}% (Grade: {result['letter_grade']})")
        print(f"Approach: {result['approach']}")
        if advanced_result:
            if 'error' in advanced_result:
                print(f"\nAdvanced Eval Error: {advanced_result['error']}")
            else:
                print("\nLENGTH-ADJUSTED METRICS:")
                print(f"Final Score: {advanced_result['final_score']:.4f}")
                print(f"Topic Coverage (adjusted): {advanced_result['topic_coverage']:.4f} (raw {advanced_result['coverage_raw']:.4f}; penalty {advanced_result['coverage_penalty']:.4f})")
                print(f"Global Similarity: {advanced_result['global_similarity']:.4f}")
                print(f"Redundancy Penalty: {advanced_result['redundancy_penalty']:.4f} (redundancy {advanced_result['redundancy']:.4f})")
                print(f"Info Density: {advanced_result['info_density']:.6f} (info score {advanced_result['info_score']:.4f})")
                print(f"Length Factor: {advanced_result['length_factor']:.4f}")
                print(f"Concepts Covered: {advanced_result['concepts_covered']} / {advanced_result['concept_count']}")
        print()
        
        print("BREAKDOWN (Human-Adjusted):")
        print("-" * 40)
        breakdown = result['breakdown']
        for metric, data in breakdown.items():
            name = metric.replace('_', ' ').title()
            print(f"{name:20} | {data['score']:.3f} | Weight: {data['weight']:.0%} | Contrib: {data['contribution']:.1f}%")
        
        print()
        print("SUMMARY ANALYSIS:")
        print("-" * 40)
        stats = result['summary_stats']
        print(f"Original length: {stats['original_words']} words")
        print(f"Summary length: {stats['summary_words']} words")
        print(f"Compression ratio: {stats['compression_ratio']:.1%}")
        
        if not stats['sklearn_available']:
            print("ℹ️  Using word-overlap fallback (sklearn not available)")
        
        print()
        print("ASSESSMENT:")
        print("-" * 40)
        if breakdown['semantic_similarity']['score'] >= 0.7:
            print("✅ Good semantic similarity - captures the meaning")
        elif breakdown['semantic_similarity']['score'] >= 0.5:
            print("⚠️  Moderate semantic similarity")
        else:
            print("❌ Poor semantic similarity")
            
        if breakdown['topic_coverage']['score'] >= 0.7:
            print("✅ Excellent topic coverage")
        elif breakdown['topic_coverage']['score'] >= 0.5:
            print("⚠️  Good topic coverage with some gaps")
        else:
            print("❌ Poor topic coverage - missing key themes")
            
        if breakdown['factual_accuracy']['score'] >= 0.9:
            print("✅ No contradictions detected")
        elif breakdown['factual_accuracy']['score'] >= 0.7:
            print("⚠️  Minor consistency issues")
        else:
            print("❌ Significant contradictions detected")

if __name__ == "__main__":
    main()