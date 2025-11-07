#!/usr/bin/env python3
"""
balanced_recall_grader.py -- Balanced general-purpose recall grader
No hardcoded patterns, but realistic expectations for summarization
"""

import re
import json
import argparse
from collections import Counter
from typing import Dict, List, Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sklearn_available = True
except ImportError:
    sklearn_available = False

class BalancedRecallGrader:
    def __init__(self):
        self.stop_words = {
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 
            'she', 'her', 'it', 'its', 'they', 'them', 'their', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'a', 'an', 'the', 'and',
            'or', 'but', 'if', 'then', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'said'
        }
    
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
    
    def extract_important_words(self, text: str, top_n: int = 20) -> List[str]:
        """Extract most important content words"""
        text_clean = self.clean_text(text).lower()
        words = [w for w in re.findall(r'\b\w+\b', text_clean) 
                if w not in self.stop_words and len(w) > 3]
        
        word_freq = Counter(words)
        return [word for word, count in word_freq.most_common(top_n)]
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract meaningful concepts (2-3 word phrases + important single words)"""
        text_clean = self.clean_text(text).lower()
        concepts = []
        
        # Get important single words
        important_words = self.extract_important_words(text, 15)
        concepts.extend(important_words)
        
        # Get meaningful 2-3 word phrases
        words = text_clean.split()
        for length in [2, 3]:
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i+length])
                phrase_words = phrase.split()
                
                # Keep phrases with at least one important content word
                if any(word in important_words for word in phrase_words):
                    concepts.append(phrase)
        
        # Remove duplicates and return most frequent
        concept_freq = Counter(concepts)
        return [concept for concept, count in concept_freq.most_common(25)]
    
    def calculate_semantic_overlap(self, original: str, summary: str) -> float:
        """Calculate semantic overlap allowing for paraphrasing"""
        if not sklearn_available:
            return self.calculate_word_overlap(original, summary)
        
        try:
            orig_clean = self.clean_text(original)
            summ_clean = self.clean_text(summary)
            
            if len(orig_clean.split()) < 10 or len(summ_clean.split()) < 5:
                return self.calculate_word_overlap(original, summary)
            
            vectorizer = TfidfVectorizer(
                max_features=300,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=1
            )
            
            corpus = [orig_clean, summ_clean]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
            
        except Exception:
            return self.calculate_word_overlap(original, summary)
    
    def calculate_word_overlap(self, original: str, summary: str) -> float:
        """Fallback word overlap calculation"""
        orig_words = set(self.extract_important_words(original, 30))
        summ_words = set(self.extract_important_words(summary, 30))
        
        if not orig_words:
            return 0.0
        
        overlap = len(orig_words & summ_words)
        return overlap / len(orig_words)
    
    def assess_concept_preservation(self, original: str, summary: str) -> float:
        """Check if key concepts are preserved (allowing paraphrasing)"""
        orig_concepts = self.extract_key_concepts(original)
        summary_lower = summary.lower()
        
        if not orig_concepts:
            return 0.5
        
        preserved_count = 0
        for concept in orig_concepts:
            concept_words = concept.split()
            
            # Exact match
            if concept in summary_lower:
                preserved_count += 1
                continue
            
            # Partial match (at least half the words present)
            matches = sum(1 for word in concept_words if word in summary_lower)
            if matches >= len(concept_words) // 2 and matches >= 1:
                preserved_count += 0.5
        
        return preserved_count / len(orig_concepts)
    
    def assess_message_understanding(self, original: str, summary: str) -> float:
        """Assess overall message understanding"""
        # Use semantic similarity as base
        semantic_sim = self.calculate_semantic_overlap(original, summary)
        
        # Check for summary quality indicators
        summary_lower = summary.lower()
        
        # Good summary indicators
        good_indicators = [
            r'\boverall\b', r'\bin summary\b', r'\bto sum up\b',
            r'\bthe main\b', r'\bthe key\b', r'\bwhat.*(?:shows|means)\b',
            r'\bthe conversation\b', r'\bthe discussion\b'
        ]
        
        has_summary_language = any(re.search(pattern, summary_lower) 
                                  for pattern in good_indicators)
        
        # Meta-understanding bonus
        meta_bonus = 0.15 if has_summary_language else 0.0
        
        # Length appropriateness (summaries shouldn't be too short or too long)
        orig_words = len(self.clean_text(original).split())
        summ_words = len(summary.split())
        
        if orig_words > 0:
            compression = summ_words / orig_words
            if 0.05 <= compression <= 0.4:  # Reasonable compression range
                length_bonus = 0.1
            elif compression < 0.05:  # Too short
                length_bonus = -0.2
            else:  # Too long
                length_bonus = -0.1
        else:
            length_bonus = 0.0
        
        return min(1.0, semantic_sim + meta_bonus + length_bonus)
    
    def assess_accuracy(self, original: str, summary: str) -> float:
        """Check for obvious inaccuracies or contradictions"""
        accuracy = 1.0
        
        # Check for unsupported strong claims
        summary_lower = summary.lower()
        
        # Strong absolute statements that could be wrong
        absolute_patterns = [
            r'\b(?:always|never|all|none|every|no one)\b',
            r'\b(?:definitely|certainly|absolutely|completely)\b'
        ]
        
        strong_claims = 0
        for pattern in absolute_patterns:
            strong_claims += len(re.findall(pattern, summary_lower))
        
        # Small penalty for too many strong claims (summaries should be moderate)
        if strong_claims > 2:
            accuracy -= 0.1
        
        # Check for obvious contradictions with emotional tone
        orig_lower = self.clean_text(original).lower()
        
        # If original shows balanced discussion but summary claims one-sidedness
        if ('balanced' in orig_lower or 'respectful' in orig_lower) and \
           ('one-sided' in summary_lower or 'biased' in summary_lower):
            # This could be accurate (describing perceived bias), so small penalty only
            accuracy -= 0.05
        
        return max(0.0, accuracy)
    
    def grade_summary(self, original_file: str, summary_file: str) -> Dict[str, Any]:
        """Grade summary with balanced expectations"""
        
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = f.read()
        except Exception as e:
            return {"error": f"Error reading files: {e}"}
        
        # Calculate balanced metrics
        concept_preservation = self.assess_concept_preservation(original, summary)
        message_understanding = self.assess_message_understanding(original, summary)
        accuracy = self.assess_accuracy(original, summary)
        
        # Balanced weights that recognize summarization realities
        weights = {
            'concept_preservation': 0.40,   # Key concepts preserved (allowing paraphrasing)
            'message_understanding': 0.40,  # Overall understanding and semantic similarity
            'accuracy': 0.20                # Basic accuracy without contradictions
        }
        
        scores = {
            'concept_preservation': concept_preservation,
            'message_understanding': message_understanding,
            'accuracy': accuracy
        }
        
        # Calculate weighted overall score
        overall = sum(scores[metric] * weights[metric] for metric in scores)
        percentage = overall * 100
        
        # Realistic grading scale for summaries
        def get_grade(pct):
            if pct >= 80: return 'A'     # Excellent summary
            elif pct >= 65: return 'B'   # Good summary 
            elif pct >= 50: return 'C'   # Adequate summary
            elif pct >= 35: return 'D'   # Poor summary
            else: return 'F'             # Very poor summary
        
        # Get analysis details
        orig_concepts = self.extract_key_concepts(original)
        summ_words = self.extract_important_words(summary, 20)
        semantic_sim = self.calculate_semantic_overlap(original, summary)
        
        return {
            'overall_percentage': round(percentage, 1),
            'overall_score': round(overall, 3),
            'letter_grade': get_grade(percentage),
            'approach': 'Balanced Dynamic Analysis',
            'breakdown': {
                metric: {
                    'score': round(scores[metric], 3),
                    'weight': weights[metric],
                    'contribution': round(scores[metric] * weights[metric] * 100, 1)
                } for metric in scores
            },
            'analysis_details': {
                'semantic_similarity': round(semantic_sim, 3),
                'key_concepts_identified': len(orig_concepts),
                'concepts_sample': orig_concepts[:8],
                'summary_key_words': summ_words[:8],
                'sklearn_available': sklearn_available
            },
            'summary_stats': {
                'original_words': len(self.clean_text(original).split()),
                'summary_words': len(summary.split()),
                'compression_ratio': round(len(summary.split()) / len(self.clean_text(original).split()), 3) if original.split() else 0
            }
        }

def main():
    parser = argparse.ArgumentParser(description="Balanced Recall Grader - Realistic Summarization Assessment")
    parser.add_argument("original_file", help="Original text file")
    parser.add_argument("summary_file", help="Summary text file")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    
    args = parser.parse_args()
    
    grader = BalancedRecallGrader()
    result = grader.grade_summary(args.original_file, args.summary_file)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("="*70)
        print("BALANCED RECALL ACCURACY ASSESSMENT")
        print("="*70)
        print(f"Overall Score: {result['overall_percentage']}% (Grade: {result['letter_grade']})")
        print(f"Approach: {result['approach']}")
        print()
        
        print("BALANCED BREAKDOWN:")
        print("-" * 50)
        breakdown = result['breakdown']
        for metric, data in breakdown.items():
            name = metric.replace('_', ' ').title()
            print(f"{name:20} | {data['score']:.3f} | Weight: {data['weight']:.0%} | Contribution: {data['contribution']:.1f}%")
        
        print()
        print("ANALYSIS INSIGHTS:")
        print("-" * 50)
        analysis = result['analysis_details']
        print(f"Semantic similarity: {analysis['semantic_similarity']:.3f}")
        print(f"Key concepts in original: {analysis['key_concepts_identified']}")
        print(f"Concept sample: {', '.join(analysis['concepts_sample'])}")
        print(f"Summary key words: {', '.join(analysis['summary_key_words'])}")
        
        if not analysis['sklearn_available']:
            print("ℹ️  Using word-overlap fallback (sklearn not available)")
        
        print()
        print("SUMMARY QUALITY:")
        print("-" * 50)
        if breakdown['concept_preservation']['score'] >= 0.7:
            print("✅ Strong concept preservation")
        elif breakdown['concept_preservation']['score'] >= 0.4:
            print("⚠️  Moderate concept preservation")
        else:
            print("❌ Weak concept preservation")
            
        if breakdown['message_understanding']['score'] >= 0.7:
            print("✅ Good message understanding")
        elif breakdown['message_understanding']['score'] >= 0.4:
            print("⚠️  Basic message understanding")
        else:
            print("❌ Poor message understanding")
            
        if breakdown['accuracy']['score'] >= 0.9:
            print("✅ High accuracy")
        elif breakdown['accuracy']['score'] >= 0.7:
            print("⚠️  Generally accurate")
        else:
            print("❌ Accuracy issues")

if __name__ == "__main__":
    main()