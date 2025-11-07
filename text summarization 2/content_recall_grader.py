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
        """Calculate semantic overlap using TF-IDF cosine similarity"""
        orig_clean = self.clean_text(original)
        summ_clean = self.clean_text(summary)
        
        if not sklearn_available:
            # Fallback to simple word overlap
            orig_words = set(orig_clean.lower().split())
            summ_words = set(summ_clean.lower().split())
            if not orig_words:
                return 0.0
            overlap = len(orig_words.intersection(summ_words))
            return overlap / len(orig_words)
        
        try:
            # Use TF-IDF for better semantic understanding
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),  # Include 3-grams for better phrase understanding
                stop_words='english',
                min_df=1,
                lowercase=True
            )
            
            corpus = [orig_clean, summ_clean]
            
            if len(' '.join(corpus).split()) < 10:
                # Too short for meaningful TF-IDF
                orig_words = set(orig_clean.lower().split())
                summ_words = set(summ_clean.lower().split())
                if not orig_words:
                    return 0.0
                overlap = len(orig_words.intersection(summ_words))
                return overlap / len(orig_words)
            
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
            
        except Exception:
            # Fallback to word overlap
            orig_words = set(orig_clean.lower().split())
            summ_words = set(summ_clean.lower().split())
            if not orig_words:
                return 0.0
            overlap = len(orig_words.intersection(summ_words))
            return overlap / len(orig_words)
    
    def assess_topic_coverage(self, original: str, summary: str) -> float:
        """Assess if the main topics/themes are covered"""
        orig_clean = self.clean_text(original).lower()
        summ_clean = summary.lower()
        
        # Define stop words
        stop_words = {'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 
                     'she', 'her', 'it', 'its', 'they', 'them', 'their', 'this', 'that',
                     'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'can', 'a', 'an', 'the', 'and',
                     'or', 'but', 'if', 'then', 'because', 'as', 'until', 'while', 'of',
                     'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                     'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'said', 'also', 'just',
                     'like', 'about', 'from', 'into', 'than', 'more', 'most', 'some',
                     'any', 'here', 'there', 'when', 'where', 'why', 'how', 'what',
                     'who', 'which', 'whose', 'whom', 'get', 'got', 'getting', 'goes',
                     'going', 'went', 'come', 'came', 'coming', 'see', 'saw', 'seeing',
                     'know', 'knew', 'known', 'knowing', 'think', 'thought', 'thinking',
                     'way', 'ways', 'time', 'times', 'thing', 'things', 'people',
                     'person', 'make', 'made', 'making', 'take', 'took', 'taken',
                     'taking', 'use', 'used', 'using', 'work', 'worked', 'working'}
        
        orig_words = [w for w in re.findall(r'\b\w+\b', orig_clean) 
                     if w not in stop_words and len(w) > 3]
        
        # Get the most important topics (frequent meaningful words)
        word_freq = Counter(orig_words)
        important_topics = [word for word, count in word_freq.most_common(15)]
        
        if not important_topics:
            return 0.5  # Default
        
        # Check how many important topics are covered in the summary
        topics_covered = 0
        for topic in important_topics:
            if topic in summ_clean:
                topics_covered += 1
            else:
                # Check for related terms (simple stemming/similarity)
                topic_root = topic[:4] if len(topic) > 4 else topic
                if any(word.startswith(topic_root) for word in summ_clean.split()):
                    topics_covered += 0.7  # Partial credit for related words
        
        return min(1.0, topics_covered / len(important_topics))
    
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
        
        # Weights focused on content recall
        weights = {
            'semantic_similarity': 0.50,  # Does it capture the same meaning?
            'topic_coverage': 0.40,       # Does it cover the main topics?
            'factual_accuracy': 0.10      # No major contradictions?
        }
        
        scores = {
            'semantic_similarity': semantic_similarity,
            'topic_coverage': topic_coverage,
            'factual_accuracy': factual_accuracy
        }
        
        # Calculate weighted overall score
        overall = sum(scores[metric] * weights[metric] for metric in scores)
        percentage = overall * 100
        
        # Realistic grading scale for content recall
        def get_grade(pct):
            if pct >= 75: return 'A'    # Excellent content recall
            elif pct >= 60: return 'B'  # Good content recall
            elif pct >= 45: return 'C'  # Adequate content recall
            elif pct >= 30: return 'D'  # Poor content recall
            else: return 'F'            # Very poor content recall
        
        return {
            'overall_percentage': round(percentage, 1),
            'overall_score': round(overall, 3),
            'letter_grade': get_grade(percentage),
            'approach': 'Pure Content Recall - Meaning Over Wording',
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
            }
        }

def main():
    parser = argparse.ArgumentParser(description="Content Recall Grader - Focus on Meaning Preservation")
    parser.add_argument("original_file", help="Original text file")
    parser.add_argument("summary_file", help="Summary text file")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    
    args = parser.parse_args()
    
    grader = ContentRecallGrader()
    result = grader.grade_summary(args.original_file, args.summary_file)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("CONTENT RECALL ASSESSMENT")
        print("=" * 60)
        print(f"Overall Score: {result['overall_percentage']}% (Grade: {result['letter_grade']})")
        print(f"Approach: {result['approach']}")
        print()
        
        print("BREAKDOWN:")
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
        if breakdown['semantic_similarity']['score'] >= 0.6:
            print("✅ Good semantic similarity - captures the meaning")
        elif breakdown['semantic_similarity']['score'] >= 0.4:
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