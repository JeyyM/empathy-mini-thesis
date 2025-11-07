#!/usr/bin/env python3
"""
general_recall_grader.py -- General-purpose recall accuracy grader
Dynamically analyzes any text without hardcoded patterns
"""

import re
import json
import argparse
from collections import Counter
from typing import Dict, List, Any, Tuple, Set

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    sklearn_available = True
except ImportError:
    sklearn_available = False

class GeneralRecallGrader:
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
        """Clean conversation artifacts and normalize text"""
        # Remove ChatGPT conversation UI elements
        patterns = [
            r'Skip to content\s*', r'This is a copy of a conversation.*?\n',
            r'Report conversation\s*', r'You said:\s*', r'ChatGPT said:\s*',
            r'Attach\s*', r'Search\s*', r'Study\s*', r'Voice\s*',
            r'No file chosen\s*', r'ChatGPT can make mistakes.*?\n'
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_key_phrases(self, text: str, min_freq: int = 2) -> List[str]:
        """Dynamically extract important phrases from text"""
        text_clean = self.clean_text(text).lower()
        
        # Extract noun phrases and important multi-word expressions
        phrases = []
        
        # Find repeated multi-word phrases (2-4 words)
        for n in range(2, 5):
            words = text_clean.split()
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                # Filter out phrases that are mostly stop words
                phrase_words = phrase.split()
                content_words = [w for w in phrase_words if w not in self.stop_words and len(w) > 2]
                if len(content_words) >= n // 2:  # At least half should be content words
                    phrases.append(phrase)
        
        # Count phrase frequency and return those above threshold
        phrase_counts = Counter(phrases)
        key_phrases = [phrase for phrase, count in phrase_counts.items() if count >= min_freq]
        
        # Add single important words that appear frequently
        words = [w for w in text_clean.split() if w not in self.stop_words and len(w) > 3]
        word_counts = Counter(words)
        important_words = [word for word, count in word_counts.most_common(30) if count >= min_freq]
        
        return key_phrases + important_words
    
    def extract_semantic_concepts(self, text: str) -> Dict[str, List[str]]:
        """Dynamically identify semantic concepts and themes"""
        text_clean = self.clean_text(text).lower()
        
        concepts = {
            'entities': [],      # People, organizations, concepts
            'actions': [],       # What happened, what was done
            'themes': [],        # Recurring themes
            'positions': [],     # Different viewpoints or stances
            'outcomes': []       # Results, conclusions, decisions
        }
        
        # Find entities (capitalized words/phrases in original)
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(entity_pattern, text)
        concepts['entities'] = list(set(entities))
        
        # Find action patterns (verbs followed by objects)
        action_patterns = [
            r'\b(?:discuss|talk|debate|argue|explain|describe|mention|address|consider|examine)\w*\s+\w+',
            r'\b(?:support|oppose|defend|criticize|question|challenge|acknowledge|recognize)\w*\s+\w+',
            r'\b(?:protect|prevent|allow|permit|restrict|limit|enable|encourage)\w*\s+\w+'
        ]
        
        for pattern in action_patterns:
            actions = re.findall(pattern, text_clean)
            concepts['actions'].extend(actions)
        
        # Find thematic words that repeat (potential themes)
        words = [w for w in text_clean.split() if w not in self.stop_words and len(w) > 4]
        word_freq = Counter(words)
        themes = [word for word, count in word_freq.most_common(20) if count >= 3]
        concepts['themes'] = themes
        
        # Find position indicators
        position_indicators = [
            r'\bi think\s+\w+', r'\bi believe\s+\w+', r'\bi feel\s+\w+',
            r'\bmy view\s+\w+', r'\bin my opinion\s+\w+',
            r'\bthe argument\s+\w+', r'\bthe position\s+\w+',
            r'\bthe concern\s+\w+', r'\bthe worry\s+\w+'
        ]
        
        for pattern in position_indicators:
            positions = re.findall(pattern, text_clean)
            concepts['positions'].extend(positions)
        
        # Find outcome indicators
        outcome_patterns = [
            r'\bthe result\s+\w+', r'\bthe conclusion\s+\w+',
            r'\bin the end\s+\w+', r'\bfinally\s+\w+',
            r'\boverall\s+\w+', r'\bto summarize\s+\w+'
        ]
        
        for pattern in outcome_patterns:
            outcomes = re.findall(pattern, text_clean)
            concepts['outcomes'].extend(outcomes)
        
        return concepts
    
    def calculate_concept_overlap(self, orig_concepts: Dict, summ_concepts: Dict) -> float:
        """Calculate how well summary concepts match original concepts - allow for reasonable compression"""
        total_overlap = 0.0
        total_possible = 0.0
        
        for category in orig_concepts:
            orig_set = set(orig_concepts[category])
            summ_set = set(summ_concepts[category])
            
            if orig_set:
                # Direct overlap
                direct_overlap = len(orig_set.intersection(summ_set))
                
                # Partial overlap - check for semantic similarity
                partial_overlap = 0
                for orig_concept in orig_set:
                    if orig_concept not in summ_set:  # Not already counted
                        for summ_concept in summ_set:
                            if self._concepts_semantically_related(orig_concept, summ_concept):
                                partial_overlap += 0.5  # Partial credit
                                break
                
                category_overlap = direct_overlap + partial_overlap
                total_overlap += category_overlap
                # Be more lenient - don't expect all concepts to be preserved in a summary
                total_possible += min(len(orig_set), len(orig_set) * 0.6)  # Expect 60% max preservation
        
        return total_overlap / total_possible if total_possible > 0 else 0.0
    
    def _concepts_semantically_related(self, concept1: str, concept2: str) -> bool:
        """Check if two concepts are semantically related"""
        # Simple semantic relatedness check
        words1 = set(w for w in concept1.split() if w not in self.stop_words and len(w) > 2)
        words2 = set(w for w in concept2.split() if w not in self.stop_words and len(w) > 2)
        
        if not words1 or not words2:
            return False
        
        # Check for shared words or similar themes
        overlap = len(words1.intersection(words2))
        return overlap > 0  # Any shared content words suggest relation
    
    def calculate_semantic_similarity(self, original: str, summary: str) -> float:
        """Calculate semantic similarity using TF-IDF or fallback to phrase matching"""
        orig_clean = self.clean_text(original)
        summ_clean = self.clean_text(summary)
        
        if not sklearn_available:
            # Fallback: phrase overlap method
            orig_phrases = set(self.extract_key_phrases(orig_clean))
            summ_phrases = set(self.extract_key_phrases(summ_clean))
            
            if not orig_phrases:
                return 0.0
            
            overlap = len(orig_phrases.intersection(summ_phrases))
            return overlap / len(orig_phrases)
        
        try:
            # Use TF-IDF for semantic similarity
            vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=1
            )
            
            corpus = [orig_clean, summ_clean]
            if len(' '.join(corpus).split()) < 10:  # Too short for TF-IDF
                return self.calculate_semantic_similarity(original, summary)  # Use fallback
            
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
            
        except Exception:
            # Fallback to phrase matching
            return self.calculate_semantic_similarity(original, summary)
    
    def assess_content_recall(self, original: str, summary: str) -> float:
        """Assess how well the summary recalls the main content - focus on meaning preservation"""
        # Extract key elements from both texts
        orig_phrases = self.extract_key_phrases(original)
        summ_phrases = self.extract_key_phrases(summary)
        
        orig_concepts = self.extract_semantic_concepts(original)
        summ_concepts = self.extract_semantic_concepts(summary)
        
        # Focus on whether IMPORTANT content is preserved, not total volume
        orig_phrase_set = set(orig_phrases[:20])  # Focus on top 20 most important phrases
        summ_phrase_set = set(summ_phrases)
        
        if not orig_phrase_set:
            phrase_recall = 0.5  # Default if no key phrases identified
        else:
            # Allow for paraphrasing by checking for partial matches
            matched_phrases = 0
            for orig_phrase in orig_phrase_set:
                orig_words = set(orig_phrase.split())
                # Check both exact and semantic matches
                for summ_phrase in summ_phrase_set:
                    summ_words = set(summ_phrase.split())
                    # Consider match if significant word overlap OR semantic similarity
                    overlap = len(orig_words.intersection(summ_words))
                    if overlap >= min(2, len(orig_words) // 2):
                        matched_phrases += 1
                        break
                else:
                    # Check for semantic equivalence (different wording, same meaning)
                    for summ_phrase in summ_phrase_set:
                        # Simple semantic check - if they share conceptual words
                        if self._phrases_semantically_similar(orig_phrase, summ_phrase):
                            matched_phrases += 0.7  # Partial credit for paraphrasing
                            break
            
            phrase_recall = min(1.0, matched_phrases / len(orig_phrase_set))
        
        # Calculate concept recall - focus on core themes being preserved
        concept_recall = self.calculate_concept_overlap(orig_concepts, summ_concepts)
        
        # Weight towards concept recall (meaning) over exact phrase recall (wording)
        content_recall = 0.3 * phrase_recall + 0.7 * concept_recall
        
        return content_recall
    
    def _phrases_semantically_similar(self, phrase1: str, phrase2: str) -> bool:
        """Check if two phrases are semantically similar despite different wording"""
        words1 = set(w for w in phrase1.split() if w not in self.stop_words)
        words2 = set(w for w in phrase2.split() if w not in self.stop_words)
        
        if not words1 or not words2:
            return False
        
        # Check for synonym-like relationships or shared roots
        # Simple approach: if they share conceptual similarity
        overlap = len(words1.intersection(words2))
        min_overlap = min(len(words1), len(words2))
        
        # If at least 30% overlap in content words, consider similar
        return overlap >= max(1, min_overlap * 0.3)
    
    def assess_message_understanding(self, original: str, summary: str) -> float:
        """Assess if the core message/meaning was understood - focus on content only"""
        # Use semantic similarity as the only indicator - no bonus for summary language
        semantic_sim = self.calculate_semantic_similarity(original, summary)
        
        return semantic_sim
    
    def assess_accuracy(self, original: str, summary: str) -> float:
        """Focus only on contradictions - ignore overgeneralization concerns"""
        orig_clean = self.clean_text(original).lower()
        summ_clean = summary.lower()
        
        # Start with perfect accuracy
        accuracy = 1.0
        
        # Check for obvious contradictions in meaning/sentiment
        # If summary mentions something is "positive" but original suggests "negative" etc.
        polarity_words = {
            'positive': ['good', 'great', 'excellent', 'successful', 'effective', 'supportive', 'helpful'],
            'negative': ['bad', 'poor', 'failed', 'unsuccessful', 'ineffective', 'harmful', 'problematic'],
            'agreement': ['agreed', 'consensus', 'harmony', 'aligned', 'supportive', 'understanding'],
            'disagreement': ['disagreed', 'conflict', 'tension', 'opposed', 'hostile', 'argumentative']
        }
        
        for polarity, words in polarity_words.items():
            if any(word in summ_clean for word in words):
                # Check if original supports this characterization
                opposite_polarity = {
                    'positive': 'negative', 'negative': 'positive',
                    'agreement': 'disagreement', 'disagreement': 'agreement'
                }.get(polarity)
                
                if opposite_polarity and any(word in orig_clean for word in polarity_words[opposite_polarity]):
                    # Clear contradiction in meaning - penalty
                    accuracy -= 0.15
        
        return max(0.0, accuracy)
    
    def grade_summary(self, original_file: str, summary_file: str) -> Dict[str, Any]:
        """Grade summary focusing on general recall accuracy and understanding"""
        
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = f.read()
        except Exception as e:
            return {"error": f"Error reading files: {e}"}
        
        # Calculate general accuracy metrics
        content_recall = self.assess_content_recall(original, summary)
        message_understanding = self.assess_message_understanding(original, summary)
        factual_accuracy = self.assess_accuracy(original, summary)
        
        # Get additional analysis data
        orig_phrases = self.extract_key_phrases(original)
        summ_phrases = self.extract_key_phrases(summary)
        orig_concepts = self.extract_semantic_concepts(original)
        summ_concepts = self.extract_semantic_concepts(summary)
        
        # Weights focused purely on content recall accuracy
        weights = {
            'content_recall': 0.60,        # Primary focus: Did you remember the main content?
            'message_understanding': 0.30,  # Core meaning preserved despite different wording?
            'factual_accuracy': 0.10       # No contradictions in meaning?
        }
        
        scores = {
            'content_recall': content_recall,
            'message_understanding': message_understanding,
            'factual_accuracy': factual_accuracy
        }
        
        # Calculate weighted overall score
        overall = sum(scores[metric] * weights[metric] for metric in scores)
        percentage = overall * 100
        
        # Grade scale - more lenient since we only care about content recall
        def get_grade(pct):
            if pct >= 85: return 'A'    # Excellent content recall
            elif pct >= 70: return 'B'  # Good content recall 
            elif pct >= 55: return 'C'  # Adequate content recall
            elif pct >= 40: return 'D'  # Poor content recall
            else: return 'F'            # Very poor content recall
        
        return {
            'overall_percentage': round(percentage, 1),
            'overall_score': round(overall, 3),
            'letter_grade': get_grade(percentage),
            'approach': 'Pure Content Recall Focus - Meaning Over Wording',
            'breakdown': {
                metric: {
                    'score': round(scores[metric], 3),
                    'weight': weights[metric],
                    'contribution': round(scores[metric] * weights[metric] * 100, 1)
                } for metric in scores
            },
            'analysis_details': {
                'original_key_phrases': len(orig_phrases),
                'summary_key_phrases': len(summ_phrases),
                'original_concepts': {k: len(v) for k, v in orig_concepts.items()},
                'summary_concepts': {k: len(v) for k, v in summ_concepts.items()},
                'sklearn_available': sklearn_available
            },
            'summary_stats': {
                'original_words': len(self.clean_text(original).split()),
                'summary_words': len(summary.split()),
                'compression_ratio': round(len(summary.split()) / len(self.clean_text(original).split()), 3) if original.split() else 0
            }
        }

def main():
    parser = argparse.ArgumentParser(description="General Recall Accuracy Grader - Dynamic Analysis")
    parser.add_argument("original_file", help="Original text file")
    parser.add_argument("summary_file", help="Summary text file")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    
    args = parser.parse_args()
    
    grader = GeneralRecallGrader()
    result = grader.grade_summary(args.original_file, args.summary_file)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("="*70)
        print("PURE CONTENT RECALL ASSESSMENT")
        print("="*70)
        print(f"Overall Score: {result['overall_percentage']}% (Grade: {result['letter_grade']})")
        print(f"Approach: {result['approach']}")
        print()
        
        print("CONTENT RECALL BREAKDOWN:")
        print("-" * 50)
        breakdown = result['breakdown']
        for metric, data in breakdown.items():
            name = metric.replace('_', ' ').title()
            print(f"{name:20} | {data['score']:.3f} | Weight: {data['weight']:.0%} | Contribution: {data['contribution']:.1f}%")
        
        print()
        print("DYNAMIC ANALYSIS RESULTS:")
        print("-" * 50)
        analysis = result['analysis_details']
        print(f"Key phrases identified in original: {analysis['original_key_phrases']}")
        print(f"Key phrases found in summary: {analysis['summary_key_phrases']}")
        
        orig_concept_total = sum(analysis['original_concepts'].values())
        summ_concept_total = sum(analysis['summary_concepts'].values())
        print(f"Concepts identified in original: {orig_concept_total}")
        print(f"Concepts found in summary: {summ_concept_total}")
        
        if not analysis['sklearn_available']:
            print("ℹ️  Using phrase-matching fallback (sklearn not available)")
        
        print()
        print("ASSESSMENT:")
        print("-" * 50)
        if breakdown['content_recall']['score'] >= 0.8:
            print("✅ Excellent content recall")
        elif breakdown['content_recall']['score'] >= 0.6:
            print("⚠️  Good content recall with some gaps")
        else:
            print("❌ Poor content recall - missing key information")
            
        if breakdown['message_understanding']['score'] >= 0.8:
            print("✅ Strong message understanding")
        elif breakdown['message_understanding']['score'] >= 0.6:
            print("⚠️  Adequate message understanding")
        else:
            print("❌ Weak message understanding")
            
        if breakdown['factual_accuracy']['score'] >= 0.9:
            print("✅ Highly accurate")
        elif breakdown['factual_accuracy']['score'] >= 0.7:
            print("⚠️  Mostly accurate with minor issues")
        else:
            print("❌ Accuracy concerns detected")

if __name__ == "__main__":
    main()