#!/usr/bin/env python3
"""
improved_grader.py -- Enhanced Summary Quality Grader
Better suited for reflective and analytical summaries
"""

import argparse
import json
import re
import sys
from collections import Counter
from typing import List, Dict, Tuple, Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sklearn_available = True
except ImportError:
    sklearn_available = False

class ImprovedSummaryGrader:
    def __init__(self, summary_type="reflective"):
        """
        summary_type: 'extractive' (traditional) or 'reflective' (analytical/insight-based)
        """
        self.summary_type = summary_type
        self._setup_weights()
    
    def _setup_weights(self):
        """Configure scoring weights based on summary type"""
        if self.summary_type == "reflective":
            # For reflective summaries, prioritize semantic understanding over word overlap
            self.weights = {
                "semantic_similarity": 0.35,  # Most important - captures meaning
                "conceptual_coverage": 0.25,  # Covers key concepts, not just words
                "insight_quality": 0.20,      # Meta-analysis and understanding
                "coherence": 0.15,           # Logical flow and clarity
                "length_appropriateness": 0.05  # Less critical for reflective summaries
            }
        else:  # extractive
            # Traditional weights for fact-heavy summaries
            self.weights = {
                "semantic_similarity": 0.25,
                "conceptual_coverage": 0.30,
                "insight_quality": 0.10,
                "coherence": 0.20,
                "length_appropriateness": 0.15
            }
    
    def clean_chatgpt_conversation(self, text: str) -> str:
        """Remove ChatGPT conversation UI elements"""
        patterns_to_remove = [
            r'Skip to content\s*',
            r'This is a copy of a conversation between ChatGPT & Anonymous\.\s*',
            r'Report conversation\s*',
            r'You said:\s*',
            r'ChatGPT said:\s*',
            r'Attach\s*', r'Search\s*', r'Study\s*', r'Voice\s*',
            r'No file chosen\s*',
            r'ChatGPT can make mistakes\. Check important info\.\s*',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'^\s+|\s+$', '', text)
        return text
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts rather than just frequent words"""
        # Clean text
        text = self.clean_chatgpt_conversation(text)
        
        # Enhanced concept extraction
        concepts = []
        
        # Multi-word concepts (more meaningful than single words)
        concept_patterns = [
            r'\b(?:LGBT|LGBTQ)\s+(?:rights?|people|community|issues?|education)\b',
            r'\b(?:religious|faith|traditional)\s+(?:beliefs?|values?|institutions?|schools?)\b',
            r'\b(?:legal|constitutional|democratic)\s+(?:protections?|frameworks?|rights?)\b',
            r'\b(?:harm|discrimination|violence|harassment)\s+(?:prevention|reduction)\b',
            r'\b(?:cultural|social)\s+(?:change|norms?|sensitivity|traditions?)\b',
            r'\b(?:free|freedom of)\s+(?:speech|religion|association)\b',
            r'\b(?:medical|healthcare|clinical)\s+(?:intervention|treatment|decision|care)\b',
            r'\b(?:parental|parents?)\s+(?:rights?|concerns?|input)\b',
            r'\b(?:public|educational)\s+(?:policy|policies|institutions?|curricula?)\b',
            r'\b(?:evidence|research|data)\s+(?:based|driven|informed)\b'
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend([match.lower() for match in matches])
        
        # Key thematic words
        thematic_words = [
            'balance', 'balancing', 'tension', 'compromise', 'trade-off', 'trade-offs',
            'respect', 'respectful', 'dialogue', 'understanding', 'empathy',
            'protect', 'protection', 'safeguard', 'safety',
            'inclusion', 'discrimination', 'equality', 'fairness',
            'conscience', 'conviction', 'belief', 'doctrine',
            'harm', 'wellbeing', 'mental health', 'violence',
            'democracy', 'majority', 'minority', 'pluralism',
            'conversation', 'debate', 'discussion', 'exchange'
        ]
        
        text_lower = text.lower()
        for word in thematic_words:
            if word in text_lower:
                concepts.append(word)
        
        return list(set(concepts))  # Remove duplicates
    
    def calculate_semantic_similarity(self, original: str, summary: str) -> float:
        """Enhanced semantic similarity using TF-IDF with concept focus"""
        if not sklearn_available:
            # Fallback to concept overlap
            orig_concepts = set(self.extract_key_concepts(original))
            summ_concepts = set(self.extract_key_concepts(summary))
            if not orig_concepts:
                return 0.0
            return len(orig_concepts & summ_concepts) / len(orig_concepts)
        
        try:
            # Clean texts
            orig_clean = self.clean_chatgpt_conversation(original)
            summ_clean = self.clean_chatgpt_conversation(summary)
            
            # Use TF-IDF with concept-friendly settings
            vectorizer = TfidfVectorizer(
                max_features=300,
                ngram_range=(1, 3),  # Include phrases
                min_df=1,
                stop_words='english'
            )
            
            corpus = [orig_clean, summ_clean]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Boost if key concepts are preserved
            orig_concepts = set(self.extract_key_concepts(original))
            summ_concepts = set(self.extract_key_concepts(summary))
            
            if orig_concepts:
                concept_overlap = len(orig_concepts & summ_concepts) / len(orig_concepts)
                # Combine TF-IDF with concept preservation
                return 0.6 * similarity + 0.4 * concept_overlap
            
            return similarity
            
        except Exception:
            return 0.0
    
    def calculate_conceptual_coverage(self, original: str, summary: str) -> Dict[str, Any]:
        """Evaluate how well key concepts are covered"""
        orig_concepts = self.extract_key_concepts(original)
        summ_concepts = self.extract_key_concepts(summary)
        
        if not orig_concepts:
            return {"coverage": 1.0, "covered": [], "missed": []}
        
        orig_set = set(orig_concepts)
        summ_set = set(summ_concepts)
        
        covered = list(orig_set & summ_set)
        missed = list(orig_set - summ_set)
        
        coverage = len(covered) / len(orig_set)
        
        return {
            "coverage": coverage,
            "covered": covered,
            "missed": missed,
            "total_concepts": len(orig_set)
        }
    
    def calculate_insight_quality(self, original: str, summary: str) -> float:
        """Evaluate the meta-analytical quality of the summary"""
        summary_lower = summary.lower()
        
        # Check for meta-commentary indicators (good for reflective summaries)
        meta_indicators = [
            r'\boverall\b', r'\bin summary\b', r'\bto sum up\b',
            r'\bthe main\b', r'\bthe key\b', r'\bthe core\b',
            r'\bthe central\b', r'\bthe underlying\b',
            r'\bwhat this shows\b', r'\bwhat emerged\b',
            r'\bthe conversation (was|revealed|showed)\b',
            r'\bthe discussion (was|focused|centered)\b'
        ]
        
        meta_score = sum(1 for pattern in meta_indicators 
                        if re.search(pattern, summary_lower)) / len(meta_indicators)
        
        # Check for analysis of perspectives/balance
        perspective_indicators = [
            r'\bboth sides?\b', r'\bboth (views?|perspectives?|positions?)\b',
            r'\bbalanced?\b', r'\bbalancing\b', r'\bnuanced?\b',
            r'\bcomplex(ity)?\b', r'\btension\b', r'\btrade-?offs?\b',
            r'\brespectful\b', r'\bthoughtful\b'
        ]
        
        perspective_score = sum(1 for pattern in perspective_indicators 
                               if re.search(pattern, summary_lower)) / len(perspective_indicators)
        
        # Check for identification of key themes/insights
        theme_indicators = [
            r'\bthe theme\b', r'\bthe issue\b', r'\bthe question\b',
            r'\bthe challenge\b', r'\bthe problem\b', r'\bthe debate\b',
            r'\bhow to\b', r'\bwhere to draw\b', r'\bhow do (you|we)\b'
        ]
        
        theme_score = sum(1 for pattern in theme_indicators 
                         if re.search(pattern, summary_lower)) / len(theme_indicators)
        
        # Combine scores
        insight_score = (meta_score * 0.4 + perspective_score * 0.4 + theme_score * 0.2)
        
        # Bonus for reflective language
        if self.summary_type == "reflective":
            reflective_bonus = len(re.findall(r'\b(felt|seemed|appeared|turned out|ended up)\b', 
                                             summary_lower)) * 0.05
            insight_score = min(1.0, insight_score + reflective_bonus)
        
        return insight_score
    
    def calculate_coherence(self, summary: str) -> float:
        """Evaluate logical flow and readability"""
        sentences = re.split(r'[.!?]+', summary)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 2:
            return 0.5
        
        coherence = 0.8  # Base score
        
        # Check for transition words/phrases
        transitions = [
            r'\bhowever\b', r'\bbut\b', r'\balthough\b', r'\bwhile\b',
            r'\binstead\b', r'\brather\b', r'\bactually\b', r'\bin fact\b',
            r'\bwhen (i|we)\b', r'\bif (i|we)\b', r'\bwhat\b'
        ]
        
        summary_lower = summary.lower()
        transition_count = sum(1 for pattern in transitions 
                              if re.search(pattern, summary_lower))
        
        if transition_count > 0:
            coherence += min(0.2, transition_count * 0.05)
        
        # Check for logical progression markers
        progression = [
            r'\bfirst\b', r'\binitially\b', r'\bstarting\b',
            r'\bthen\b', r'\bnext\b', r'\blater\b',
            r'\bfinally\b', r'\bin (the )?end\b', r'\boverall\b'
        ]
        
        progression_count = sum(1 for pattern in progression 
                               if re.search(pattern, summary_lower))
        
        if progression_count > 1:
            coherence += 0.1
        
        return min(1.0, coherence)
    
    def calculate_length_appropriateness(self, original: str, summary: str) -> float:
        """Evaluate if summary length is appropriate"""
        orig_words = len(self.clean_chatgpt_conversation(original).split())
        summ_words = len(summary.split())
        
        if orig_words == 0:
            return 0.0
        
        compression_ratio = summ_words / orig_words
        
        if self.summary_type == "reflective":
            # Reflective summaries can be more flexible with length
            ideal_range = (0.05, 0.35)
        else:
            # Extractive summaries should be more compressed
            ideal_range = (0.05, 0.25)
        
        if ideal_range[0] <= compression_ratio <= ideal_range[1]:
            return 1.0
        elif compression_ratio < ideal_range[0]:
            return compression_ratio / ideal_range[0]
        elif compression_ratio <= ideal_range[1] + 0.1:
            return 1.0 - (compression_ratio - ideal_range[1]) * 2
        else:
            return max(0.2, 0.8 - (compression_ratio - ideal_range[1] - 0.1))
    
    def grade_summary(self, original_file: str, summary_file: str) -> Dict[str, Any]:
        """Main grading function with improved scoring"""
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = f.read()
        except Exception as e:
            return {"error": f"File reading error: {e}"}
        
        # Calculate individual scores
        semantic_sim = self.calculate_semantic_similarity(original, summary)
        conceptual_cov = self.calculate_conceptual_coverage(original, summary)
        insight_quality = self.calculate_insight_quality(original, summary)
        coherence = self.calculate_coherence(summary)
        length_score = self.calculate_length_appropriateness(original, summary)
        
        # Calculate weighted overall score
        overall = (
            self.weights["semantic_similarity"] * semantic_sim +
            self.weights["conceptual_coverage"] * conceptual_cov["coverage"] +
            self.weights["insight_quality"] * insight_quality +
            self.weights["coherence"] * coherence +
            self.weights["length_appropriateness"] * length_score
        )
        
        # Convert to percentage and letter grade
        percentage = round(overall * 100, 1)
        letter_grade = self._get_letter_grade(percentage)
        
        return {
            "overall_percentage": percentage,
            "overall_score": round(overall, 3),
            "letter_grade": letter_grade,
            "summary_type": self.summary_type,
            "breakdown": {
                "semantic_similarity": {
                    "score": round(semantic_sim, 3),
                    "weight": self.weights["semantic_similarity"],
                    "contribution": round(semantic_sim * self.weights["semantic_similarity"] * 100, 1)
                },
                "conceptual_coverage": {
                    "score": round(conceptual_cov["coverage"], 3),
                    "weight": self.weights["conceptual_coverage"],
                    "contribution": round(conceptual_cov["coverage"] * self.weights["conceptual_coverage"] * 100, 1),
                    "details": conceptual_cov
                },
                "insight_quality": {
                    "score": round(insight_quality, 3),
                    "weight": self.weights["insight_quality"],
                    "contribution": round(insight_quality * self.weights["insight_quality"] * 100, 1)
                },
                "coherence": {
                    "score": round(coherence, 3),
                    "weight": self.weights["coherence"],
                    "contribution": round(coherence * self.weights["coherence"] * 100, 1)
                },
                "length_appropriateness": {
                    "score": round(length_score, 3),
                    "weight": self.weights["length_appropriateness"],
                    "contribution": round(length_score * self.weights["length_appropriateness"] * 100, 1)
                }
            },
            "summary_stats": {
                "original_words": len(self.clean_chatgpt_conversation(original).split()),
                "summary_words": len(summary.split()),
                "compression_ratio": round(len(summary.split()) / len(self.clean_chatgpt_conversation(original).split()), 3) if original.split() else 0,
                "concepts_found": len(self.extract_key_concepts(original)),
                "concepts_covered": len(conceptual_cov["covered"])
            }
        }
    
    def _get_letter_grade(self, percentage: float) -> str:
        """Convert percentage to letter grade"""
        if percentage >= 90: return 'A'
        elif percentage >= 80: return 'B'
        elif percentage >= 70: return 'C'
        elif percentage >= 60: return 'D'
        else: return 'F'

def main():
    parser = argparse.ArgumentParser(description="Improved Summary Quality Grader")
    parser.add_argument("original_file", help="Original text file")
    parser.add_argument("summary_file", help="Summary text file")
    parser.add_argument("--type", choices=["reflective", "extractive"], default="reflective",
                       help="Summary type (default: reflective)")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    
    args = parser.parse_args()
    
    grader = ImprovedSummaryGrader(summary_type=args.type)
    result = grader.grade_summary(args.original_file, args.summary_file)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("="*70)
        print("IMPROVED SUMMARY QUALITY ASSESSMENT")
        print("="*70)
        print(f"Overall Score: {result['overall_percentage']}% (Grade: {result['letter_grade']})")
        print(f"Summary Type: {result['summary_type'].title()}")
        print()
        
        print("DETAILED BREAKDOWN:")
        print("-" * 50)
        for metric, data in result['breakdown'].items():
            name = metric.replace('_', ' ').title()
            print(f"{name:20} | {data['score']:.3f} | Weight: {data['weight']:.0%} | Contribution: {data['contribution']:.1f}%")
        
        print()
        print("CONCEPTUAL ANALYSIS:")
        print("-" * 50)
        cov = result['breakdown']['conceptual_coverage']['details']
        print(f"Concepts identified: {result['summary_stats']['concepts_found']}")
        print(f"Concepts covered: {result['summary_stats']['concepts_covered']}")
        print(f"Coverage rate: {cov['coverage']:.1%}")
        
        if cov['covered']:
            print(f"Covered concepts: {', '.join(cov['covered'][:8])}")
        if cov['missed']:
            print(f"Missed concepts (sample): {', '.join(cov['missed'][:8])}")

if __name__ == "__main__":
    main()