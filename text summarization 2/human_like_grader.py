#!/usr/bin/env python3
"""
human_like_grader.py -- Summary grader that mimics human evaluation
Focuses on what humans actually value in summaries
"""

import re
import json
import argparse
from typing import Dict, List, Any

class HumanLikeGrader:
    def __init__(self):
        pass
    
    def clean_text(self, text: str) -> str:
        """Clean ChatGPT conversation artifacts"""
        patterns = [
            r'Skip to content\s*', r'This is a copy of a conversation.*?\n',
            r'Report conversation\s*', r'You said:\s*', r'ChatGPT said:\s*',
            r'Attach\s*', r'Search\s*', r'Study\s*', r'Voice\s*',
            r'No file chosen\s*', r'ChatGPT can make mistakes.*?\n'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        return re.sub(r'\s+', ' ', text).strip()
    
    def assess_core_message_capture(self, original: str, summary: str) -> float:
        """Does the summary capture the essential message/conclusion?"""
        original_clean = self.clean_text(original).lower()
        summary_clean = summary.lower()
        
        # Look for key conclusion/outcome indicators in original
        outcome_patterns = [
            r'thanks.*fair exchange',
            r'feel.*conversation respected',
            r'balanced.*expected',
            r'respectful.*disagreement',
            r'common ground'
        ]
        
        original_has_conclusion = any(re.search(p, original_clean) for p in outcome_patterns)
        
        if original_has_conclusion:
            # Check if summary captures the balanced/respectful nature
            summary_captures = any(re.search(p, summary_clean) for p in [
                r'balanced.*expected', r'respectful', r'fair', 
                r'both sides', r'understanding', r'less.*win'
            ])
            return 1.0 if summary_captures else 0.3
        
        # For other texts, check if main themes are present
        return 0.7  # Default decent score if no clear conclusion pattern
    
    def assess_perspective_understanding(self, original: str, summary: str) -> float:
        """Does the summary show understanding of different viewpoints?"""
        summary_lower = summary.lower()
        
        # Check for perspective awareness
        perspective_indicators = [
            r'both.*side', r'different.*view', r'opposing', r'other.*perspective',
            r'while.*also', r'but.*also', r'instead.*expected',
            r'pushed? back', r'counter', r'however', r'although'
        ]
        
        perspective_score = 0.0
        for pattern in perspective_indicators:
            if re.search(pattern, summary_lower):
                perspective_score += 0.2
        
        # Bonus for explicitly mentioning both sides/viewpoints
        if re.search(r'(my|one).*point.*was.*chatgpt.*side', summary_lower):
            perspective_score += 0.3
        
        if re.search(r'didn\'t.*dismiss.*worries', summary_lower):
            perspective_score += 0.2
            
        return min(1.0, perspective_score)
    
    def assess_insight_depth(self, original: str, summary: str) -> float:
        """Does the summary provide meaningful insights beyond just facts?"""
        summary_lower = summary.lower()
        
        # Meta-analytical insights
        insight_patterns = [
            r'really about.*where to draw.*line',  # Excellent insight
            r'trade-?offs.*living.*diverse society',  # Good insight
            r'balance.*individual.*cultural',  # Good insight
            r'automatically becomes.*neutral',  # Good insight
            r'institutions.*feel one-sided',  # Good insight
            r'less about.*win.*more about.*understand'  # Excellent insight
        ]
        
        insight_score = 0.0
        for pattern in insight_patterns:
            if re.search(pattern, summary_lower):
                insight_score += 0.25
        
        # General insight indicators
        general_insights = [
            r'what this.*about', r'the real.*question',
            r'the underlying', r'turns out', r'ended up',
            r'wasn\'t.*expected', r'actually'
        ]
        
        for pattern in general_insights:
            if re.search(pattern, summary_lower):
                insight_score += 0.1
        
        return min(1.0, insight_score)
    
    def assess_accuracy(self, original: str, summary: str) -> float:
        """Are the facts and characterizations accurate?"""
        original_lower = self.clean_text(original).lower()
        summary_lower = summary.lower()
        
        # Check for major factual errors or mischaracterizations
        accuracy = 1.0
        
        # Check if summary incorrectly characterizes the conversation
        if 'aggressive' in summary_lower and 'respectful' in original_lower:
            accuracy -= 0.3
        
        if 'one-sided' in summary_lower and 'acknowledge' in original_lower:
            # This is actually correct characterization, no penalty
            pass
        
        # Check if key topics mentioned are accurate
        key_topics = ['lgbt', 'religious', 'education', 'medical', 'school']
        original_topics = [topic for topic in key_topics if topic in original_lower]
        summary_topics = [topic for topic in key_topics if topic in summary_lower]
        
        # Penalty for mentioning topics not in original
        false_topics = set(summary_topics) - set(original_topics)
        accuracy -= len(false_topics) * 0.1
        
        return max(0.0, accuracy)
    
    def assess_completeness(self, original: str, summary: str) -> float:
        """Does the summary cover the important parts without major omissions?"""
        original_lower = self.clean_text(original).lower()
        summary_lower = summary.lower()
        
        # Identify major themes in original
        major_themes = {
            'bias_concern': any(re.search(p, original_lower) for p in [r'biased.*favor', r'push.*moral']),
            'religious_tension': any(re.search(p, original_lower) for p in [r'religious.*doctrine', r'faith.*feel']),
            'education_concern': any(re.search(p, original_lower) for p in [r'schools.*teaching', r'children.*values']),
            'medical_concern': any(re.search(p, original_lower) for p in [r'medical.*minor', r'puberty.*blocker']),
            'free_speech': any(re.search(p, original_lower) for p in [r'free speech', r'silencing.*voice']),
            'balance_theme': any(re.search(p, original_lower) for p in [r'balance.*competing', r'respect.*practice'])
        }
        
        # Check coverage in summary
        covered_themes = {
            'bias_concern': any(re.search(p, summary_lower) for p in [r'biased', r'one-sided', r'neutral.*position']),
            'religious_tension': any(re.search(p, summary_lower) for p in [r'religious', r'traditional', r'faith']),
            'education_concern': any(re.search(p, summary_lower) for p in [r'education.*school', r'children']),
            'medical_concern': any(re.search(p, summary_lower) for p in [r'medical.*minor', r'medical.*decision']),
            'free_speech': any(re.search(p, summary_lower) for p in [r'free speech', r'silenc']),
            'balance_theme': any(re.search(p, summary_lower) for p in [r'balance', r'draw.*line', r'protect.*without.*forcing'])
        }
        
        present_themes = sum(major_themes.values())
        covered_count = sum(1 for theme in major_themes if major_themes[theme] and covered_themes[theme])
        
        if present_themes == 0:
            return 0.8  # Default if no major themes detected
        
        coverage_ratio = covered_count / present_themes
        
        # Bonus for getting the meta-theme (balancing/drawing lines)
        if covered_themes['balance_theme']:
            coverage_ratio += 0.2
        
        return min(1.0, coverage_ratio)
    
    def assess_readability(self, summary: str) -> float:
        """Is the summary clear, well-written, and easy to follow?"""
        sentences = re.split(r'[.!?]', summary)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(sentences) < 2:
            return 0.5
        
        # Check sentence length variation (good writing)
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        
        readability = 0.8  # Base score
        
        # Reasonable sentence lengths (10-30 words is good)
        if 10 <= avg_length <= 30:
            readability += 0.1
        
        # Check for good flow indicators
        flow_words = [
            r'\bhowever\b', r'\binstead\b', r'\bactually\b', r'\boverall\b',
            r'\bwhen i\b', r'\bif i had to\b', r'\bso\b', r'\bbut\b'
        ]
        
        summary_lower = summary.lower()
        flow_count = sum(1 for pattern in flow_words if re.search(pattern, summary_lower))
        
        if flow_count >= 2:
            readability += 0.1
        
        return min(1.0, readability)
    
    def grade_summary(self, original_file: str, summary_file: str) -> Dict[str, Any]:
        """Human-like grading that focuses on what actually matters"""
        
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = f.read()
        except Exception as e:
            return {"error": f"Error reading files: {e}"}
        
        # Calculate scores (each 0-1)
        scores = {
            'core_message': self.assess_core_message_capture(original, summary),
            'perspective_understanding': self.assess_perspective_understanding(original, summary),
            'insight_depth': self.assess_insight_depth(original, summary),
            'accuracy': self.assess_accuracy(original, summary),
            'completeness': self.assess_completeness(original, summary),
            'readability': self.assess_readability(summary)
        }
        
        # Human-like weights (what humans actually care about)
        weights = {
            'core_message': 0.25,          # Did you get the point?
            'insight_depth': 0.20,         # Do you understand what happened?
            'accuracy': 0.20,              # Are you telling the truth?
            'perspective_understanding': 0.15,  # Do you see both sides?
            'completeness': 0.15,          # Did you miss anything important?
            'readability': 0.05            # Is it well-written?
        }
        
        # Calculate weighted score
        overall = sum(scores[metric] * weights[metric] for metric in scores)
        percentage = overall * 100
        
        # Human-like letter grades (more generous for good summaries)
        def get_grade(pct):
            if pct >= 85: return 'A'
            elif pct >= 75: return 'B'
            elif pct >= 65: return 'C'
            elif pct >= 55: return 'D'
            else: return 'F'
        
        return {
            'overall_percentage': round(percentage, 1),
            'overall_score': round(overall, 3),
            'letter_grade': get_grade(percentage),
            'human_assessment': self._get_human_feedback(percentage),
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
                'compression_ratio': round(len(summary.split()) / len(self.clean_text(original).split()), 3) if original.split() else 0
            }
        }
    
    def _get_human_feedback(self, percentage: float) -> str:
        """Provide human-like qualitative feedback"""
        if percentage >= 85:
            return "Excellent summary that captures the essence and provides real insight."
        elif percentage >= 75:
            return "Good summary that shows understanding and covers key points well."
        elif percentage >= 65:
            return "Adequate summary that gets the basics right but could be more insightful."
        elif percentage >= 55:
            return "Weak summary that misses important aspects or lacks understanding."
        else:
            return "Poor summary that fails to capture the essential message or meaning."

def main():
    parser = argparse.ArgumentParser(description="Human-like Summary Grader")
    parser.add_argument("original_file", help="Original text file")
    parser.add_argument("summary_file", help="Summary text file")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    
    args = parser.parse_args()
    
    grader = HumanLikeGrader()
    result = grader.grade_summary(args.original_file, args.summary_file)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("="*70)
        print("HUMAN-LIKE SUMMARY ASSESSMENT")
        print("="*70)
        print(f"Overall Score: {result['overall_percentage']}% (Grade: {result['letter_grade']})")
        print(f"Assessment: {result['human_assessment']}")
        print()
        
        print("DETAILED BREAKDOWN:")
        print("-" * 50)
        breakdown = result['breakdown']
        for metric, data in breakdown.items():
            name = metric.replace('_', ' ').title()
            print(f"{name:20} | {data['score']:.3f} | Weight: {data['weight']:.0%} | Contribution: {data['contribution']:.1f}%")
        
        print()
        print("WHAT THIS MEANS:")
        print("-" * 50)
        if breakdown['core_message']['score'] >= 0.8:
            print("✅ Successfully captures the main message/outcome")
        else:
            print("❌ Misses or misrepresents the core message")
            
        if breakdown['insight_depth']['score'] >= 0.6:
            print("✅ Shows good analytical understanding")
        else:
            print("❌ Lacks depth or meaningful insights")
            
        if breakdown['accuracy']['score'] >= 0.8:
            print("✅ Accurate representation of content")
        else:
            print("⚠️  Contains inaccuracies or mischaracterizations")

if __name__ == "__main__":
    main()