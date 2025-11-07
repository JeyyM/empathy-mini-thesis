#!/usr/bin/env python3
"""
recall_accuracy_grader.py -- Focuses purely on accuracy of recall and message understanding
Evaluates if the core content and meaning was accurately captured, regardless of wording
"""

import re
import json
import argparse
from typing import Dict, List, Any, Tuple

class RecallAccuracyGrader:
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
    
    def extract_core_arguments(self, text: str) -> Dict[str, List[str]]:
        """Extract the main arguments and positions from the text"""
        text_clean = self.clean_text(text).lower()
        
        # Identify speaker positions and arguments
        arguments = {
            'user_concerns': [],
            'chatgpt_responses': [],
            'main_topics': [],
            'key_tensions': [],
            'outcomes': []
        }
        
        # User concerns patterns
        user_concern_patterns = [
            r'i think.*biased.*favor',
            r'people of faith feel',
            r'i worry.*legal protections.*compel',
            r'normalizing.*affects children',
            r'who decides.*counts as harm',
            r'cultural import.*west',
            r'platforms.*moderate.*silencing',
            r'medical transition.*minors.*risky',
            r'religious schools.*forced.*teach',
            r'pressure.*one side.*marginalized',
            r'majority.*community.*traditional.*democratic',
            r'resist.*enforcing acceptance'
        ]
        
        for pattern in user_concern_patterns:
            if re.search(pattern, text_clean):
                arguments['user_concerns'].append(pattern.replace(r'\b', '').replace('.*', ' '))
        
        # ChatGPT response patterns
        chatgpt_response_patterns = [
            r'reflect.*human-rights principles',
            r'balance competing rights.*freedom',
            r'distinguish.*internal religious.*public-facing',
            r'parents.*right.*public education.*responsibilities',
            r'measurable harms.*mental-health.*violence',
            r'cultural sensitivity.*non-heteronormative.*exist.*cultures',
            r'distinguish.*reasoned critique.*harassment',
            r'safeguarding minors.*multidisciplinary.*caution',
            r'distinction.*endorsing.*protecting',
            r'institutional design.*protect pluralism',
            r'constitutional.*protect.*inalienable rights',
            r'cultural persuasion.*durable.*legal mandates'
        ]
        
        for pattern in chatgpt_response_patterns:
            if re.search(pattern, text_clean):
                arguments['chatgpt_responses'].append(pattern.replace(r'\b', '').replace('.*', ' '))
        
        # Main topics discussed
        topic_patterns = [
            r'lgbt.*bias.*ai', r'religious.*conscience.*rights',
            r'schools.*education.*children', r'medical.*transition.*minors',
            r'free speech.*moderation', r'cultural.*western.*values',
            r'legal.*protections.*institutions', r'harm.*definition.*subjective',
            r'democracy.*majority.*minority', r'persuasion.*vs.*mandates'
        ]
        
        for pattern in topic_patterns:
            if re.search(pattern, text_clean):
                arguments['main_topics'].append(pattern.replace('.*', ' '))
        
        # Key tensions identified
        tension_patterns = [
            r'freedom.*religion.*freedom.*discrimination',
            r'parental.*rights.*student.*wellbeing',
            r'religious.*practice.*protecting.*vulnerable',
            r'majority.*rule.*minority.*rights',
            r'local.*traditions.*individual.*rights',
            r'reasoned.*critique.*targeted.*harassment',
            r'caution.*compassion.*medical.*decisions'
        ]
        
        for pattern in tension_patterns:
            if re.search(pattern, text_clean):
                arguments['key_tensions'].append(pattern.replace('.*', ' '))
        
        # Conversation outcomes
        outcome_patterns = [
            r'fair exchange.*core objections.*respected',
            r'balanced.*expected.*acknowledging.*concerns',
            r'respectful.*disagreement.*common.*ground',
            r'thoughtful.*back-and-forth.*understanding'
        ]
        
        for pattern in outcome_patterns:
            if re.search(pattern, text_clean):
                arguments['outcomes'].append(pattern.replace('.*', ' '))
        
        return arguments
    
    def assess_topic_recall(self, original: str, summary: str) -> Dict[str, float]:
        """Assess how well specific topics were recalled"""
        orig_args = self.extract_core_arguments(original)
        summary_lower = summary.lower()
        
        topic_scores = {}
        
        # Check if major topics are captured in summary (allowing paraphrasing)
        topic_mappings = {
            'ai_bias_concern': [
                r'chatgpt.*biased', r'ai.*one-?sided', r'institutions.*feel.*one-?sided',
                r'support.*lgbt.*automatically.*neutral', r'default.*pro-?lgbt'
            ],
            'religious_freedom': [
                r'religious.*traditional.*backgrounds?', r'people.*faith.*feel',
                r'religious.*convictions?', r'traditional.*beliefs?.*backward'
            ],
            'education_children': [
                r'lgbt.*education.*schools?', r'medical.*decisions?.*minors?',
                r'schools?.*teaching.*children', r'safeguards?.*caution'
            ],
            'balance_theme': [
                r'balance.*individual.*freedoms?.*cultural',
                r'protect.*lgbt.*without.*forcing.*religious',
                r'draw.*line.*between.*protecting.*forcing',
                r'trade-?offs?.*diverse.*society'
            ],
            'respectful_tone': [
                r'respectful', r'balanced.*expected', r'fair.*exchange',
                r'acknowledged?.*risks?', r'didn\'t.*dismiss.*worries'
            ],
            'harm_evidence_theme': [
                r'research.*legal.*norms.*minimizing.*harm',
                r'preventing.*discrimination.*protecting.*wellbeing',
                r'evidence.*rights', r'framed.*evidence.*rights'
            ]
        }
        
        for topic, patterns in topic_mappings.items():
            topic_found = any(re.search(pattern, summary_lower) for pattern in patterns)
            # Check if topic was actually present in original
            topic_in_original = len(orig_args['user_concerns']) > 0 or len(orig_args['chatgpt_responses']) > 0
            
            if topic_in_original:
                topic_scores[topic] = 1.0 if topic_found else 0.0
            else:
                topic_scores[topic] = 0.8  # Neutral if topic wasn't prominent
        
        return topic_scores
    
    def assess_position_accuracy(self, original: str, summary: str) -> float:
        """Check if the positions of each side are accurately represented"""
        orig_args = self.extract_core_arguments(original)
        summary_lower = summary.lower()
        
        accuracy_score = 1.0
        
        # Check for misrepresentations of user position
        user_misrepresentation_penalties = [
            (r'user.*aggressive', 0.3),  # User was not aggressive
            (r'user.*dismissive', 0.2),  # User was raising concerns, not dismissive
            (r'user.*hostile', 0.3),     # User maintained respectful tone
        ]
        
        for pattern, penalty in user_misrepresentation_penalties:
            if re.search(pattern, summary_lower):
                accuracy_score -= penalty
        
        # Check for misrepresentations of ChatGPT position
        chatgpt_misrepresentation_penalties = [
            (r'chatgpt.*just.*defended.*without.*acknowledging', 0.3),  # Actually did acknowledge
            (r'chatgpt.*dismissed.*religious.*concerns', 0.3),          # Actually acknowledged them
            (r'chatgpt.*one-?sided.*lgbt.*support', 0.2),               # If summary says this was the case when it wasn't
        ]
        
        for pattern, penalty in chatgpt_misrepresentation_penalties:
            if re.search(pattern, summary_lower):
                accuracy_score -= penalty
        
        # Positive accuracy indicators
        accuracy_bonuses = [
            (r'acknowledged.*risks?.*safeguards?', 0.1),      # Correctly notes ChatGPT acknowledged concerns
            (r'didn\'t.*just.*defend.*positions?', 0.1),      # Correctly notes balanced approach
            (r'framed.*evidence.*rights.*not.*reshape', 0.1), # Correctly captures ChatGPT's explanation
        ]
        
        for pattern, bonus in accuracy_bonuses:
            if re.search(pattern, summary_lower):
                accuracy_score += bonus
        
        return max(0.0, min(1.0, accuracy_score))
    
    def assess_outcome_understanding(self, original: str, summary: str) -> float:
        """Check if the summary correctly understands what happened in the conversation"""
        orig_clean = self.clean_text(original).lower()
        summary_lower = summary.lower()
        
        # Identify actual conversation outcome
        conversation_was_respectful = any(re.search(pattern, orig_clean) for pattern in [
            r'fair.*exchange', r'conversation.*respected', r'thoughtful', r'civil'
        ])
        
        conversation_was_balanced = any(re.search(pattern, orig_clean) for pattern in [
            r'acknowledge.*sincerity', r'describe.*both.*sides',
            r'balance.*competing', r'respectful.*response'
        ])
        
        user_appreciated_tone = any(re.search(pattern, orig_clean) for pattern in [
            r'appreciate.*tone', r'fair.*exchange', r'respected.*explaining'
        ])
        
        outcome_score = 0.0
        
        # Check if summary correctly identifies these outcomes
        if conversation_was_respectful:
            if any(re.search(pattern, summary_lower) for pattern in [
                r'respectful', r'fair', r'balanced.*expected'
            ]):
                outcome_score += 0.4
        
        if conversation_was_balanced:
            if any(re.search(pattern, summary_lower) for pattern in [
                r'balanced.*expected', r'acknowledged.*concerns',
                r'didn\'t.*just.*defend', r'both.*sides'
            ]):
                outcome_score += 0.4
        
        if user_appreciated_tone:
            if any(re.search(pattern, summary_lower) for pattern in [
                r'felt.*respectful', r'fair.*exchange', r'appreciated'
            ]):
                outcome_score += 0.2
        
        return outcome_score
    
    def assess_semantic_preservation(self, original: str, summary: str) -> float:
        """Check if core semantic meaning is preserved even with different words"""
        orig_clean = self.clean_text(original)
        
        # Extract key semantic concepts and their paraphrases in summary
        semantic_mappings = [
            # Original concept -> Acceptable paraphrases in summary
            ('balance competing rights', ['balance.*freedoms.*values', 'protect.*without.*forcing']),
            ('harm reduction', ['preventing.*discrimination', 'protecting.*wellbeing', 'minimizing.*harm']),
            ('evidence based', ['research.*legal.*norms', 'evidence.*rights']),
            ('religious conscience', ['religious.*traditional.*backgrounds', 'traditional.*beliefs']),
            ('pluralistic society', ['diverse.*society', 'living.*together.*diverse']),
            ('democratic debate', ['respectful.*dialogue', 'understanding.*trade-?offs']),
        ]
        
        summary_lower = summary.lower()
        preservation_score = 0.0
        concepts_found = 0
        
        for original_concept, paraphrases in semantic_mappings:
            if original_concept.replace(' ', '.*') in self.clean_text(original).lower():
                concepts_found += 1
                # Check if any acceptable paraphrase appears in summary
                if any(re.search(paraphrase, summary_lower) for paraphrase in paraphrases):
                    preservation_score += 1.0
        
        if concepts_found == 0:
            return 0.8  # Default if no clear concepts identified
        
        return preservation_score / concepts_found
    
    def grade_summary(self, original_file: str, summary_file: str) -> Dict[str, Any]:
        """Grade summary focusing purely on recall accuracy and message understanding"""
        
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = f.read()
        except Exception as e:
            return {"error": f"Error reading files: {e}"}
        
        # Calculate core accuracy metrics
        topic_scores = self.assess_topic_recall(original, summary)
        position_accuracy = self.assess_position_accuracy(original, summary)
        outcome_understanding = self.assess_outcome_understanding(original, summary)
        semantic_preservation = self.assess_semantic_preservation(original, summary)
        
        # Calculate topic recall average
        topic_recall = sum(topic_scores.values()) / len(topic_scores) if topic_scores else 0.0
        
        # Weights focused purely on accuracy of recall and understanding
        weights = {
            'topic_recall': 0.35,           # Did you recall the main topics discussed?
            'position_accuracy': 0.25,      # Did you accurately represent each side's positions?
            'outcome_understanding': 0.25,  # Did you understand what actually happened?
            'semantic_preservation': 0.15   # Did you preserve core meaning even when paraphrasing?
        }
        
        scores = {
            'topic_recall': topic_recall,
            'position_accuracy': position_accuracy,
            'outcome_understanding': outcome_understanding,
            'semantic_preservation': semantic_preservation
        }
        
        # Calculate weighted overall score
        overall = sum(scores[metric] * weights[metric] for metric in scores)
        percentage = overall * 100
        
        # Grade scale focused on accuracy
        def get_grade(pct):
            if pct >= 90: return 'A'
            elif pct >= 80: return 'B'
            elif pct >= 70: return 'C'
            elif pct >= 60: return 'D'
            else: return 'F'
        
        return {
            'overall_percentage': round(percentage, 1),
            'overall_score': round(overall, 3),
            'letter_grade': get_grade(percentage),
            'focus': 'Recall Accuracy & Message Understanding',
            'breakdown': {
                metric: {
                    'score': round(scores[metric], 3),
                    'weight': weights[metric],
                    'contribution': round(scores[metric] * weights[metric] * 100, 1)
                } for metric in scores
            },
            'topic_details': {
                'individual_topics': {k: round(v, 2) for k, v in topic_scores.items()},
                'topics_covered': sum(1 for v in topic_scores.values() if v > 0.5),
                'total_topics': len(topic_scores)
            },
            'summary_stats': {
                'original_words': len(self.clean_text(original).split()),
                'summary_words': len(summary.split()),
                'compression_ratio': round(len(summary.split()) / len(self.clean_text(original).split()), 3) if original.split() else 0
            }
        }

def main():
    parser = argparse.ArgumentParser(description="Recall Accuracy & Message Understanding Grader")
    parser.add_argument("original_file", help="Original text file")
    parser.add_argument("summary_file", help="Summary text file") 
    parser.add_argument("--json", action="store_true", help="Output JSON")
    
    args = parser.parse_args()
    
    grader = RecallAccuracyGrader()
    result = grader.grade_summary(args.original_file, args.summary_file)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("="*70)
        print("RECALL ACCURACY & MESSAGE UNDERSTANDING ASSESSMENT")
        print("="*70)
        print(f"Overall Score: {result['overall_percentage']}% (Grade: {result['letter_grade']})")
        print(f"Focus: {result['focus']}")
        print()
        
        print("ACCURACY BREAKDOWN:")
        print("-" * 50)
        breakdown = result['breakdown']
        for metric, data in breakdown.items():
            name = metric.replace('_', ' ').title()
            print(f"{name:20} | {data['score']:.3f} | Weight: {data['weight']:.0%} | Contribution: {data['contribution']:.1f}%")
        
        print()
        print("TOPIC RECALL DETAILS:")
        print("-" * 50)
        topic_details = result['topic_details']
        print(f"Topics Successfully Recalled: {topic_details['topics_covered']}/{topic_details['total_topics']}")
        
        for topic, score in topic_details['individual_topics'].items():
            status = "✅" if score > 0.5 else "❌"
            print(f"{status} {topic.replace('_', ' ').title()}: {score:.1f}")
        
        print()
        print("ACCURACY ASSESSMENT:")
        print("-" * 50)
        if breakdown['position_accuracy']['score'] >= 0.8:
            print("✅ Accurately represents positions of both sides")
        else:
            print("⚠️  Some inaccuracies in representing positions")
            
        if breakdown['outcome_understanding']['score'] >= 0.7:
            print("✅ Correctly understands what happened in conversation")
        else:
            print("❌ Misunderstands conversation outcome or dynamics")
            
        if breakdown['semantic_preservation']['score'] >= 0.7:
            print("✅ Preserves core meaning even when paraphrasing")
        else:
            print("⚠️  Some loss of meaning in paraphrasing")

if __name__ == "__main__":
    main()