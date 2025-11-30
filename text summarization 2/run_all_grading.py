#!/usr/bin/env python3
"""
Run grading on all chat/summary pairs and calculate averages per group
"""

import os
import json
from main import ContentRecallGrader

def run_grading():
    base_path = "files"
    groups = ["neutral", "opposing", "similar"]
    
    results = {group: [] for group in groups}
    
    grader = ContentRecallGrader()
    
    for group in groups:
        group_path = os.path.join(base_path, group)
        
        if not os.path.exists(group_path):
            print(f"⚠️  Folder not found: {group_path}")
            continue
        
        # Get all chat files
        chat_files = [f for f in os.listdir(group_path) if f.endswith("Chat.txt")]
        
        print(f"\n{'='*60}")
        print(f"GROUP: {group.upper()}")
        print(f"{'='*60}")
        
        for chat_file in sorted(chat_files):
            # Derive summary filename
            name = chat_file.replace("Chat.txt", "")
            summary_file = f"{name}Summary.txt"
            
            chat_path = os.path.join(group_path, chat_file)
            summary_path = os.path.join(group_path, summary_file)
            
            if not os.path.exists(summary_path):
                print(f"⚠️  Summary not found for: {name}")
                continue
            
            # Grade the summary
            result = grader.grade_summary(chat_path, summary_path)
            
            if "error" in result:
                print(f"❌ Error grading {name}: {result['error']}")
                continue
            
            results[group].append({
                "name": name,
                "percentage": result["overall_percentage"],
                "grade": result["letter_grade"],
                "breakdown": result["breakdown"]
            })
            
            print(f"\n{name}:")
            print(f"  Score: {result['overall_percentage']}% (Grade: {result['letter_grade']})")
            print(f"  - Semantic Similarity: {result['breakdown']['semantic_similarity']['score']:.3f}")
            print(f"  - Topic Coverage: {result['breakdown']['topic_coverage']['score']:.3f}")
            print(f"  - Factual Accuracy: {result['breakdown']['factual_accuracy']['score']:.3f}")
    
    # Calculate and display averages
    print(f"\n\n{'='*60}")
    print("SUMMARY BY GROUP")
    print(f"{'='*60}\n")
    
    for group in groups:
        if not results[group]:
            print(f"{group.upper()}: No results")
            continue
        
        avg_percentage = sum(r["percentage"] for r in results[group]) / len(results[group])
        avg_semantic = sum(r["breakdown"]["semantic_similarity"]["score"] for r in results[group]) / len(results[group])
        avg_topics = sum(r["breakdown"]["topic_coverage"]["score"] for r in results[group]) / len(results[group])
        avg_accuracy = sum(r["breakdown"]["factual_accuracy"]["score"] for r in results[group]) / len(results[group])
        
        grade_counts = {}
        for r in results[group]:
            grade_counts[r["grade"]] = grade_counts.get(r["grade"], 0) + 1
        
        print(f"{group.upper()}:")
        print(f"  Count: {len(results[group])}")
        print(f"  Average Score: {avg_percentage:.1f}%")
        print(f"  Average Semantic Similarity: {avg_semantic:.3f}")
        print(f"  Average Topic Coverage: {avg_topics:.3f}")
        print(f"  Average Factual Accuracy: {avg_accuracy:.3f}")
        print(f"  Grade Distribution: {dict(sorted(grade_counts.items()))}")
        print()
    
    # Overall summary
    all_results = []
    for group in groups:
        all_results.extend(results[group])
    
    if all_results:
        overall_avg = sum(r["percentage"] for r in all_results) / len(all_results)
        print(f"{'='*60}")
        print(f"OVERALL AVERAGE ACROSS ALL GROUPS: {overall_avg:.1f}%")
        print(f"Total Summaries Graded: {len(all_results)}")
        print(f"{'='*60}")

if __name__ == "__main__":
    run_grading()
