#!/usr/bin/env python3

import os
import json
import argparse
from collections import defaultdict

def analyze_repoqa_files(directory="repoqabench"):
    """
    Analyze JSON files in the specified directory to count statistics
    
    Args:
        directory (str): Directory containing JSON files to analyze
    """
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # # Get list of JSON files
    # json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    # Recursively find all .json files
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    if not json_files:
        print(f"No JSON files found in '{directory}'.")
        return
    
    print(f"Found {len(json_files)} JSON files to analyze.")
    
    # Initialize counters
    total_files = 0
    total_issue_comments = 0
    total_code_context = 0
    total_questions = 0
    total_questions_generated = 0
    
    # Stats per file
    file_stats = defaultdict(dict)
    
    # Process each file
    for file_path in json_files:
        json_file = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Count issue_comments
            issue_comments_count = len(data.get('issue_comments', []))
            total_issue_comments += issue_comments_count
            file_stats[json_file]['issue_comments'] = issue_comments_count
            
            # Count code_context
            code_context_count = len(data.get('code_context', []))
            total_code_context += code_context_count
            file_stats[json_file]['code_context'] = code_context_count
            
            # Count questions
            questions_count = len(data.get('questions', []))
            total_questions += questions_count
            file_stats[json_file]['questions'] = questions_count
            
            # Count questions_generated
            questions_generated_count = len(data.get('questions_generated', []))
            total_questions_generated += questions_generated_count
            file_stats[json_file]['questions_generated'] = questions_generated_count
            
            total_files += 1
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Calculate averages
    avg_issue_comments = total_issue_comments / total_files if total_files > 0 else 0
    avg_code_context = total_code_context / total_files if total_files > 0 else 0
    
    # Print summary
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total files processed: {total_files}")
    print(f"Average issue_comments per file: {avg_issue_comments:.2f}")
    print(f"Average code_context per file: {avg_code_context:.2f}")
    print(f"Total questions across all files: {total_questions}")
    print(f"Total questions_generated across all files: {total_questions_generated}")
    
    # Print per-file stats
    print("\n=== PER-FILE STATISTICS ===")
    for file, stats in file_stats.items():
        print(f"\n{file}:")
        print(f"  issue_comments: {stats['issue_comments']}")
        print(f"  code_context: {stats['code_context']}")
        print(f"  questions: {stats['questions']}")
        print(f"  questions_generated: {stats['questions_generated']}")
    
    return {
        'total_files': total_files,
        'avg_issue_comments': avg_issue_comments,
        'avg_code_context': avg_code_context,
        'total_questions': total_questions,
        'total_questions_generated': total_questions_generated,
        'file_stats': file_stats
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze JSON files in repoqabench directory')
    parser.add_argument('--directory', default='repoqabench', help='Directory containing JSON files')
    
    args = parser.parse_args()
    analyze_repoqa_files(args.directory)

if __name__ == "__main__":
    main()