#!/usr/bin/env python3

import os
import subprocess
import argparse
import time
import json

def calculate_num_pairs(comment_count, base=5, min_comments=10, scale=0.2):
    """
    Calculate the number of QA pairs to generate based on comment count.
    
    Args:
        comment_count (int): Number of comments
        base (int): Minimum number of QA pairs
        min_comments (int): Minimum comment threshold
        scale (float): Scaling factor
        
    Returns:
        int: Number of QA pairs to generate
    """
    if comment_count <= min_comments:
        return base
    
    return base + int((comment_count - min_comments) * scale)

def process_all_json_files(directory="repoqabench", api_key=None, num_pairs=5):
    """
    Process all JSON files in the specified directory using qa_extractor.py
    
    Args:
        directory (str): Directory containing JSON files to process
        api_key (str): OpenAI API key for generating QA pairs
        num_pairs (int): Default number of QA pairs to generate per file
    """
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # Get list of JSON files
    json_files = []
    for root, _, files in os.walk(directory):
        json_files.extend([os.path.join(root, f) for f in files if f.endswith('.json')])
    
    if not json_files:
        print(f"No JSON files found in '{directory}'.")
        return
    
    print(f"Found {len(json_files)} JSON files to process.")
    
    # Process each file
    for i, file_path in enumerate(json_files):
        print(f"\nProcessing file {i+1}/{len(json_files)}: {file_path}")
        
        try:
            # Check if file exists and is valid JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    
                    # Determine number of QA pairs based on comment count
                    pairs_to_generate = num_pairs
                    if 'issue_comments' in data:
                        comment_count = len(data['issue_comments'])
                        pairs_to_generate = calculate_num_pairs(comment_count)
                        print(f"Found {comment_count} comments, generating {pairs_to_generate} QA pairs")
                    
                except json.JSONDecodeError:
                    print(f"Error: {file_path} is not a valid JSON file. Skipping.")
                    continue
            
            cmd = ["python", "qa_extractor.py", "--input", file_path]
            
            # Add API key if provided
            if api_key:
                cmd.extend(["--api-key", api_key])
            
            # Add calculated num_pairs
            cmd.extend(["--num-pairs", str(pairs_to_generate)])
            
            # Run the extractor
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Print output
            print(result.stdout)
            
            # Check for errors
            if result.returncode != 0:
                print(f"Error processing {file_path}:")
                print(result.stderr)
            
        except Exception as e:
            print(f"Exception processing {file_path}: {e}")
        
        # Add a small delay between processing files to avoid rate limiting
        time.sleep(1)
    
    print("\nProcessing complete!")

def main():
    parser = argparse.ArgumentParser(description='Run qa_extractor.py on all JSON files in repoqabench directory')
    parser.add_argument('--directory', default='repoqabench', help='Directory containing JSON files')
    parser.add_argument('--api-key', help='OpenAI API key')
    parser.add_argument('--num-pairs', type=int, default=5, help='Default number of QA pairs to generate per file')
    
    args = parser.parse_args()
    
    # Use API key from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Warning: No OpenAI API key provided. Generation of QA pairs will be skipped.")
        print("Set OPENAI_API_KEY environment variable or use --api-key.")
    
    process_all_json_files(args.directory, api_key, args.num_pairs)

if __name__ == "__main__":
    main()