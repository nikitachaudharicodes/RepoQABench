#!/usr/bin/env python3

import os
import subprocess
import argparse
import time

def process_all_json_files(directory="repoqabench", api_key=None, num_pairs=5):
    """
    Process all JSON files in the specified directory using qa_extractor.py
    
    Args:
        directory (str): Directory containing JSON files to process
        api_key (str): OpenAI API key for generating QA pairs
        num_pairs (int): Number of QA pairs to generate per file
    """
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # Get list of JSON files
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in '{directory}'.")
        return
    
    print(f"Found {len(json_files)} JSON files to process.")
    
    # Process each file
    for i, json_file in enumerate(json_files):
        file_path = os.path.join(directory, json_file)
        print(f"\nProcessing file {i+1}/{len(json_files)}: {json_file}")
        
        try:
            cmd = ["python", "qa_extractor.py", "--input", file_path]
            
            # Add API key if provided
            if api_key:
                cmd.extend(["--api-key", api_key])
            
            # Add num_pairs if not default
            if num_pairs != 5:
                cmd.extend(["--num-pairs", str(num_pairs)])
            
            # Run the extractor
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Print output
            print(result.stdout)
            
            # Check for errors
            if result.returncode != 0:
                print(f"Error processing {json_file}:")
                print(result.stderr)
            
        except Exception as e:
            print(f"Exception processing {json_file}: {e}")
        
        # Add a small delay between processing files to avoid rate limiting
        time.sleep(1)
    
    print("\nProcessing complete!")

def main():
    parser = argparse.ArgumentParser(description='Run qa_extractor.py on all JSON files in repoqabench directory')
    parser.add_argument('--directory', default='repoqabench', help='Directory containing JSON files')
    parser.add_argument('--api-key', help='OpenAI API key')
    parser.add_argument('--num-pairs', type=int, default=5, help='Number of QA pairs to generate per file')
    
    args = parser.parse_args()
    
    # Use API key from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Warning: No OpenAI API key provided. Generation of QA pairs will be skipped.")
        print("Set OPENAI_API_KEY environment variable or use --api-key.")
    
    process_all_json_files(args.directory, api_key, args.num_pairs)

if __name__ == "__main__":
    main()