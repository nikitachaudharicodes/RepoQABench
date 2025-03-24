#!/usr/bin/env python3

import os
import subprocess
import time
import argparse

def read_urls_from_file(file_path):
    """Read URLs from a file, one URL per line."""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return []
        
    with open(file_path, 'r') as file:
        # Strip whitespace and filter out empty lines
        urls = [line.strip() for line in file if line.strip()]
    
    return urls

def process_all_urls():
    """Process all URLs from the issue_urls.txt and pr_urls.txt files."""
    issue_urls = read_urls_from_file('issue_urls.txt')
    pr_urls = read_urls_from_file('pr_urls.txt')
    
    if not issue_urls:
        print("No issue URLs found in issue_urls.txt")
        return
    
    # If PR URLs file exists but has fewer entries, pad with None
    if pr_urls and len(pr_urls) < len(issue_urls):
        pr_urls.extend([None] * (len(issue_urls) - len(pr_urls)))
    # If PR URLs file doesn't exist or is empty, use None for all issues
    elif not pr_urls:
        pr_urls = [None] * len(issue_urls)
    
    print(f"Found {len(issue_urls)} issues to process")
    
    # Get GitHub token from environment variable
    github_token = os.environ.get('GITHUB_TOKEN')
    token_arg = f"--token {github_token}" if github_token else ""
    
    # Process each issue with corresponding PR (if available)
    successful = 0
    failed = 0
    
    for i, issue_url in enumerate(issue_urls):
        pr_url = pr_urls[i] if i < len(pr_urls) else None
        pr_arg = f"--pr {pr_url}" if pr_url else ""
        
        print(f"\nProcessing {i+1}/{len(issue_urls)}: {issue_url}")
        if pr_url:
            print(f"With PR: {pr_url}")
        
        try:
            # Construct the command to run the original script
            cmd = f"python github_issue_scraper.py --url {issue_url} {pr_arg} {token_arg}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Success: {issue_url}")
                successful += 1
            else:
                print(f"Failed: {issue_url}")
                print(f"Error output: {result.stderr}")
                failed += 1
                
        except Exception as e:
            print(f"Failed to process {issue_url}: {e}")
            failed += 1
        
        # Respect rate limits
        time.sleep(2)
    
    print(f"\nProcessing complete. Successfully processed {successful} issues. Failed: {failed}")

def main():
    parser = argparse.ArgumentParser(description='Process multiple GitHub issues from text files')
    parser.add_argument('--token', help='GitHub API token (override environment variable)')
    
    args = parser.parse_args()
    
    # Set GitHub token as environment variable if provided
    if args.token:
        os.environ['GITHUB_TOKEN'] = args.token
    
    process_all_urls()

if __name__ == "__main__":
    main()