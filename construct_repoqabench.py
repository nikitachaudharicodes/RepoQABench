#!/usr/bin/env python3

import os
import json
import requests
import argparse
import time
import subprocess
from urllib.parse import urlparse
import re

class RepoQABenchConstructor:
    def __init__(self, github_token=None):
        """
        Initialize the RepoQABench constructor.
        
        Args:
            github_token (str, optional): GitHub API token for authenticated requests.
        """
        self.headers = {'Accept': 'application/vnd.github.v3+json'}
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"
        
        # Create output directories if they don't exist
        os.makedirs("repoqabench", exist_ok=True)
        os.makedirs("issue_pr_pairs", exist_ok=True)
    
    def parse_github_repo_url(self, repo_url):
        """Extract owner and repo name from GitHub repository URL."""
        parsed_url = urlparse(repo_url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        if len(path_parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")
            
        owner = path_parts[0]
        repo = path_parts[1]
        
        return owner, repo
    
    def get_filtered_issues(self, owner, repo, per_page=100):
        """
        Get issues that meet the filtering criteria:
        1. Closed issues
        2. At least 10 comments
        3. Tagged as 'good first issue'
        4. Has exactly 1 associated PR
        
        Returns:
            list: List of filtered issue-PR pairs
        """
        issue_pr_pairs = []
        page = 1
        total_issues_fetched = 0
        total_issues_filtered = 0
        
        print(f"Fetching issues for {owner}/{repo}...")
        
        while True:
            # Query for closed issues with 'good first issue' label
            issues_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
            params = {
                'state': 'closed', 
                'labels': 'good first issue',
                'per_page': per_page,
                'page': page
            }
            
            try:
                issues_response = requests.get(issues_url, headers=self.headers, params=params)
                issues_response.raise_for_status()
                issues = issues_response.json()
                
                # Stop if no more issues
                if not issues:
                    break
                
                total_issues_fetched += len(issues)
                print(f"Fetched {len(issues)} issues from page {page}...")
                
                for issue in issues:
                    try:
                        # Skip if it's a PR (GitHub API returns PRs as issues too)
                        if 'pull_request' in issue:
                            continue
                        
                        issue_number = issue['number']
                        issue_url = issue['html_url']
                        comments_count = issue['comments']
                        
                        # Filter: At least 10 comments
                        if comments_count < 10:
                            continue
                        
                        # Get associated PRs by checking PR references in the issue timeline
                        prs = self.get_associated_prs(owner, repo, issue_number)
                        
                        # Filter: Exactly 1 associated PR
                        if len(prs) != 1:
                            continue
                        
                        # Save the issue-PR pair
                        issue_pr_pairs.append({
                            'issue_url': issue_url,
                            'pr_url': prs[0],
                            'issue_number': issue_number,
                            'comments_count': comments_count
                        })
                        
                        total_issues_filtered += 1
                        print(f"Found matching issue-PR pair: {issue_url} -> {prs[0]}")
                        
                    except Exception as e:
                        print(f"Error processing issue {issue.get('number', 'unknown')}: {e}")
                
                # Move to the next page
                page += 1
                
                # Respect GitHub API rate limits
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching issues: {e}")
                
                # Check if rate limited
                if issues_response.status_code == 403 and 'X-RateLimit-Remaining' in issues_response.headers:
                    remaining = int(issues_response.headers['X-RateLimit-Remaining'])
                    if remaining == 0:
                        reset_time = int(issues_response.headers['X-RateLimit-Reset'])
                        wait_time = max(0, reset_time - time.time()) + 1
                        print(f"Rate limited. Waiting for {wait_time:.0f} seconds...")
                        time.sleep(wait_time)
                        continue
                
                break
        
        print(f"Total issues fetched: {total_issues_fetched}")
        print(f"Total issues matching criteria: {total_issues_filtered}")
        
        return issue_pr_pairs
    
    def get_associated_prs(self, owner, repo, issue_number):
        """Find PRs associated with an issue using the GitHub API."""
        associated_prs = []
        
        # Method 1: Check issue timeline for PR references
        timeline_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/timeline"
        headers = {**self.headers, 'Accept': 'application/vnd.github.mockingbird-preview+json'}
        
        try:
            timeline_response = requests.get(timeline_url, headers=headers)
            timeline_response.raise_for_status()
            timeline = timeline_response.json()
            
            for event in timeline:
                if event.get('event') == 'cross-referenced':
                    source = event.get('source', {})
                    if source.get('type') == 'pull_request':
                        pr_url = source.get('issue', {}).get('html_url')
                        if pr_url:
                            associated_prs.append(pr_url)
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching issue timeline: {e}")
        
        # Method 2: Check issue body and comments for PR references
        if not associated_prs:
            # Get issue details
            issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
            
            try:
                issue_response = requests.get(issue_url, headers=self.headers)
                issue_response.raise_for_status()
                issue_data = issue_response.json()
                
                # Check issue body for PR references
                body = issue_data.get('body', '')
                if body:
                    pr_refs = self.extract_pr_urls(body, owner, repo)
                    associated_prs.extend(pr_refs)
                
                # Check comments for PR references
                comments_url = issue_data.get('comments_url')
                if comments_url:
                    comments_response = requests.get(comments_url, headers=self.headers)
                    comments_response.raise_for_status()
                    comments = comments_response.json()
                    
                    for comment in comments:
                        comment_body = comment.get('body', '')
                        if comment_body:
                            pr_refs = self.extract_pr_urls(comment_body, owner, repo)
                            associated_prs.extend(pr_refs)
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching issue details: {e}")
        
        # Remove duplicates
        return list(set(associated_prs))
    
    def extract_pr_urls(self, text, owner, repo):
        """Extract PR URLs from text content."""
        # Pattern for GitHub PR URLs
        pr_patterns = [
            # Full URLs
            re.compile(r'https://github\.com/[^/]+/[^/]+/pull/\d+'),
            # Relative URLs with owner/repo
            re.compile(rf'/{owner}/{repo}/pull/(\d+)'),
            # PR references like #123 (make sure it's a PR, not an issue)
            re.compile(r'#(\d+)')
        ]
        
        pr_urls = []
        
        # Extract full URLs
        for match in pr_patterns[0].finditer(text):
            pr_urls.append(match.group(0))
        
        # Extract relative URLs
        for match in pr_patterns[1].finditer(text):
            pr_num = match.group(1)
            pr_urls.append(f"https://github.com/{owner}/{repo}/pull/{pr_num}")
        
        # Extract PR references like #123
        # Note: This is less reliable as it could be an issue number
        # We'd need to verify these are PRs
        
        return pr_urls
    
    def save_issue_pr_pairs(self, repo_name, issue_pr_pairs):
        """Save issue-PR pairs to a JSON file."""
        if not issue_pr_pairs:
            print(f"No issue-PR pairs found for {repo_name}")
            return
        
        output_file = os.path.join('issue_pr_pairs', f"{repo_name}_pairs.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'repo_name': repo_name,
                'issue_pr_pairs': issue_pr_pairs
            }, f, indent=2)
        
        print(f"Saved {len(issue_pr_pairs)} issue-PR pairs to {output_file}")
        
        # Also create issue_urls.txt and pr_urls.txt for legacy script compatibility
        issue_urls_file = os.path.join('issue_pr_pairs', f"{repo_name}_issue_urls.txt")
        pr_urls_file = os.path.join('issue_pr_pairs', f"{repo_name}_pr_urls.txt")
        
        with open(issue_urls_file, 'w', encoding='utf-8') as f:
            for pair in issue_pr_pairs:
                f.write(f"{pair['issue_url']}\n")
        
        with open(pr_urls_file, 'w', encoding='utf-8') as f:
            for pair in issue_pr_pairs:
                f.write(f"{pair['pr_url']}\n")
        
        print(f"Created {issue_urls_file} and {pr_urls_file} for legacy script compatibility")
        
        return output_file
    
    def process_repo(self, repo_name, repo_url):
        """Process a single repository to find and save issue-PR pairs."""
        try:
            owner, repo = self.parse_github_repo_url(repo_url)
            print(f"\nProcessing repository: {owner}/{repo} ({repo_name})")
            
            # Get filtered issues
            issue_pr_pairs = self.get_filtered_issues(owner, repo)
            
            # Save issue-PR pairs
            self.save_issue_pr_pairs(repo_name, issue_pr_pairs)
            
            return len(issue_pr_pairs)
            
        except Exception as e:
            print(f"Error processing repository {repo_name}: {e}")
            return 0
    
    def process_issue_pr_pairs(self, repo_name, github_token=None):
        """Process issue-PR pairs using the github_issue_scraper.py script."""
        issue_urls_file = os.path.join('issue_pr_pairs', f"{repo_name}_issue_urls.txt")
        pr_urls_file = os.path.join('issue_pr_pairs', f"{repo_name}_pr_urls.txt")
        
        if not os.path.exists(issue_urls_file) or not os.path.exists(pr_urls_file):
            print(f"Issue-PR files not found for {repo_name}")
            return 0
        
        # Read issue and PR URLs
        with open(issue_urls_file, 'r') as f:
            issue_urls = [line.strip() for line in f if line.strip()]
        
        with open(pr_urls_file, 'r') as f:
            pr_urls = [line.strip() for line in f if line.strip()]
        
        if len(issue_urls) != len(pr_urls):
            print(f"Mismatch in number of issues ({len(issue_urls)}) and PRs ({len(pr_urls)})")
            return 0
        
        token_arg = f"--token {github_token}" if github_token else ""
        successful = 0
        
        for i, (issue_url, pr_url) in enumerate(zip(issue_urls, pr_urls)):
            print(f"\nProcessing {i+1}/{len(issue_urls)}: {issue_url}")
            print(f"With PR: {pr_url}")
            
            try:
                # Construct the command to run the original script
                cmd = f"python github_issue_scraper.py --url {issue_url} --pr {pr_url} {token_arg}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"Success: {issue_url}")
                    successful += 1
                else:
                    print(f"Failed: {issue_url}")
                    print(f"Error output: {result.stderr}")
                    
            except Exception as e:
                print(f"Failed to process {issue_url}: {e}")
            
            # Respect rate limits
            time.sleep(2)
        
        print(f"\nProcessing complete for {repo_name}. Successfully processed {successful} issues.")
        return successful
    
    def generate_qa_pairs(self, repo_name, api_key=None, num_pairs=5):
        """Generate QA pairs for the processed issues."""
        # Find JSON files for this repository
        repo_dir = os.path.join("repoqabench", f"{repo_name}")
        
        if not os.path.exists(repo_dir):
            print(f"Repository directory {repo_dir} not found")
            return 0
        
        json_files = [os.path.join(repo_dir, f) for f in os.listdir(repo_dir) if f.endswith('.json')]
        
        if not json_files:
            print(f"No JSON files found in {repo_dir}")
            return 0
        
        print(f"Found {len(json_files)} JSON files to process for QA extraction.")
        
        api_arg = f"--api-key {api_key}" if api_key else ""
        num_pairs_arg = f"--num-pairs {num_pairs}" if num_pairs != 5 else ""
        successful = 0
        
        for i, json_file in enumerate(json_files):
            print(f"\nProcessing file {i+1}/{len(json_files)}: {json_file}")
            
            try:
                cmd = f"python qa_extractor.py --input {json_file} {api_arg} {num_pairs_arg}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"Success: {json_file}")
                    successful += 1
                else:
                    print(f"Failed: {json_file}")
                    print(f"Error output: {result.stderr}")
                    
            except Exception as e:
                print(f"Failed to process {json_file}: {e}")
            
            # Add a small delay between processing files
            time.sleep(1)
        
        print(f"\nQA extraction complete for {repo_name}. Successfully processed {successful} files.")
        return successful

def main():
    parser = argparse.ArgumentParser(description='Construct RepoQABench dataset from GitHub repositories')
    parser.add_argument('--repos-file', default='repositories.json', help='JSON file containing repositories to process')
    parser.add_argument('--token', help='GitHub API token (override environment variable)')
    parser.add_argument('--openai-key', help='OpenAI API key for QA generation')
    parser.add_argument('--num-pairs', type=int, default=5, help='Number of QA pairs to generate per file')
    parser.add_argument('--skip-filtering', action='store_true', help='Skip issue filtering step')
    parser.add_argument('--skip-scraping', action='store_true', help='Skip issue-PR scraping step')
    parser.add_argument('--skip-qa-gen', action='store_true', help='Skip QA generation step')
    
    args = parser.parse_args()
    
    # Get GitHub token from args or environment
    github_token = args.token or os.environ.get('GITHUB_TOKEN')
    if not github_token:
        print("Warning: No GitHub API token provided. API rate limits will be restricted.")
        print("Set GITHUB_TOKEN environment variable or use --token.")
    
    # Get OpenAI API key from args or environment
    openai_key = args.openai_key or os.environ.get('OPENAI_API_KEY')
    if not openai_key and not args.skip_qa_gen:
        print("Warning: No OpenAI API key provided. QA generation will be limited.")
        print("Set OPENAI_API_KEY environment variable or use --openai-key.")
    
    # Read repositories from JSON file
    try:
        with open(args.repos_file, 'r') as f:
            repositories = json.load(f)
    except Exception as e:
        print(f"Error reading repositories file: {e}")
        return
    
    constructor = RepoQABenchConstructor(github_token=github_token)
    
    # Process each repository
    for repo_data in repositories:
        repo_name = repo_data.get('repo_name')
        repo_url = repo_data.get('repo_url')
        
        if not repo_name or not repo_url:
            print("Invalid repository data, skipping.")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing repository: {repo_name} ({repo_url})")
        print(f"{'='*60}")
        
        # Step 1: Filter issues and find issue-PR pairs
        if not args.skip_filtering:
            num_pairs = constructor.process_repo(repo_name, repo_url)
            print(f"Found {num_pairs} issue-PR pairs for {repo_name}")
        
        # Step 2: Scrape issues and PRs
        if not args.skip_scraping:
            num_processed = constructor.process_issue_pr_pairs(repo_name, github_token)
            print(f"Processed {num_processed} issue-PR pairs for {repo_name}")
        
        # Step 3: Generate QA pairs
        if not args.skip_qa_gen:
            num_qa_generated = constructor.generate_qa_pairs(repo_name, openai_key, args.num_pairs)
            print(f"Generated QA pairs for {num_qa_generated} files from {repo_name}")
    
    print("\nAll repositories processed successfully!")

if __name__ == "__main__":
    main()