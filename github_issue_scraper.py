import os
import re
import json
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse
import base64

class GitHubIssueScraper:
    def __init__(self, github_token=None):
        """
        Initialize the GitHub issue scraper.
        
        Args:
            github_token (str, optional): GitHub API token for authenticated requests.
                                          Higher rate limits with authentication.
        """
        self.headers = {'Accept': 'application/vnd.github.v3+json'}
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"
        
        # Create output directory if it doesn't exist
        os.makedirs("repoqabench", exist_ok=True)
    
    def parse_github_url(self, issue_url):
        """Parse GitHub URL to extract owner, repo, and issue ID."""
        parsed_url = urlparse(issue_url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        if len(path_parts) < 4 or path_parts[2] != 'issues':
            raise ValueError(f"Invalid GitHub issue URL: {issue_url}")
            
        owner = path_parts[0]
        repo = path_parts[1]
        issue_id = path_parts[3]
        
        return owner, repo, issue_id
    
    def parse_pr_url(self, pr_url):
        """Parse GitHub PR URL to extract owner, repo, and PR number."""
        parsed_url = urlparse(pr_url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        if len(path_parts) < 4 or path_parts[2] != 'pull':
            raise ValueError(f"Invalid GitHub PR URL: {pr_url}")
            
        owner = path_parts[0]
        repo = path_parts[1]
        pr_number = path_parts[3]
        
        return owner, repo, pr_number
    
    def get_issue_data(self, owner, repo, issue_id):
        """
        Fetch issue data using the GitHub API.
        
        Returns:
            dict: Issue data including title, description and comments
        """
        # Get issue details
        issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_id}"
        issue_response = requests.get(issue_url, headers=self.headers)
        issue_response.raise_for_status()
        issue_data = issue_response.json()
        
        # Get issue comments
        comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_id}/comments"
        comments_response = requests.get(comments_url, headers=self.headers)
        comments_response.raise_for_status()
        comments_data = comments_response.json()
        
        # Process the data
        issue_title = issue_data.get('title', '')
        issue_description = issue_data.get('body', '')
        
        # Include title in the description
        full_description = f"# {issue_title}\n\n{issue_description}"
        
        # Extract comments text
        comments = []
        for comment in comments_data:
            comments.append({
                'id': comment['id'],
                'user': comment['user']['login'],
                'body': comment['body']
            })
        
        # Find connected PR if any
        pr_link = None
        if issue_data.get('pull_request'):
            pr_link = issue_data['pull_request']['html_url']
        else:
            # Look for PR references in the issue or comments
            pr_pattern = re.compile(r'https://github\.com/[^/]+/[^/]+/pull/\d+')
            
            # Check issue body first
            pr_matches = pr_pattern.findall(issue_description or '')
            if pr_matches:
                pr_link = pr_matches[0]
            else:
                # Check comments
                for comment in comments:
                    pr_matches = pr_pattern.findall(comment['body'] or '')
                    if pr_matches:
                        pr_link = pr_matches[0]
                        break
        
        return {
            'issue_title': issue_title,
            'issue_description': full_description,
            'issue_comments': comments,
            'pr_link': pr_link
        }
    
    def get_pr_files(self, pr_owner, pr_repo, pr_number):
        """Extract files changed by the PR and get their content."""
        # Get the list of files changed by the PR
        files_url = f"https://api.github.com/repos/{pr_owner}/{pr_repo}/pulls/{pr_number}/files"
        files_response = requests.get(files_url, headers=self.headers)
        
        if files_response.status_code != 200:
            print(f"Error fetching PR files: {files_response.status_code} - {files_response.text}")
            return []
            
        files_data = files_response.json()
        
        # Get content for each file
        files_content = []
        for file_data in files_data:
            filename = file_data['filename']
            
            # Only include code files, skip binary files or very large files
            if self._is_code_file(filename) and file_data.get('status') != 'removed':
                try:
                    # Get the file content from the raw URL
                    raw_url = file_data.get('raw_url')
                    if raw_url:
                        content_response = requests.get(raw_url, headers=self.headers)
                        if content_response.status_code == 200:
                            files_content.append({
                                'filename': filename,
                                'content': content_response.text
                            })
                    else:
                        # Alternative: get content from GitHub API
                        contents_url = f"https://api.github.com/repos/{pr_owner}/{pr_repo}/contents/{filename}"
                        contents_response = requests.get(contents_url, headers=self.headers)
                        if contents_response.status_code == 200:
                            content_data = contents_response.json()
                            if isinstance(content_data, dict) and content_data.get('encoding') == 'base64' and content_data.get('content'):
                                content = base64.b64decode(content_data['content']).decode('utf-8', errors='replace')
                                files_content.append({
                                    'filename': filename,
                                    'content': content
                                })
                except Exception as e:
                    print(f"Error getting content for {filename}: {e}")
                    continue
                    
                # Respect GitHub API rate limits
                time.sleep(0.5)
                    
        return files_content
    
    def _is_code_file(self, filename):
        """Determine if a file is likely a code file based on extension."""
        code_extensions = [
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.h', 
            '.hpp', '.cs', '.go', '.rb', '.php', '.swift', '.kt', '.rs', '.sh',
            '.r', '.scala', '.pl', '.pm', '.sql', '.html', '.css', '.scss'
        ]
        return any(filename.endswith(ext) for ext in code_extensions)
    
    def scrape_issue(self, issue_url, pr_url=None):
        """
        Scrape a GitHub issue and create a JSON file with the data.
        
        Args:
            issue_url (str): URL to the GitHub issue
            pr_url (str, optional): URL to the associated PR
        
        Returns:
            str: Path to the created JSON file
        """
        try:
            # Parse the GitHub URL
            owner, repo, issue_id = self.parse_github_url(issue_url)
            repo_name = f"{owner}_{repo}"
            os.makedirs(f"repoqabench/{repo_name}", exist_ok=True)
            
            # Get issue data
            issue_data = self.get_issue_data(owner, repo, issue_id)
            
            # Override PR link if provided
            if pr_url:
                issue_data['pr_link'] = pr_url
            
            # Get PR files content if PR link exists
            code_context = []
            if issue_data['pr_link']:
                try:
                    pr_owner, pr_repo, pr_number = self.parse_pr_url(issue_data['pr_link'])
                    code_context = self.get_pr_files(pr_owner, pr_repo, pr_number)
                except Exception as e:
                    print(f"Error processing PR {issue_data['pr_link']}: {e}")
            
            # Combine issue title, description and comments for text_context
            text_context = issue_data['issue_description'] or ""
            for comment in issue_data['issue_comments']:
                text_context += "\n\n" + (comment['body'] or "")
            
            # Create the output data structure
            output_data = {
                'repo_name': repo_name,
                'issue_id': issue_id,
                'issue_description': issue_data['issue_description'],
                'issue_comments': issue_data['issue_comments'],
                'text_context': text_context,
                'pr_link': issue_data['pr_link'],
                'code_context': code_context
            }
            
            # Write to JSON file
            output_filename = f"{repo_name}_{issue_id}.json"
            output_path = os.path.join(f"repoqabench/{repo_name}", output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            print(f"Successfully scraped issue and saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error scraping issue {issue_url}: {e}")
            raise

def main():
    """Main function to run the scraper with command line args."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Scrape GitHub issues into JSON files')
    parser.add_argument('--url', help='GitHub issue URL to scrape')
    parser.add_argument('--pr', help='GitHub PR URL to associate with this issue (optional)')
    parser.add_argument('--token', help='GitHub API token (override environment variable)')
    
    args = parser.parse_args()
    
    # Get token from environment variable if not provided in command line
    github_token = args.token or os.environ.get('GITHUB_TOKEN')
    
    scraper = GitHubIssueScraper(github_token=github_token)
    scraper.scrape_issue(args.url, pr_url=args.pr)

if __name__ == "__main__":
    main()