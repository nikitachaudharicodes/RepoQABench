import os
import re
import json
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse
import base64
import logging


class GitHubIssueScraper:
    def __init__(self, github_token=None, output_dir="data/github_issues"):
        """
        Initialize the GitHub issue scraper.
        """
        self.headers = {'Accept': 'application/vnd.github.v3+json'}
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup logger
        logging.basicConfig(
            filename=os.path.join(self.output_dir, "scraper.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def parse_github_url(self, issue_url):
        parsed_url = urlparse(issue_url)
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 4 or path_parts[2] != 'issues':
            raise ValueError(f"Invalid GitHub issue URL: {issue_url}")
        return path_parts[0], path_parts[1], path_parts[3]

    def parse_pr_url(self, pr_url):
        parsed_url = urlparse(pr_url)
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 4 or path_parts[2] != 'pull':
            raise ValueError(f"Invalid GitHub PR URL: {pr_url}")
        return path_parts[0], path_parts[1], path_parts[3]

    def get_issue_data(self, owner, repo, issue_id):
        issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_id}"
        comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_id}/comments"

        issue_response = requests.get(issue_url, headers=self.headers)
        comments_response = requests.get(comments_url, headers=self.headers)
        issue_response.raise_for_status()
        comments_response.raise_for_status()

        issue_data = issue_response.json()
        comments_data = comments_response.json()

        title = issue_data.get('title', '')
        description = issue_data.get('body', '')
        full_description = f"# {title}\n\n{description}"

        comments = [{
            'id': c['id'],
            'user': c['user']['login'],
            'body': c['body']
        } for c in comments_data]

        pr_link = issue_data.get('pull_request', {}).get('html_url')
        if not pr_link:
            pr_matches = re.findall(r'https://github\.com/[^/]+/[^/]+/pull/\d+', description or '')
            if not pr_matches:
                for c in comments:
                    pr_matches = re.findall(r'https://github\.com/[^/]+/[^/]+/pull/\d+', c['body'])
                    if pr_matches:
                        pr_link = pr_matches[0]
                        break

        return {
            'issue_title': title,
            'issue_description': full_description,
            'issue_comments': comments,
            'pr_link': pr_link
        }

    def get_pr_files(self, owner, repo, pr_number):
        files_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
        response = requests.get(files_url, headers=self.headers)
        if response.status_code != 200:
            logging.warning(f"PR fetch failed: {response.status_code} - {response.text}")
            return []

        files_data = response.json()
        files_content = []
        for file in files_data:
            name = file['filename']
            if self._is_code_file(name) and file.get('status') != 'removed':
                try:
                    content = None
                    if file.get('raw_url'):
                        r = requests.get(file['raw_url'], headers=self.headers)
                        if r.status_code == 200:
                            content = r.text
                    if content:
                        files_content.append({'filename': name, 'content': content})
                except Exception as e:
                    logging.error(f"Error downloading {name}: {e}")
                time.sleep(0.5)
        return files_content

    def _is_code_file(self, filename):
        extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.html', '.css', '.go', '.rs']
        return any(filename.endswith(ext) for ext in extensions)

    def scrape_issue(self, issue_url, pr_url=None):
        try:
            owner, repo, issue_id = self.parse_github_url(issue_url)
            repo_name = f"{owner}_{repo}"
            issue_data = self.get_issue_data(owner, repo, issue_id)
            if pr_url:
                issue_data['pr_link'] = pr_url

            code_context = []
            if issue_data['pr_link']:
                try:
                    pr_owner, pr_repo, pr_number = self.parse_pr_url(issue_data['pr_link'])
                    code_context = self.get_pr_files(pr_owner, pr_repo, pr_number)
                except Exception as e:
                    logging.error(f"Error with PR {issue_data['pr_link']}: {e}")

            text_context = issue_data['issue_description']
            for comment in issue_data['issue_comments']:
                text_context += "\n\n" + comment.get('body', '')

            output_data = {
                'repo_name': repo_name,
                'issue_id': issue_id,
                'issue_description': issue_data['issue_description'],
                'issue_comments': issue_data['issue_comments'],
                'text_context': text_context,
                'pr_link': issue_data['pr_link'],
                'code_context': code_context
            }

            filename = f"{repo_name}_{issue_id}.json"
            output_path = os.path.join(self.output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logging.info(f"Scraped {issue_url} → {output_path}")
            print(f"✅ Saved: {output_path}")
            return output_path

        except Exception as e:
            logging.error(f"Scraping failed for {issue_url}: {e}")
            print(f"❌ Error scraping: {e}")
            raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scrape a GitHub issue into a JSON file.")
    parser.add_argument('--url', required=True, help="GitHub issue URL")
    parser.add_argument('--pr', help="Optional PR URL (overrides automatic detection)")
    parser.add_argument('--token', help="GitHub API token")
    parser.add_argument('--output_dir', default="data/github_issues", help="Where to save output JSONs")

    args = parser.parse_args()
    scraper = GitHubIssueScraper(github_token=args.token or os.getenv("GITHUB_TOKEN"),
                                 output_dir=args.output_dir)
    scraper.scrape_issue(args.url, pr_url=args.pr)


if __name__ == "__main__":
    main()
