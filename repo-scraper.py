import asyncio
import aiohttp
import asyncpg
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# GitHub API authentication
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# PostgreSQL configuration
DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# List of repositories to scrape (owner, repo_name)
repositories_to_scrape = [
    ("processing", "p5.js-web-editor"),
    ("processing", "p5.js-website"),
    ("connieliu0", "p5.js-showcase"),
    ("processing", "p5.js"),
    ("jina-ai", "GSoC"),
    ("pandas-dev", "pandas"),
    ("numpy", "numpy"),
    ("facebookresearch", "faiss"),
    ("fossology", "fossology"),
]

async def fetch(session, url):
    """ Fetch data from GitHub API with error handling """
    async with session.get(url, headers=HEADERS) as response:
        if response.status == 403:
            print(f"⚠️ Rate limited! Skipping {url}")
            return None
        return await response.json()

async def fetch_paginated(session, url):
    """ Fetch all paginated results from GitHub API """
    results = []
    page = 1
    while True:
        paginated_url = f"{url}&page={page}"
        data = await fetch(session, paginated_url)
        if not data:
            break  # Stop when no more data is returned
        results.extend(data)
        page += 1
        if len(data) < 100:  # If less than 100, it's the last page
            break
    return results

async def fetch_repo_data(session, owner, repo):
    """ Fetch repository metadata, issues, and pull requests with pagination """
    repo_url = f"{GITHUB_API_URL}/repos/{owner}/{repo}"
    issues_url = f"{repo_url}/issues?state=all&per_page=100"
    pulls_url = f"{repo_url}/pulls?state=all&per_page=100"

    repo_data = await fetch(session, repo_url)
    issues_data = await fetch_paginated(session, issues_url)
    pulls_data = await fetch_paginated(session, pulls_url)

    return repo_data, issues_data, pulls_data

async def fetch_total_comments(session, comments_url):
    """ Fetch the total number of PR comments from the API """
    comments_data = await fetch(session, comments_url)
    if comments_data is None:
        return 0
    return len(comments_data)

async def fetch_pr_details(session, pr_url):
    """ Fetch PR details including changed files """
    pr_details = await fetch(session, pr_url)
    if pr_details is None:
        return 0
    return pr_details.get('changed_files', 0)

async def save_to_db(conn, repo_data, issues_data, pulls_data, session):
    """ Save scraped data to PostgreSQL """
    if not repo_data:
        return
    
    # Convert timestamps
    created_at = datetime.strptime(repo_data['created_at'], "%Y-%m-%dT%H:%M:%SZ")
    updated_at = datetime.strptime(repo_data['updated_at'], "%Y-%m-%dT%H:%M:%SZ")

    # Save repository metadata
    await conn.execute('''
        INSERT INTO repositories (repo_id, name, owner, stars, forks, primary_language, license, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (repo_id) DO NOTHING
    ''', repo_data['id'], repo_data['name'], repo_data['owner']['login'], repo_data['stargazers_count'],
       repo_data['forks_count'], repo_data['language'], repo_data['license']['name'] if repo_data['license'] else None,
       created_at, updated_at)

    # Save issues (apply filtering for open issues)
    for issue in issues_data:
        issue_state = issue['state']  # "open" or "closed"
        num_comments = issue.get('comments', 0)  # Default to 0 if missing

        # Include only closed issues or open issues with more than 2 comments
        if issue_state == "closed" or num_comments > 2:
            created_at = datetime.strptime(issue['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            closed_at = datetime.strptime(issue['closed_at'], "%Y-%m-%dT%H:%M:%SZ") if issue.get('closed_at') else None

            await conn.execute('''
                INSERT INTO issues (issue_id, repo_id, title, body, author, created_at, closed_at, labels, num_comments)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (issue_id) DO NOTHING
            ''', issue['id'], repo_data['id'], issue['title'], str(issue.get('body', ''))[:5000], 
               issue['user']['login'], created_at, closed_at, 
               [label['name'] for label in issue.get('labels', [])], num_comments)

    # Save pull requests
    for pr in pulls_data:
        created_at = datetime.strptime(pr['created_at'], "%Y-%m-%dT%H:%M:%SZ")
        merged_at = datetime.strptime(pr['merged_at'], "%Y-%m-%dT%H:%M:%SZ") if pr.get('merged_at') else None

        # Fetch actual number of comments using API calls
        comments_count = await fetch_total_comments(session, pr['comments_url'])
        review_comments_count = await fetch_total_comments(session, pr['review_comments_url'])
        total_comments = comments_count + review_comments_count

        # Fetch number of changed files
        changed_files = await fetch_pr_details(session, pr['url'])

        await conn.execute('''
            INSERT INTO pull_requests (pr_id, repo_id, title, body, author, created_at, merged_at, changed_files, num_comments)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (pr_id) DO NOTHING
        ''', pr['id'], repo_data['id'], pr['title'], str(pr.get('body', ''))[:5000],
           pr['user']['login'], created_at, merged_at, changed_files, total_comments)

async def main(repos):
    """ Main function to scrape and store data """
    conn = await asyncpg.connect(**DB_CONFIG)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_repo_data(session, owner, repo) for owner, repo in repos]
        results = await asyncio.gather(*tasks)

        for (repo_data, issues_data, pulls_data) in results:
            await save_to_db(conn, repo_data, issues_data, pulls_data, session)

    await conn.close()

if __name__ == "__main__":
    asyncio.run(main(repositories_to_scrape))
    


