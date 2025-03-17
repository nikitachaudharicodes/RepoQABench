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
            print("⚠️ Rate limited! Try again later.")
            return None
        return await response.json()

async def fetch_repo_data(session, owner, repo):
    """ Fetch repository metadata, issues, and pull requests """
    repo_url = f"{GITHUB_API_URL}/repos/{owner}/{repo}"
    issues_url = f"{repo_url}/issues?state=all&per_page=100"
    pulls_url = f"{repo_url}/pulls?state=all&per_page=100"

    repo_data = await fetch(session, repo_url)
    issues_data = await fetch(session, issues_url)
    pulls_data = await fetch(session, pulls_url)

    return repo_data, issues_data, pulls_data

async def save_to_db(conn, repo_data, issues_data, pulls_data):
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

    # Save issues
    for issue in issues_data:
        created_at = datetime.strptime(issue['created_at'], "%Y-%m-%dT%H:%M:%SZ")
        closed_at = datetime.strptime(issue['closed_at'], "%Y-%m-%dT%H:%M:%SZ") if issue.get('closed_at') else None

        await conn.execute('''
            INSERT INTO issues (issue_id, repo_id, title, body, author, created_at, closed_at, labels, num_comments)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (issue_id) DO NOTHING
        ''', issue['id'], repo_data['id'], issue['title'], str(issue.get('body', ''))[:5000],  # ✅ Fix applied
           issue['user']['login'], created_at, closed_at, 
           [label['name'] for label in issue['labels']], issue['comments'])

    # Save pull requests
    for pr in pulls_data:
        created_at = datetime.strptime(pr['created_at'], "%Y-%m-%dT%H:%M:%SZ")
        merged_at = datetime.strptime(pr['merged_at'], "%Y-%m-%dT%H:%M:%SZ") if pr.get('merged_at') else None

        await conn.execute('''
            INSERT INTO pull_requests (pr_id, repo_id, title, body, author, created_at, merged_at, changed_files, num_comments)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (pr_id) DO NOTHING
        ''', pr['id'], repo_data['id'], pr['title'], str(pr.get('body', ''))[:5000],  # ✅ Fix applied
           pr['user']['login'], created_at, merged_at, pr.get('changed_files', 0), pr.get('comments', 0))

async def main(repos):
    """ Main function to scrape and store data """
    conn = await asyncpg.connect(**DB_CONFIG)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_repo_data(session, owner, repo) for owner, repo in repos]
        results = await asyncio.gather(*tasks)

        for (repo_data, issues_data, pulls_data) in results:
            await save_to_db(conn, repo_data, issues_data, pulls_data)

    await conn.close()

if __name__ == "__main__":
    asyncio.run(main(repositories_to_scrape))
