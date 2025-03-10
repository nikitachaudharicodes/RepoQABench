import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}", 
    "Accept": "application/vnd.github.v3+json"
}

OWNER = "fossology"
REPO = "fossology"

def fetch_issues(owner, repo, state="closed"):
    """
    Fetch closed issues from a a GitHub repository.
    """

    url = f"https://api.github.com/repos/{owner}/{repo}/issues?state={state}&per_page=100"
    response = requests.get(url, headers = HEADERS)

    if response.status_code != 200:
        print(f"Error fetching issues: {response.status_code}")
        print(f"Response: {response.text}") 
        return []
    issues = response.json()
    issue_list = []

    for issue in issues:
        if "pull_request" not in issue:
            issue_data = {
                "repo": repo,
                "issue_number": issue["number"],
                "issue_title": issue["title"],
                "issue_body": issue["body"] or "No Description",
                "comments_url": issue["comments_url"],
                "created_at": issue["created_at"],
                "labels": [label["name"] for label in issue.get("labels", [])]
            }
            issue_data["comments"] = fetch_comments(issue["comments_url"])
            issue_list.append(issue_data)
    return issue_list


def fetch_comments(issue_url):
    """
    Fetch comments for a given issue URL.
    """
    response = requests.get(issue_url, headers = HEADERS)

    if response.status_code != 200:
        print(f"Error fetching comments: {response.status_code}")
        print(f"Response: {response.text}") 
        return []
    
    comments = response.json()
    comments_list = [
        {"author": comment["user"]["login"], "comment": comment["body"]}
        for comment in comments
    ]
    return comments_list

    # for comment in comments:
    #     comment_data = {
    #         "comment_id": comment["id"],
    #         "comment_body": comment["body"] or "No Description",
    #         "created_at": comment["created_at"],
    #         "author": comment["user"]["login"]
    #     }
    #     comment_list.append(comment_data)
    
    # return comment_list


def fetch_repo_metadata(owner, repo):
    """
    Fetch metadata for a given repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(url, headers = HEADERS)

    if response.status_code != 200:
        print(f"Error fetching repository metadata: {response.status_code}")
        print(f"Response: {response.text}") 
        return None
    
    repo_data = response.json()

    metadata = {
        "repo_name": repo_data["full_name"],
        "description": repo_data["description"] or "No Description",
        "language": repo_data["language"],
        "stars": repo_data["stargazers_count"],
        "forks": repo_data["forks_count"],
        "contributors_url": repo_data["contributors_url"],
        "license": repo_data["license"]["name"] if repo_data.get("license") else "No License",
        "open_issues": repo_data["open_issues_count"],
        "created_at": repo_data["created_at"],
        "updated_at": repo_data["updated_at"]
    }

    return metadata


repo_metdata = fetch_repo_metadata(OWNER, REPO)
issues_data = fetch_issues(OWNER, REPO)


benchmark_data = {
    "repository": repo_metdata,
    "qa_pairs": issues_data
}

output_path = "data/fossology_benchmark.json"
os.makedirs("data", exist_ok = True)

with open(output_path, "w") as f:
    json.dump(benchmark_data, f, indent = 4)

print(f"Saved benchmark dataset to {output_path}!")

# if issues_data:
#     with open("fossology_issues.json", "w") as f:
#         json.dump(issues_data, f, indent = 4)
#     print(f"Saved {len(issues_data)} issues from Fossology!")
# else:
#     print("No issues found or request failed.")

    




# response = requests.get("https://api.github.com/user", headers = HEADERS)

# if response.status_code == 200:
#     print("Authenticated successfully")
# else:
#     print("Failed to authenticate")
