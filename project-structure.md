1. collect repositories
2. extract qa pairs
3. structuring the dataset
4. evaluating models against repoQA

step1: collect data (scraping repos)
step2: process & structure data
step3: store dataset
step4: model benchmarking


step 1:
- collect repositories (GSoC, GitHub Trending, Apache Foundation)
- extract repository issues, pull request discussions, and commit messages.
- store extracted questions & answers into a structured dataset

how? 
- use github api to pull issues, commits, and PR discussions.
- filtering logic to remove noisy or irrelevant data
- nlp models to refine extracted questions


within a repo, what are our sources?
- issues (primary source)
- pull requests (PRs)
- commit messages
- discussions 
- readME & Docs


problems identified: 
1, open issues might not have comments. what will the gold standard answer be then? should we filter based on the number of comments?

Issues -> Questions
Comments -> Answers
Repository Metadata -> Language, stars, forks
Pull request comments (if useful)
Commit Messages (for historical context)



how do we identify golden answers?
- maintainers response -> check if commenter is the repo owner or contributor
- most upvoted/agreed comment -> count replies or reactions
- combination of comments -> merge multiple relevant responses
- ai generated answer(optional) (if no good answer exists, generate one using gpt-4)




----------------

Data to Scrape:

A. Repository Metadata (Basic info)

repo_id: Unique identifier (from GitHub API)
name: Repository name
owner: Organization or individual maintaining it
stars: Popularity indicator
forks: Number of forks
contributors: Number of active contributors
primary_language: The main language used
license: Legal usage constraints
created_at, updated_at: Timestamps for tracking changes