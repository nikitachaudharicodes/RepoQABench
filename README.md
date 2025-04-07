# RepoQABench

1. Covers repository level understanding
2. QA pairs include both code snippets & natural language explanations
3. Supports multiple programming languages (Python, Java, C etc)
4. Evaluates both retrieval-based and generative models.


Structure of RepoQABench

Field                        Description
Repository Name              Name of the Repo
Programming Language         python, java, c, etc
Question                     dev-level repo-wide query
Golden Answer(Text & Code)   natural lang explanation + extracted relevant code snippet
Supporting COntext           retrieved files, issue discussions
License                      mit, apache, bsd, etc



Example Usage:

github_issue_scraper.py
python src/scraping/github_issue_scraper.py \
  --url https://github.com/pandas-dev/pandas/issues/13852 \
  --output_dir data/github_issues \
  --token $GITHUB_TOKEN



Folder structure:
repoqabench/
├── data/                    ← Raw and processed JSON files
│   ├── github_issues/       ← All scraped GitHub issue files
│   ├── qa_pairs/            ← Generated QA pairs (per issue or repo)
│   ├── embeddings/          ← File, function, class embeddings
│   ├── retrieval_outputs/   ← Top-k retrieval results
│   └── llm_outputs/         ← Model completions (e.g., CQ7B_output.json)

├── src/                     ← All Python source code
│   ├── scraping/            ← GitHub issue/PR scrapers (JSON only)
│   │   └── github_issue_scraper.py
│   ├── generation/          ← QA and LLM output generation
│   │   ├── qa_extractor.py
│   │   └── model_wrappers/
│   │       ├── CodeQwen7B.py
│   │       └── DeepSeekCoder.py
│   ├── retrieval/           ← Code retrieval logic
│   │   ├── create_embedding.py
│   │   └── json_code_retrieval.py
│   ├── evaluation/          ← Metric scripts (to be added)
│   │   └── evaluate_retrieval.py
│   └── utils/               ← Shared utilities
│       ├── file_io.py
│       └── prompts.py

├── notebooks/               ← Optional: Analysis, demo notebooks

├── README.md                ← Overview, setup, usage instructions
├── requirements.txt         ← Python dependencies
└── .env / .gitignore        ← API keys & exclusions
