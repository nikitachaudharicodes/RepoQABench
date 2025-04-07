# RepoQABench: A Repository-Level Code Question Answering Benchmark

## 📚 Overview
RepoQABench is a research-grade benchmark designed to evaluate code question answering (QA) systems at the **repository level**. Unlike typical function-level code QA tasks, RepoQABench ties questions to entire GitHub issues and expects answers that may span multiple files or functions within a codebase. It provides:

- A rich dataset of GitHub issues and associated code context
- Automatically generated and human-verified QA pairs
- Embeddings and retrieval systems for linking natural language to code
- Scripts to evaluate QA model performance using industry-standard metrics

---

## 🗂️ Project Structure

```bash
repoqabench/
├── data/                      # Stores raw and processed input data
│   ├── repoqabench/          # JSON files per GitHub issue (qa_pairs, metadata)
│   └── embedding/            # Code embeddings (file, function, class level)
│       ├── file_embeddings.json
│       ├── function_embeddings.json
│       └── class_embeddings.json
│
├── output/                   # Model predictions and final benchmark output
│   └── CQ7B_output.json      # Example output file with predictions vs golden answers
│
├── src/                      # Source code directory
│   ├── scraping/             # GitHub issue + PR data extraction
│   │   ├── issue_scraper.py
│   │   └── pr_diff_scraper.py
│   │
│   ├── generation/           # QA pair and code context generation
│   │   ├── qa_extractor.py
│   │   └── golden_answer_generator.py
│   │
│   ├── embedding/            # Code embedding generation and retrieval
│   │   ├── create_embedding.py
│   │   └── json_code_retrieval.py
│   │
│   └── evaluation/           # QA prediction evaluation scripts
│       └── evaluate_model.py
│
├── requirements.txt
└── README.md
```

---

## 🔄 Workflow Pipeline

### 1. **Scraping GitHub Issues and PRs**
- **Files:** `src/scraping/issue_scraper.py`, `pr_diff_scraper.py`
- **Goal:** Collect issues, associated PRs, titles, bodies, labels, and code diffs
- **Output:** JSON files in `data/repoqabench/` per issue (e.g., `pandas-dev_pandas_12345.json`)

### 2. **Generating QA Pairs**
- **Files:** `src/generation/qa_extractor.py`, `golden_answer_generator.py`
- **Goal:**
  - Generate natural language questions (e.g., "What is the issue here?")
  - Extract golden answers from PR diffs, commit messages, or human-written responses
- **Output:** Each JSON file contains `qa_pairs` with fields: `question`, `golden_answer`, and `issue_number`

### 3. **Creating Code Embeddings**
- **File:** `src/embedding/create_embedding.py`
- **Goal:**
  - Crawl the entire GitHub repo (via API)
  - Extract code blocks (functions, classes, files)
  - Generate embeddings using CodeBERT
- **Output:**
  - `file_embeddings.json`
  - `function_embeddings.json`
  - `class_embeddings.json`

### 4. **Retrieving Relevant Code**
- **File:** `src/embedding/json_code_retrieval.py`
- **Goal:** Given a question or issue, find relevant code snippets via embedding similarity
- **Usage:** `find_relevant_code_for_issue(issue_number)` or `suggest_code_changes()`

### 5. **Generating Predictions (Optional)**
- **Model Integration:** You may use any language model (e.g., GPT-4, CodeLlama) to predict answers to questions using code + issue context.
- **Predictions:** Should be formatted in `output/CQ7B_output.json`:
```json
{
  "question": "...",
  "prediction": "...",
  "golden_answer": "..."
}
```

### 6. **Evaluating Model Performance**
- **File:** `src/evaluation/evaluate_model.py`
- **Metrics:**
  - Exact Match (EM)
  - BLEU
  - ROUGE-L
  - BERTScore
- **Usage:**
```bash
python src/evaluation/evaluate_model.py
```
- **Output:** Summary statistics of how well your model performed vs golden answers

---

## 📊 Example Evaluation Output
```
Exact Match: 0.1250
Average ROUGE-L: 0.6042
Average BLEU: 0.4238
Average BERTScore (F1): 0.8112
```

---

## 🔬 Use Cases
- Evaluating LLMs like GPT-4, CodeLlama, or custom fine-tuned QA models
- Research on cross-file reasoning and semantic code understanding
- Training retrieval-augmented QA systems with code context

---

## 🛠️ Setup Instructions
1. Clone this repo
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Add your GitHub token to `.env`:
```
GITHUB_TOKEN=ghp_abc123...
```
4. Run embedding or retrieval pipelines from `src/`

---

## 📌 Acknowledgements
- Microsoft CodeBERT
- GitHub REST API v3
- pandas-dev/pandas repo (used for benchmark examples)

---

## 🔮 Future Work
- Automate human verification for QA pairs
- Expand to multiple repos (not just pandas)
- Create a web leaderboard for model submissions
- Add task-specific baselines (e.g., RAG, CoT)

---

## 🧪 Citation
Coming soon!

---

## ❓ Questions?
Ping us at `repoqabench@domain.com` or open an issue.

