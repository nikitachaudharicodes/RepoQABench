import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
import evaluate
from functools import lru_cache
import ast
import re
import subprocess

# ---------------------- Configuration ----------------------
SUPPORTED_MODELS = {
    "roberta": "deepset/roberta-base-squad2",
    "distilbert": "distilbert-base-uncased-distilled-squad",
    "bert_large": "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
}

# Code embedding model for retrieval
CODE_EMBED_MODEL = "microsoft/codebert-base"
# Number of top code snippets to retrieve per question
TOP_K = 3

# Sample GitHub repositories to evaluate (owner, repo)
SAMPLE_REPOS = [
    ("oppia", "oppia"),
    ("astropy", "astropy"),
    ("CiviWiki", "OpenCiviWiki")
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------- Utility: clone/fetch repos ----------------------
def clone_sample_repos(base_path: str):
    """
    Clone or update each sample GitHub repository into base_path.
    """
    os.makedirs(base_path, exist_ok=True)
    for owner, repo in SAMPLE_REPOS:
        local_dir = os.path.join(base_path, f"{owner}_{repo}")
        repo_url = f"https://github.com/{owner}/{repo}.git"
        if not os.path.isdir(local_dir):
            print(f"Cloning {owner}/{repo} into {local_dir}...")
            subprocess.run(["git", "clone", repo_url, local_dir], check=True)
        else:
            print(f"Updating {owner}/{repo} in {local_dir}...")
            subprocess.run(["git", "-C", local_dir, "pull"], check=True)

# ---------------------- Embedding Setup ----------------------
embed_tokenizer = AutoTokenizer.from_pretrained(CODE_EMBED_MODEL)
embed_model = AutoModel.from_pretrained(CODE_EMBED_MODEL).to(device).eval()

@lru_cache(maxsize=512)
def snippet_embedding(text: str) -> torch.Tensor:
    inputs = embed_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

# ---------------------- Local Repository Code Extraction ----------------------

def extract_python_components(file_content):
    functions = []
    classes = []
    try:
        tree = ast.parse(file_content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                snippet = file_content[node.lineno-1:node.end_lineno]
                functions.append(snippet)
            elif isinstance(node, ast.ClassDef):
                snippet = file_content[node.lineno-1:node.end_lineno]
                classes.append(snippet)
    except SyntaxError:
        pass
    return functions + classes


def extract_javascript_components(file_content):
    snippets = []
    func_pattern = r'(function\s+\w+\s*\([^)]*\)\s*{|(?:const|let|var)\s+\w+\s*=\s*\([^)]*\)\s*=>)'
    class_pattern = r'class\s+\w+'
    for match in re.finditer(func_pattern, file_content):
        snippets.append(match.group(0))
    for match in re.finditer(class_pattern, file_content):
        snippets.append(match.group(0))
    return snippets


def extract_code_snippets_from_repo(repo_dir: str) -> list:
    snippets = []
    for root, _, files in os.walk(repo_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1]
            path = os.path.join(root, fname)
            if ext in ['.py', '.js', '.jsx', '.ts', '.tsx']:
                try:
                    content = open(path, 'r', encoding='utf-8').read()
                except:
                    continue
                if ext == '.py':
                    parts = extract_python_components(content)
                else:
                    parts = extract_javascript_components(content)
                for snippet in parts:
                    snippets.append({'content': snippet})
    return snippets

# ---------------------- Retrieval ----------------------

def simple_retrieve(question: str, text_context: str, code_snippets: list, k: int = TOP_K) -> list:
    query = question + "\n" + text_context
    q_emb = snippet_embedding(query)
    sims = []
    for snippet in code_snippets:
        content = snippet.get('content', '')
        if not content:
            continue
        emb = snippet_embedding(content)
        sim = torch.cosine_similarity(q_emb, emb).item()
        sims.append((sim, content))
    top = sorted(sims, key=lambda x: x[0], reverse=True)[:k]
    return [c for sim, c in top if sim > 0]

# ---------------------- QA Inference ----------------------

def run_qa_model(model_name: str, questions: list, contexts: list) -> list:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device).eval()
    answers = []
    for question, context in tqdm(zip(questions, contexts), total=len(questions), desc=f"Answering {model_name}"):
        try:
            inputs = tokenizer.encode_plus(
                question,
                context,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=384,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits) + 1
            answer = tokenizer.decode(
                inputs['input_ids'][0][start_idx:end_idx],
                skip_special_tokens=True
            )
        except:
            answer = ""
        answers.append(answer)
    return answers

# ---------------------- Evaluation & Metrics ----------------------

squad_metric = evaluate.load("squad")
rouge_metric = evaluate.load("rouge")


def evaluate_model_on_jsons(path: str, model_key: str, code_index: dict) -> pd.DataFrame:
    rows = []
    for root, _, files in os.walk(path):
        # load local snippets for this repo
        repo_dir = next((r for r in code_index if root.startswith(r)), None)
        code_snips = code_index.get(repo_dir, [])
        for file in files:
            if not file.endswith('.json'): continue
            data = json.load(open(os.path.join(root, file)))
            questions = data.get('questions', [])
            golds     = data.get('golden_answers', [])
            if not questions or len(questions) != len(golds): continue
            text_ctx = data.get('text_context', '')
            contexts = [
                "\n\n".join([text_ctx] + simple_retrieve(q, text_ctx, code_snips))
                for q in questions
            ]
            preds = run_qa_model(SUPPORTED_MODELS[model_key], questions, contexts)
            for q, p, g in zip(questions, preds, golds):
                rows.append({'model': model_key, 'question': q, 'predicted': p, 'gold': g})
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> dict:
    preds = [{'id': str(i), 'prediction_text': p} for i, p in enumerate(df['predicted'].fillna(''))]
    refs  = [{'id': str(i), 'answers': {'text': [g], 'answer_start': [0]}} for i, g in enumerate(df['gold'].fillna(''))]
    squad = squad_metric.compute(predictions=preds, references=refs)
    rouge = rouge_metric.compute(predictions=df['predicted'].tolist(), references=df['gold'].tolist())
    return {'EM': squad.get('exact_match',0), 'F1': squad.get('f1',0), 'ROUGE-L': rouge.get('rougeL',0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='Root dir for sample repos')
    parser.add_argument('--models', nargs='+', default=list(SUPPORTED_MODELS.keys()))
    args = parser.parse_args()

    # Clone or update sample repos
    clone_sample_repos(args.path)

    # Precompute code snippets for each repo
    code_index = {}
    for dirpath in os.listdir(args.path):
        repo_dir = os.path.join(args.path, dirpath)
        if os.path.isdir(repo_dir):
            print(f"Indexing code snippets for {dirpath}...")
            code_index[repo_dir] = extract_code_snippets_from_repo(repo_dir)

    os.makedirs('results_ex', exist_ok=True)
    metrics_summary = []

    for m in args.models:
        df = evaluate_model_on_jsons(args.path, m, code_index)
        if df.empty:
            print(f'No data for {m}')
            continue
        df.to_csv(f'results_ex/{m}_predictions.csv', index=False)
        metrics = compute_metrics(df)
        metrics_summary.append({'model': m, **metrics})
        print(f'{m}: {metrics}')

    ms_df = pd.DataFrame(metrics_summary)
    ms_df.to_csv('results_ex/metrics_summary.csv', index=False)
    ax = ms_df.set_index('model').plot(kind='bar', figsize=(10,5), ylim=(0,100))
    ax.set_ylabel('Score (%)')
    ax.set_title('QA Metrics by Model')
    plt.tight_layout()
    plt.savefig('results_ex/metrics_plot.png')

if __name__ == '__main__':
    main()