import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import evaluate

# Define supported models, including a code-specialized model
SUPPORTED_MODELS = {
    "roberta": "deepset/roberta-base-squad2",
    "distilbert": "distilbert-base-uncased-distilled-squad",
    "bert_large": "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
}

# Number of top code snippets to retrieve per question
TOP_K = 3

# Load evaluation metrics
squad_metric = evaluate.load("squad")
rouge_metric = evaluate.load("rouge")


def simple_retrieve(question, text_context, code_snippets, k=TOP_K):
    """
    Score each snippet by overlap with question + text_context tokens,
    return top-k snippet contents.
    """
    query_tokens = set((question + " " + text_context).lower().split())
    scored = []
    for snippet in code_snippets:
        content = snippet.get("content", "")
        tokens = set(content.lower().split())
        overlap = len(tokens & query_tokens)
        scored.append((overlap, content))
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
    return [c for score, c in top if score > 0]


def run_qa_model(model_name, questions, contexts):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

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
        except Exception:
            answer = ""
        answers.append(answer)
    return answers


def evaluate_model_on_jsons(path, model_key):
    rows = []
    for root, _, files in os.walk(path):
        for file in files:
            if not file.endswith('.json'):
                continue
            data = json.load(open(os.path.join(root, file)))
            questions = data.get('questions', [])
            golds = data.get('golden_answers', [])
            if not questions or len(questions) != len(golds):
                continue

            text_ctx = data.get('text_context', '')
            code_snips = data.get('code_context', [])
            contexts = [
                "\n\n".join([text_ctx] + simple_retrieve(q, text_ctx, code_snips))
                for q in questions
            ]

            preds = run_qa_model(SUPPORTED_MODELS[model_key], questions, contexts)
            for q, p, g in zip(questions, preds, golds):
                rows.append({
                    'model': model_key,
                    'question': q,
                    'predicted': p,
                    'gold': g
                })
    return pd.DataFrame(rows)


def compute_metrics(df):
    preds_dict = [{
        'id': str(i),
        'prediction_text': pred
    } for i, pred in enumerate(df['predicted'].fillna(""))]
    refs = [{
        'id': str(i),
        'answers': {'text': [g], 'answer_start': [0]}
    } for i, g in enumerate(df['gold'].fillna(""))]
    squad = squad_metric.compute(
        predictions=preds_dict,
        references=refs
    )

    rouge = rouge_metric.compute(
        predictions=df['predicted'].fillna("").tolist(),
        references=df['gold'].fillna("").tolist()
    )

    return {
        'EM': squad.get('exact_match', 0.0),
        'F1': squad.get('f1', 0.0),
        'ROUGE-L': rouge.get('rougeL', 0.0)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--models', nargs='+', default=list(SUPPORTED_MODELS.keys()))
    args = parser.parse_args()

    os.makedirs('results_ex', exist_ok=True)
    metrics_summary = []

    for m in args.models:
        df = evaluate_model_on_jsons(args.path, m)
        if df.empty:
            print(f'No data for {m}')
            continue
        df.to_csv(f'results_ex/{m}_predictions.csv', index=False)
        metrics = compute_metrics(df)
        metrics_summary.append({'model': m, **metrics})
        print(f'{m}: {metrics}')

    ms_df = pd.DataFrame(metrics_summary)
    ms_df.to_csv('results_ex/metrics_summary.csv', index=False)

    ax = ms_df.set_index('model').plot(
        kind='bar',
        figsize=(10, 5),
        ylim=(0, 100)
    )
    ax.set_ylabel('Score (%)')
    ax.set_title('QA Metrics by Model')
    plt.tight_layout()
    plt.savefig('results_ex/metrics_plot.png')

if __name__ == '__main__':
    main()
