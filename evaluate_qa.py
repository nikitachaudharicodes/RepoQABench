import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from evaluate import load as load_metric

# Define supported generative models
SUPPORTED_MODELS = {
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-large": "google/flan-t5-large",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
}

def run_qa_model(model_name, questions, qa_pipeline):
    answers = []
    for question in questions:
        try:
            input_text = f"Question: {question}"
            result = qa_pipeline(input_text)[0]["generated_text"]
            answers.append(result.strip())
        except Exception as e:
            answers.append("<error>")
    return answers

def compute_metrics(preds, refs):
    rouge = load_metric("rouge")
    bertscore = load_metric("bertscore")

    # ROUGE expects plain strings
    rouge_result = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    rouge_score = rouge_result["rougeL"].mid.fmeasure if hasattr(rouge_result["rougeL"], "mid") else float(rouge_result["rougeL"])

    # BERTScore expects string predictions and refs
    bert_score = bertscore.compute(predictions=preds, references=refs, lang="en")["f1"]
    avg_bert_score = sum(bert_score) / len(bert_score) if bert_score else 0.0

    return {
        "ROUGE-L": rouge_score,
        "BERTScore": avg_bert_score
    }

def evaluate_model_on_jsons(path, model_key):
    model_name = SUPPORTED_MODELS[model_key]
    qa_pipeline = pipeline("text2text-generation", model=model_name, tokenizer=model_name, max_length=512)

    benchmark_rows = []
    generated_rows = []

    for root, _, files in os.walk(path):
        for file in files:
            if not file.endswith(".json"):
                continue

            filepath = os.path.join(root, file)
            with open(filepath, "r") as f:
                data = json.load(f)

            questions = data.get("questions", [])
            golden_answers = data.get("golden_answers", [])
            questions_generated = data.get("questions_generated", [])
            golden_answers_generated = data.get("golden_answers_generated", [])

            if not questions or not golden_answers or len(questions) != len(golden_answers):
                continue
            if not questions_generated or not golden_answers_generated or len(questions_generated) != len(golden_answers_generated):
                continue

            pred_benchmark = run_qa_model(model_name, questions, qa_pipeline)
            pred_generated = run_qa_model(model_name, questions_generated, qa_pipeline)

            for q, pred, gold in zip(questions, pred_benchmark, golden_answers):
                benchmark_rows.append({"file": file, "model": model_key, "question": q, "predicted": pred, "gold": gold})
            for q, pred, gold in zip(questions_generated, pred_generated, golden_answers_generated):
                generated_rows.append({"file": file, "model": model_key, "question": q, "predicted": pred, "gold": gold})

    return pd.DataFrame(benchmark_rows), pd.DataFrame(generated_rows)

def plot_scores(df, title, filename):
    pivot = df.pivot_table(index="model", aggfunc="count", values="question")
    plt.figure(figsize=(8, 4))
    sns.barplot(x=pivot.index, y=pivot["question"])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Number of QA Pairs Evaluated")
    plt.savefig(filename)
    plt.close()

def evaluate_and_save(path, models):
    all_benchmark_df = pd.DataFrame()
    all_generated_df = pd.DataFrame()
    scores = []

    for model_key in models:
        benchmark_df, generated_df = evaluate_model_on_jsons(path, model_key)
        all_benchmark_df = pd.concat([all_benchmark_df, benchmark_df])
        all_generated_df = pd.concat([all_generated_df, generated_df])

        scores.append({
            "model": model_key,
            "benchmark_metrics": compute_metrics(benchmark_df["predicted"].tolist(), benchmark_df["gold"].tolist()),
            "generated_metrics": compute_metrics(generated_df["predicted"].tolist(), generated_df["gold"].tolist())
        })

    os.makedirs("results_abs", exist_ok=True)
    all_benchmark_df.to_csv("results_abs/benchmark_questions_abs_eval.csv", index=False)
    all_generated_df.to_csv("results_abs/generated_questions_abs_eval.csv", index=False)

    plot_scores(all_benchmark_df, "Benchmark Questions Coverage per Model", "results_abs/benchmark_abs_plot.png")
    plot_scores(all_generated_df, "Generated Questions Coverage per Model", "results_abs/generated_abs_plot.png")

    pd.DataFrame(scores).to_json("results_abs/qa_model_scores.json", indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to benchmark JSONs")
    parser.add_argument("--models", nargs="+", default=["flan-t5-base", "flan-t5-large", "mistral"], help="Models to compare")
    args = parser.parse_args()

    evaluate_and_save(args.path, args.models)
