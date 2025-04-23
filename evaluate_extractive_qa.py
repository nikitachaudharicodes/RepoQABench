import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Define supported models
SUPPORTED_MODELS = {
    "bert": "deepset/bert-base-cased-squad2",
    "roberta": "deepset/roberta-base-squad2",
    "distilbert": "distilbert-base-uncased-distilled-squad"
}

def run_qa_model(model_name, questions, context):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    answers = []
    for question in tqdm(questions, desc=f"Answering ({model_name})"):
        try:
            inputs = tokenizer.encode_plus(
                question,
                context,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=384,
                truncation=True
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                start_scores = outputs.start_logits
                end_scores = outputs.end_logits

            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores) + 1
            answer_tokens = input_ids[0][start_index:end_index]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            answers.append(answer)
        except Exception as e:
            print(f"[ERROR] {e} for question: {question}")
            answers.append("<error>")
    return answers


def evaluate_model_on_jsons(path, model_key):
    model_name = SUPPORTED_MODELS[model_key]

    benchmark_rows = []
    generated_rows = []

    for root, _, files in os.walk(path):
        files = [f for f in files if f.endswith(".json")]
        for file in tqdm(files, desc=f"Evaluating {model_key}"):
            filepath = os.path.join(root, file)
            with open(filepath, "r") as f:
                data = json.load(f)

            # Load questions and golden answers
            questions = data.get("questions", [])
            golden_answers = data.get("golden_answers", [])
            questions_generated = data.get("questions_generated", [])
            golden_answers_generated = data.get("golden_answers_generated", [])

            # Build the real context from text + code
            context_parts = []
            if data.get("text_context"):
                context_parts.append(data["text_context"])
            for snippet in data.get("code_context", []):
                content = snippet.get("content")
                if content:
                    context_parts.append(content)
            context = "\n\n".join(context_parts)

            # Skip if mismatched
            if not questions or not golden_answers or len(questions) != len(golden_answers):
                continue
            if not questions_generated or not golden_answers_generated or len(questions_generated) != len(golden_answers_generated):
                continue

            # Run QA
            pred_benchmark = run_qa_model(model_name, questions, context)
            pred_generated = run_qa_model(model_name, questions_generated, context)

            # Collect rows
            for q, pred, gold in zip(questions, pred_benchmark, golden_answers):
                benchmark_rows.append({
                    "file": file,
                    "model": model_key,
                    "question": q,
                    "predicted": pred,
                    "gold": gold
                })
            for q, pred, gold in zip(questions_generated, pred_generated, golden_answers_generated):
                generated_rows.append({
                    "file": file,
                    "model": model_key,
                    "question": q,
                    "predicted": pred,
                    "gold": gold
                })

    return pd.DataFrame(benchmark_rows), pd.DataFrame(generated_rows)


def plot_scores(df, title, filename):
    pivot = df.pivot_table(index="model", values="question", aggfunc="count")
    plt.figure(figsize=(8, 4))
    sns.barplot(x=pivot.index, y=pivot["question"])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Number of QA Pairs Evaluated")
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to benchmark JSONs")
    parser.add_argument(
        "--models", nargs="+",
        default=["bert", "roberta", "distilbert"],
        help="Models to compare"
    )
    args = parser.parse_args()

    all_benchmark_df = pd.DataFrame()
    all_generated_df = pd.DataFrame()

    for model_key in args.models:
        benchmark_df, generated_df = evaluate_model_on_jsons(args.path, model_key)
        all_benchmark_df = pd.concat([all_benchmark_df, benchmark_df], ignore_index=True)
        all_generated_df = pd.concat([all_generated_df, generated_df], ignore_index=True)

    all_benchmark_df.to_csv("benchmark_questions_eval.csv", index=False)
    all_generated_df.to_csv("generated_questions_eval.csv", index=False)

    plot_scores(all_benchmark_df, "Benchmark Questions Coverage per Model", "benchmark_plot.png")
    plot_scores(all_generated_df, "Generated Questions Coverage per Model", "generated_plot.png")

    print("Evaluation completed. Results saved to CSV files and plots generated.")
