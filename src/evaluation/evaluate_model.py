import json
from typing import List
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score

def load_predictions(path: str):
    with open(path, "r") as f:
        return json.load(f)

def compute_metrics(predictions: List[dict]):
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    em_total, rouge_total, bleu_total, bert_precisions = [], [], [], []
    
    references, candidates = [], []

    for entry in predictions:
        pred = entry['prediction'].strip()
        gold = entry['golden_answer'].strip()

        # Exact Match
        em = int(pred == gold)
        em_total.append(em)

        # ROUGE
        rouge_score = rouge.score(gold, pred)['rougeL'].fmeasure
        rouge_total.append(rouge_score)

        # BLEU
        bleu = sentence_bleu([gold.split()], pred.split())
        bleu_total.append(bleu)

        # BERTScore input
        candidates.append(pred)
        references.append(gold)

    # BERTScore (batch outside loop)
    P, R, F1 = bert_score(candidates, references, lang='en', verbose=False)
    
    return {
        "Exact Match": sum(em_total) / len(em_total),
        "Average ROUGE-L": sum(rouge_total) / len(rouge_total),
        "Average BLEU": sum(bleu_total) / len(bleu_total),
        "Average BERTScore (F1)": float(F1.mean())
    }

if __name__ == "__main__":
    data = load_predictions("output/CQ7B_output.json")
    metrics = compute_metrics(data)
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")
