"""ROUGE-L for Kairos benchmarking. Pure Python — no external dependencies."""
import re


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def _lcs_length(a, b):
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def compute_rouge_l(predictions, references):
    precisions, recalls, f1s = [], [], []

    for pred, ref in zip(predictions, references):
        pred_tokens = _tokenize(pred)
        ref_tokens = _tokenize(ref)
        if not pred_tokens or not ref_tokens:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue
        lcs = _lcs_length(pred_tokens, ref_tokens)
        p = lcs / len(pred_tokens)
        r = lcs / len(ref_tokens)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    n = len(predictions)
    return {
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "mean_precision": sum(precisions) / n if n else 0,
        "mean_recall": sum(recalls) / n if n else 0,
        "mean_f1": sum(f1s) / n if n else 0,
    }
