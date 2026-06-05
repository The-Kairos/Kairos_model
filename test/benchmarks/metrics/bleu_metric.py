"""BLEU (1-4) for Kairos benchmarking. Pure Python — no external dependencies."""
import math
import re
from collections import Counter


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _modified_precision(ref_tokens, pred_tokens, n):
    pred_ng = Counter(_ngrams(pred_tokens, n))
    ref_ng = Counter(_ngrams(ref_tokens, n))
    clipped = {ng: min(count, ref_ng.get(ng, 0)) for ng, count in pred_ng.items()}
    numerator = sum(clipped.values())
    denominator = max(sum(pred_ng.values()), 1)
    return numerator / denominator


def _sentence_bleu(ref_tokens, pred_tokens, weights):
    if not pred_tokens:
        return 0.0

    bp = 1.0
    if len(pred_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(pred_tokens))

    log_avg = 0.0
    for n, w in enumerate(weights, 1):
        if w == 0:
            continue
        p = _modified_precision(ref_tokens, pred_tokens, n)
        if p == 0:
            return 0.0
        log_avg += w * math.log(p)

    return bp * math.exp(log_avg)


def compute_bleu(predictions, references):
    weights_map = {
        "bleu_1": (1.0, 0, 0, 0),
        "bleu_2": (0.5, 0.5, 0, 0),
        "bleu_3": (0.333, 0.333, 0.334, 0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }

    results = {k: [] for k in weights_map}

    for pred, ref in zip(predictions, references):
        ref_tokens = _tokenize(ref)
        pred_tokens = _tokenize(pred)
        for key, weights in weights_map.items():
            score = _sentence_bleu(ref_tokens, pred_tokens, weights)
            results[key].append(score)

    n = len(predictions)
    output = {}
    for key in weights_map:
        output[key] = results[key]
        output[f"mean_{key}"] = sum(results[key]) / n if n else 0

    return output
