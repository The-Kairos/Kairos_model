"""BERTScore wrapper for Kairos benchmarking."""
import gc
import torch
from bert_score import score as bert_score


def compute_bertscore(predictions, references, lang="en",
                      model_type="microsoft/deberta-large-mnli", batch_size=8):
    """
    Compute BERTScore F1 between prediction/reference pairs.

    Uses deberta-large-mnli (350M params) instead of xlarge (900M) to avoid
    OOM on long scene descriptions.  Processes in small batches.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    P, R, F1 = bert_score(
        predictions, references,
        lang=lang, model_type=model_type, verbose=False,
        batch_size=batch_size, device="cuda:0",
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    return {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F1.tolist(),
        "mean_precision": P.mean().item(),
        "mean_recall": R.mean().item(),
        "mean_f1": F1.mean().item(),
    }
