"""Reuse the saved final narrative; never send the reference to generation."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent


def main():
    from src.path_utils import load_kairos_env
    from openai import AzureOpenAI
    from rouge_score import rouge_scorer

    source = ROOT / '_processed/Titanic.1997.mkv/checkpoint.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    narrative = data['narratives'][-1]['narrative']
    (OUT / 'input_narrative.txt').write_text(narrative, encoding='utf-8')
    instruction = (
        'Write a detailed factual plot synopsis of the video in approximately 700 words. '
        'Use only the supplied narrative as evidence, not prior knowledge of the film. '
        'Explain the main characters, their motivations and relationships, major events, '
        'causal connections, conflicts, resolutions, and ending where supported. '
        'Preserve any framing story and flashback structure. Use concrete events rather '
        'than thematic commentary. Omit camera descriptions, production logos, incidental '
        'sounds, speculative interpretations, and disconnected details that cannot be '
        'established as plot events. Do not invent missing details or repeat missing-detail '
        'disclaimers. Do not pad to meet the word target. Return only the synopsis in prose '
        'paragraphs, with no title or analysis.\n\nINPUT NARRATIVE:\n' + narrative
    )
    (OUT / 'generation_prompt.txt').write_text(instruction, encoding='utf-8')
    output = OUT / 'generated_summary.txt'
    if not output.exists():
        load_kairos_env(override=True)
        client = AzureOpenAI(api_version=os.environ['GPT_VERSION'],
                             azure_endpoint=os.environ['GPT_ENDPOINT'],
                             api_key=os.environ['GPT_KEY'], max_retries=0)
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=os.environ['GPT_DEPLOYMENT'],
            messages=[{'role': 'system', 'content': 'You are a precise and reliable assistant.'},
                      {'role': 'user', 'content': instruction}],
            temperature=0.2, top_p=1.0, max_tokens=16384, timeout=120)
        summary = response.choices[0].message.content
        if not summary or response.choices[0].finish_reason != 'stop':
            raise RuntimeError('Generation empty or incomplete; inspect response before scoring.')
        output.write_text(summary.strip(), encoding='utf-8')
        metadata = {'deployment': os.environ['GPT_DEPLOYMENT'], 'response_model': response.model,
                    'elapsed_seconds': time.perf_counter() - started,
                    'usage': response.usage.model_dump() if response.usage else None,
                    'source': str(source.relative_to(ROOT)), 'input_field': 'narratives[-1].narrative',
                    'input_sha256': hashlib.sha256(narrative.encode()).hexdigest(),
                    'temperature': 0.2, 'top_p': 1.0, 'reference_in_generation': False}
        (OUT / 'generation_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    reference = (OUT / 'reference_summary.txt').read_text(encoding='utf-8')
    new = output.read_text(encoding='utf-8')
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = {'settings': {'implementation': 'rouge-score', 'use_stemmer': True,
                            'scale': '0–1', 'rougeL': 'whole-summary LCS, not ROUGE-Lsum'},
               'reference_words': len(reference.split()), 'systems': {}}
    for name, text in [('revised', new)]:
        results['systems'][name] = {'words': len(text.split()),
            'scores': {k: dict(zip(['precision', 'recall', 'f1'], v))
                       for k, v in scorer.score(reference, text).items()}}
    results['length_ratio'] = len(new.split()) / len(reference.split())
    # chrF is a supplementary surface-form metric, not a factuality measure.
    from nltk.translate.chrf_score import sentence_chrf
    results['chrf'] = {'score': sentence_chrf(reference, new, min_len=1, max_len=6,
                                             beta=2.0, ignore_whitespace=True),
                       'implementation': 'nltk.sentence_chrf', 'beta': 2,
                       'character_orders': '1-6', 'case_sensitive': True,
                       'ignore_whitespace': True, 'scale': '0-1'}
    (OUT / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    # Keep the interpretation in a separate, hand-reviewed report.
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
