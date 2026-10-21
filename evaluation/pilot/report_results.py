"""Render the concise pilot comparison from the verified, saved results snapshot."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PAPER = 'https://aclanthology.org/2024.findings-acl.239.pdf#page=4'
FIELDS = ['precision', 'recall', 'f1']
ROUGE = ['rouge1', 'rouge2', 'rougeL']


def mean(movies, metric, field):
    return sum((m['bertscore'] if metric == 'bert' else m['rouge_precision_recall_f1'][metric])[field]
               for m in movies) / len(movies) * 100


def render():
    data = json.loads((BASE / 'results.json').read_text(encoding='utf-8'))
    movies = data['movies']
    assert len(movies) == 4 and all(m['saved_metrics_reproduced'] and m['reference_whitespace_normalized_match'] for m in movies)
    groups = [movies[1:], movies]
    averages = [[mean(group, metric, field) for metric in ROUGE + ['bert'] for field in FIELDS] for group in groups]
    baseline_names = list(data['published_zero_shot_test_averages'])
    baselines = {name: data['published_zero_shot_test_averages'][name] +
                      data['published_zero_shot_bertscore_precision_recall_f1'][name] for name in baseline_names}
    reported = [(2,0,'ROUGE-1 F1'),(5,1,'ROUGE-2 F1'),(8,2,'ROUGE-L F1'),
                (9,3,'BERTScore precision*'),(10,4,'BERTScore recall*'),(11,5,'BERTScore F1*')]
    lines = ['# Pilot: does Kairos beat the published zero-shot baselines?', '',
        '**No, not across all measures.** The four-movie average is numerically higher on ROUGE-1/2 F1 '
        'and lower on ROUGE-L F1 than all four zero-shot baselines. Without Titanic, ROUGE-2 is below '
        'FLAN-UL2 but above the other three. BERTScore settings are not aligned, so its higher numbers '
        'do not establish a win.', '',
        'Three movies = Baby Driver, Get Out, The Martian; four adds Titanic. References are the verified '
        'MovieSum Wikipedia summaries, kept out of generation. All values use a 0–100 scale. There are '
        '**12 local values**, not 10: P/R/F1 for each of four metrics. The paper reports only six.', '',
        '| Measure | Three-movie average | Four-movie average | Strongest published zero-shot value | Difference: three / four |',
        '|---|---:|---:|---:|---:|']
    for local_idx, paper_idx, label in reported:
        best = max(values[paper_idx] for values in baselines.values())
        a,b = (values[local_idx] for values in averages)
        lines.append(f'| {label} | {a:.2f} | {b:.2f} | {best:.2f} | {a-best:+.2f} / {b-best:+.2f} |')
    lines += ['', '*BERTScore differences are arithmetic only, not comparable performance margins. '
        'Our unrescaled roberta-large layer-17 scores use standard token truncation; the paper’s '
        'equivalent settings are unverified. Each strongest value is selected independently across baselines.', '',
        '## Margins against each zero-shot baseline', '',
        'Each cell is **three-movie / four-movie difference in score points**. Positive means numerically higher. '
        'No statistical-equivalence threshold was set, so a small difference is not labeled “comparable.”', '',
        '| Measure | FLAN-UL2 | Vicuna | TextRank + Vicuna | Moving-window Vicuna |',
        '|---|---:|---:|---:|---:|']
    for local_idx,paper_idx,label in reported:
        cells = [f'{averages[0][local_idx]-values[paper_idx]:+.2f} / {averages[1][local_idx]-values[paper_idx]:+.2f}'
                 for values in baselines.values()]
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['',f'Published baselines: [MovieSum Table 3]({PAPER}).', '',
        '## Remaining local measures', '',
        'The paper does not publish ROUGE precision or recall separately. These six local values '
        'have no published counterpart to beat:', '',
        '| Metric | Three: precision / recall | Four: precision / recall |',
        '|---|---:|---:|']
    for i,label in enumerate(['ROUGE-1','ROUGE-2','ROUGE-L']):
        cells=[f'{a[i*3]:.2f} / {a[i*3+1]:.2f}' for a in averages]
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', '## Scope', '',
        'This compares selected video diagnostics with published screenplay test-set averages, '
        'not baseline outputs for the same movies. Titanic was used for tuning, the local cuts remain '
        'unverified, and scoring preprocessing may differ. Missing plot events remain a quality concern. '
        'These results do not establish overall superiority.', '',
        'Per-movie scores, all P/R/F1 values, reference checks and hashes are in [results.json](results.json). '
        'Rebuild this report with `.venv\\Scripts\\python.exe evaluation\\pilot\\report_results.py`. '
        'Raw local runs, movies and frozen sources stay untracked.']
    (BASE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    render()
