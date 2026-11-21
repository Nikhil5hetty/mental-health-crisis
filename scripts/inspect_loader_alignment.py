"""Read-only structural preflight; prints metadata, never dataset excerpts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data/raw/between_help_and_harm'

def read(path):
    return json.loads(path.read_text())

if __name__ == '__main__':
    baseline = read(ROOT / 'human_label/H1_labeled_n206_s42.json')
    paths = sorted((ROOT / 'human_label').glob('H*.json'))
    paths += sorted((ROOT / 'llm_label').glob('*sampled_dataset_n_200_merged_n50-noSeed_156-s42-*.json'))
    paths += sorted((ROOT / 'llm_evaluator/sampled').glob('sampled_lowest_scores_n200*.json'))
    for path in paths:
        if path.name.endswith('merged_labels.json'):
            continue
        rows = read(path)
        print(path.relative_to(ROOT), 'records=',len(rows), 'fields=',list(rows[0]))
        print('inputs type=',type(rows[0].get('inputs')).__name__)
        mismatches = [i for i,(a,b) in enumerate(zip(baseline,rows)) if a['inputs'] != b['inputs']]
        print('exact inputs mismatches=',len(mismatches),'first indices=',mismatches[:10])
        if 'evals' in rows[0]:
            value=rows[0]['evals']
            print('evals type=',type(value).__name__, 'length=',len(value),'element type=',type(value[0]).__name__)
            if isinstance(value[0],dict): print('eval keys=',list(value[0]))
    scores = read(ROOT/'human_label/H1_scores_n206_s42.json')
    print('H1 label-list concatenation versus score-string equality counts:')
    for sep in ['', ' ', '\n', '\n\n']:
        print(repr(sep),sum(sep.join(a['inputs'])==b['inputs'] for a,b in zip(baseline,scores)))
