"""
Create 5-fold chromatographic condition-disjoint cross-validation splits.
Matches the evaluation protocol of the ROI paper (Kretschmer et al. 2025).

Regular split:   datasets with different column types in test vs train
Challenging split: datasets with different column AND pH (diff > 1) in test vs train

Output: cv_splits.json with fold assignments for both split types
"""

import yaml, os, json, random
import numpy as np
from collections import defaultdict

REPO_ROOT = '../RepoRT/'
CONFIG_FILE = 'triplet_m05_config.json'
OUTPUT_FILE = 'cv_splits.json'
N_FOLDS = 5
SEED = 42

def load_metadata(repo_root, dataset_ids):
    records = []
    for did in dataset_ids:
        meta_path = os.path.join(repo_root, 'processed_data', did, f'{did}_metadata.yaml')
        if not os.path.exists(meta_path):
            print(f'WARNING: no metadata for {did}')
            continue
        m = yaml.safe_load(open(meta_path))
        col_name = m.get('column', {}).get('name', 'unknown')
        eluent = m.get('eluent', {})
        pH = eluent.get('A', {}).get('pH') or eluent.get('B', {}).get('pH') or 0.0
        records.append({'id': did, 'column': col_name, 'pH': float(pH)})
    return records

def create_regular_folds(records, n_folds, seed):
    """Group by column → distribute groups across folds."""
    rng = random.Random(seed)
    col_groups = defaultdict(list)
    for r in records:
        col_groups[r['column']].append(r['id'])

    # Sort columns by size descending for better balance
    cols = sorted(col_groups.keys(), key=lambda c: -len(col_groups[c]))

    folds = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    for col in cols:
        datasets = col_groups[col]
        rng.shuffle(datasets)
        if len(datasets) <= n_folds:
            # Assign each dataset to a different fold (round-robin by size)
            for i, did in enumerate(datasets):
                target = sorted(range(n_folds), key=lambda f: fold_sizes[f])[i % n_folds]
                folds[target].append(did)
                fold_sizes[target] += 1
        else:
            # Split within column across folds
            chunk = len(datasets) // n_folds
            for f in range(n_folds):
                start = f * chunk
                end = start + chunk if f < n_folds - 1 else len(datasets)
                folds[f].extend(datasets[start:end])
                fold_sizes[f] += end - start

    return folds

def create_challenging_folds(records, n_folds, seed):
    """Group by (column + pH_bin) → distribute groups across folds.
    pH_bin = round(pH) so that pH diff > 1 guarantees different bins.
    """
    rng = random.Random(seed)
    cond_groups = defaultdict(list)
    for r in records:
        pH_bin = round(r['pH'])
        key = (r['column'], pH_bin)
        cond_groups[key].append(r['id'])

    keys = sorted(cond_groups.keys(), key=lambda k: -len(cond_groups[k]))

    folds = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    for key in keys:
        datasets = cond_groups[key]
        rng.shuffle(datasets)
        if len(datasets) <= n_folds:
            for i, did in enumerate(datasets):
                target = sorted(range(n_folds), key=lambda f: fold_sizes[f])[i % n_folds]
                folds[target].append(did)
                fold_sizes[target] += 1
        else:
            chunk = len(datasets) // n_folds
            for f in range(n_folds):
                start = f * chunk
                end = start + chunk if f < n_folds - 1 else len(datasets)
                folds[f].extend(datasets[start:end])
                fold_sizes[f] += end - start

    return folds

def folds_to_cv(folds, n_folds):
    """For each test fold, return (train_ids, test_ids)."""
    cv = []
    for test_fold in range(n_folds):
        test_ids  = folds[test_fold]
        train_ids = [did for f in range(n_folds) if f != test_fold for did in folds[f]]
        cv.append({'train': sorted(train_ids), 'test': sorted(test_ids)})
    return cv

if __name__ == '__main__':
    config = json.load(open(CONFIG_FILE))
    all_ids = config['args']['input']
    print(f'Total datasets: {len(all_ids)}')

    records = load_metadata(REPO_ROOT, all_ids)
    print(f'Loaded metadata: {len(records)} datasets')

    reg_folds  = create_regular_folds(records, N_FOLDS, SEED)
    chal_folds = create_challenging_folds(records, N_FOLDS, SEED)

    reg_cv  = folds_to_cv(reg_folds,  N_FOLDS)
    chal_cv = folds_to_cv(chal_folds, N_FOLDS)

    print('\n--- Regular split fold sizes ---')
    for i, cv in enumerate(reg_cv):
        print(f'  Fold {i}: train={len(cv["train"])}, test={len(cv["test"])}')

    print('\n--- Challenging split fold sizes ---')
    for i, cv in enumerate(chal_cv):
        print(f'  Fold {i}: train={len(cv["train"])}, test={len(cv["test"])}')

    output = {
        'n_folds': N_FOLDS,
        'seed': SEED,
        'regular': reg_cv,
        'challenging': chal_cv
    }
    json.dump(output, open(OUTPUT_FILE, 'w'), indent=2)
    print(f'\nSaved to {OUTPUT_FILE}')
