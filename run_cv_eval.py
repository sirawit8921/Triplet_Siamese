"""
Evaluate a trained model using 5-fold condition-disjoint cross-validation.
Matches the evaluation protocol of the ROI paper (Kretschmer et al. 2025).

Usage:
  # Step 1: create splits (run once)
  python3 create_cv_splits.py

  # Step 2: evaluate a trained model
  python3 run_cv_eval.py --model_path triplet_m05.pt --split_type regular
  python3 run_cv_eval.py --model_path triplet_m05.pt --split_type challenging

Output: error rate (%) ± SD (PP) — comparable to paper's Table 1
"""

import json, os, sys, io, pickle
import numpy as np
import torch
from itertools import combinations
from tap import Tap
from typing import Literal


class EvalArgs(Tap):
    model_path: str                                      # path to .pt model file
    split_type: Literal['regular', 'challenging'] = 'regular'
    splits_file: str = 'cv_splits.json'
    repo_root_folder: str = '../RepoRT/'
    batch_size: int = 512
    epsilon: float = 10 / 60                             # 10s in minutes (same as train default)
    verbose: bool = False


class DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes' and not torch.cuda.is_available():
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def load_model_and_data(model_path):
    import re
    mp = model_path if model_path.endswith('.pt') else model_path + '.pt'
    if torch.cuda.is_available():
        model = torch.load(mp, weights_only=False)
    else:
        model = torch.load(mp, map_location='cpu', weights_only=False)
        try:
            model.encoder.encoder[0].device = torch.device('cpu')
        except:
            pass
    base = re.sub(r'_ep\d+(\.pt)?$', '', re.sub(r'\.pt$', '', mp))
    data = DataUnpickler(open(f'{base}_data.pkl', 'rb')).load()
    config = json.load(open(f'{base}_config.json'))
    return model, data, config


def eval_per_dataset(df, preds, epsilon, void_rt=0.0):
    """Compute per-dataset pairwise ordering accuracy."""
    df = df.copy()
    df['pred'] = preds
    per_ds = {}
    for ds_id, grp in df.groupby('dataset_id'):
        rts   = grp['rt'].values
        preds_ = grp['pred'].values
        vrt   = void_rt
        matches = total = 0
        for i, j in combinations(range(len(grp)), 2):
            if rts[i] <= vrt and rts[j] <= vrt:
                continue
            if abs(rts[i] - rts[j]) < epsilon:
                continue
            total += 1
            if (rts[i] < rts[j]) == (preds_[i] < preds_[j]):
                matches += 1
        if total > 0:
            per_ds[ds_id] = 100.0 * (1 - matches / total)   # error rate %
    return per_ds


def evaluate_fold(model, train_data, config, test_ids, repo_root, batch_size, epsilon, verbose=False):
    from utils import Data
    from features import features as feat_obj, parse_feature_spec

    args = config.get('args', {})

    test_data = Data(
        use_system_information=args.get('sysinfo', True),
        metadata_void_rt=(not args.get('no_metadata_void_rt', False)),
        remove_void_compounds=args.get('remove_void_compounds', False),
        void_factor=args.get('void_factor', 2),
        use_usp_codes=args.get('usp_codes', False),
        use_hsm=args.get('columns_use_hsm', True),
        use_tanaka=args.get('columns_use_tanaka', True),
        use_ph=args.get('use_ph', True),
        use_column_onehot=args.get('columns_use_onehot', False),
        repo_root_folder=repo_root,
        graph_mode=True,
        encoder=args.get('mpn_encoder', 'dmpnn'),
        fallback_column=args.get('fallback_column', 'Waters ACQUITY UPLC BEH C18'),
        fallback_metadata=args.get('fallback_metadata', '0045'),
    )

    loaded = []
    for did in test_ids:
        try:
            test_data.add_dataset_id(did, repo_root_folder=repo_root,
                                      isomeric=(not args.get('no_isomeric', False)))
            loaded.append(did)
        except Exception as e:
            if verbose:
                print(f'  SKIP {did}: {e}')

    if not loaded:
        return {}

    test_data.compute_features(**parse_feature_spec(args.get('feature_type', 'None')))
    test_data.compute_graphs()
    test_data.split_data((0, 0))    # all → "train" split internally

    # Filter sys features to match training scaler (drop extra columns not in scaler)
    sys_scaler = getattr(train_data, 'sysfeature_scaler', None)
    if sys_scaler is not None and hasattr(sys_scaler, 'names'):
        scaler_names = sys_scaler.names
        test_feat_names = test_data.system_features
        if set(test_feat_names) != set(scaler_names):
            keep_idx = [i for i, n in enumerate(test_feat_names) if n in set(scaler_names)]
            if verbose:
                extra = set(test_feat_names) - set(scaler_names)
                print(f'\n  [filter] dropping {len(extra)} extra sys features: {sorted(extra)}')
            test_data.train_sys = test_data.train_sys[:, keep_idx]
            if test_data.val_sys is not None and len(test_data.val_sys) > 0:
                test_data.val_sys  = test_data.val_sys[:, keep_idx]
            if test_data.test_sys is not None and len(test_data.test_sys) > 0:
                test_data.test_sys = test_data.test_sys[:, keep_idx]
            test_data.system_features = [test_feat_names[i] for i in keep_idx]

    # Apply stored scalers from training
    if not args.get('no_standardize', False):
        test_data.standardize(
            other_descriptor_scaler=getattr(train_data, 'descriptor_scaler', None),
            other_sysfeature_scaler=sys_scaler,
            can_create_new_scaler=False,
        )

    (graphs, x, sys_feat, y) = test_data.get_split_data((0, 0))[0]

    model.eval()
    with torch.no_grad():
        preds = model.predict(graphs, x.astype('float32'), sys_feat.astype('float32'),
                              batch_size=batch_size)

    void_info = getattr(test_data, 'void_info', {})
    df = test_data.df.iloc[test_data.train_indices].copy()
    df['rt']   = y
    df['pred'] = preds

    per_ds_errors = {}
    for ds_id, grp in df.groupby('dataset_id'):
        vrt = void_info.get(ds_id, 0.0)
        rts    = grp['rt'].values
        preds_ = grp['pred'].values
        matches = total = 0
        for i, j in combinations(range(len(grp)), 2):
            if rts[i] <= vrt and rts[j] <= vrt:
                continue
            if abs(rts[i] - rts[j]) < epsilon:
                continue
            total += 1
            if (rts[i] < rts[j]) == (preds_[i] < preds_[j]):
                matches += 1
        if total > 0:
            per_ds_errors[ds_id] = 100.0 * (1 - matches / total)

    return per_ds_errors


if __name__ == '__main__':
    args = EvalArgs().parse_args()

    if not os.path.exists(args.splits_file):
        print(f'ERROR: {args.splits_file} not found — run create_cv_splits.py first')
        sys.exit(1)

    splits = json.load(open(args.splits_file))
    folds  = splits[args.split_type]
    n_folds = splits['n_folds']

    print(f'Loading model: {args.model_path}')
    model, train_data, config = load_model_and_data(args.model_path)
    model.eval()

    print(f'Split type:  {args.split_type} ({n_folds}-fold)')
    print(f'Epsilon:     {args.epsilon:.4f} min\n')

    fold_means = []
    for fold_i, fold in enumerate(folds):
        test_ids = fold['test']
        print(f'Fold {fold_i+1}/{n_folds}: {len(test_ids)} test datasets', end='', flush=True)
        per_ds = evaluate_fold(model, train_data, config, test_ids,
                               args.repo_root_folder, args.batch_size,
                               args.epsilon, args.verbose)
        if per_ds:
            fold_mean = np.mean(list(per_ds.values()))
            fold_means.append(fold_mean)
            print(f' → error rate: {fold_mean:.2f}%')
            if args.verbose:
                for ds, err in sorted(per_ds.items()):
                    print(f'    {ds}: {err:.2f}%')
        else:
            print(' → no valid pairs (skipped)')

    if fold_means:
        overall_mean = np.mean(fold_means)
        overall_sd   = np.std(fold_means)
        print(f'\n{"="*45}')
        print(f'Model:  {args.model_path}')
        print(f'Split:  {args.split_type}')
        print(f'Error rate: {overall_mean:.2f}% ± {overall_sd:.2f} PP')
        print(f'(n={len(fold_means)} folds)')
