"""
Benchmark runner for 6 datasets (10-fold CV).
Usage:
    python3 run_benchmark.py --repo_root_folder ../RepoRT/
"""

import os, sys, json, tempfile
import numpy as np
import pandas as pd
import yaml
import torch
from tap import Tap
from typing import Optional, Literal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import Data
from mapping import LADModel


# gradient_length in minutes (for normalization, as in paper Table 2)
DATASETS = [
    ('FEM_long',           '0002', 'FEM_long_RepoRT_processed_0002_withvoid',           60),
    ('IPB_Halle',          '0003', 'IPB_Halle_RepoRT_processed_0003_withvoid',          20),
    ('UniToyama_Atlantis', '0018', 'UniToyama_Atlantis_RepoRT_processed_0018_withvoid', 40),
    ('Eawag_XBridgeC18',   '0019', 'Eawag_XBridgeC18_RepoRT_processed_0019_withvoid',  30),
    ('LIFE_old',           '0054', 'LIFE_old_RepoRT_processed_0054_withvoid',            6),
    ('LIFE_new',           '0055', 'LIFE_new_RepoRT_processed_0055_withvoid',            7),
]


class Args(Tap):
    repo_root_folder: str = '../RepoRT/'
    model: str = 'models/2-step0525.pt'
    splits_folder: str = 'benchmark_splits'
    split_type: Literal['mces', 'uniform', 'both'] = 'both'
    batch_size: int = 256
    gpu: bool = False
    output: Optional[str] = None


def load_model(path):
    path = path if path.endswith('.pt') else path + '.pt'
    kw = {} if torch.cuda.is_available() else {'map_location': 'cpu'}
    model = torch.load(path, weights_only=False, **kw)
    try:
        model.encoder.encoder[0].device = torch.device('cpu')
    except Exception:
        pass
    # chemprop 1.6.1 added several attributes that older saved models lack
    _defaults = {'bond_descriptors': None, 'is_atom_bond_targets': False,
                 'atom_messages': False, 'reaction': False,
                 'undirected': False, 'aggregation': 'mean'}
    for module in model.modules():
        for attr, val in _defaults.items():
            if not hasattr(module, attr):
                setattr(module, attr, val)
        # older MPNEncoder saved dropout as a float; 1.6.1 calls it as nn.Dropout
        if isinstance(getattr(module, 'dropout', None), float):
            import torch.nn as nn
            module.dropout = nn.Dropout(p=module.dropout)
    if hasattr(model, 'extra_storage'):
        data_args = model.extra_storage['data_args']
        scaler    = model.extra_storage['sysfeature_scaler']
    else:
        # Triplet-repo models store data separately in a .pkl file
        import pickle, inspect
        from utils import Data
        pkl_path = path.replace('.pt', '_data.pkl')
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f'No extra_storage in model and no data pkl found at {pkl_path}')
        saved_data = pickle.load(open(pkl_path, 'rb'))
        # mirror only the keys that extra_storage['data_args'] contains in the original 2-step models
        DATA_ARG_KEYS = {
            'use_system_information', 'metadata_void_rt', 'remove_void_compounds',
            'use_usp_codes', 'custom_features', 'use_hsm', 'use_ph', 'use_gradient',
            'use_newonehot', 'custom_column_fields', 'columns_remove_na', 'hsm_fields',
            'graph_mode', 'encoder', 'remove_doublets', 'use_tanaka', 'tanaka_fields',
            'sys_scales', 'solvent_order',
        }
        data_args  = {k: v for k, v in vars(saved_data).items() if k in DATA_ARG_KEYS}
        scaler     = getattr(saved_data, 'sysfeature_scaler', None)
    return model, data_args, scaler


def run_fold(model, data_args_base, scaler, meta_path, train_df, test_df, batch_size, repo_root_folder):
    metadata = yaml.load(open(meta_path), yaml.SafeLoader)
    [meta] = pd.json_normalize(metadata, sep='.').to_dict(orient='records')

    data_args = {**data_args_base, 'repo_root_folder': repo_root_folder, 'remove_doublets': False}

    combined = pd.concat([
        train_df[['smiles', 'rt_minutes']].rename(columns={'rt_minutes': 'rt'}),
        test_df[['smiles']].assign(rt=float('nan')),
    ], ignore_index=True)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        combined.to_csv(f, sep='\t', index=False)
        tmp = f.name

    try:
        d = Data(**data_args)
        d.add_external_data(tmp, metadata=meta, remove_nan_rts=False,
                            tab_mode=True, isomeric=True, split_type='evaluate')
        d.compute_features(mode=None, add_descs=False)
        d.compute_graphs()
        d.split_data((0, 0))
        if scaler is not None:
            d.standardize(other_descriptor_scaler=None,
                          other_sysfeature_scaler=scaler,
                          can_create_new_scaler=False)

        (tg, tx, ts, _), (vg, vx, vs, _), (eg, ex, es, _) = d.get_split_data()
        graphs = np.concatenate((tg, eg, vg))
        X      = np.concatenate((tx, ex, vx)).astype(np.float32)
        X_sys  = np.concatenate((ts, es, vs)).astype(np.float32)

        preds = model.predict(graphs, X, X_sys, batch_size=batch_size,
                              ret_features=False, prog_bar=False)
        order = np.argsort(np.concatenate([d.train_indices, d.test_indices, d.val_indices]))
        d.df['roi'] = preds[order]

        void_thr = (meta.get('column.t0') or 0) * 2
        anchors = d.df[d.df.rt.notna() & (d.df.rt > void_thr)].copy()
        queries = d.df[d.df.rt.isna()].copy()

        mapping = LADModel(anchors, void=void_thr, ols_after=True,
                           ols_discard_if_negative=True, ols_drop_mode='2*median')
        queries = queries.copy()
        queries['rt_pred'] = mapping.get_mapping(queries.roi)
        queries = queries.merge(
            test_df[['smiles', 'rt_minutes']].rename(columns={'rt_minutes': 'rt_true'}),
            on='smiles', how='left')

        err = (queries['rt_pred'] - queries['rt_true']).dropna().abs()
        return {'n': len(err), 'mae': float(err.mean()),
                'medae': float(err.median()), 'rmse': float(np.sqrt((err**2).mean()))}
    finally:
        os.unlink(tmp)


def main():
    args = Args().parse_args()
    if args.gpu:
        torch.set_default_device('cuda')

    print(f'Loading model: {args.model}')
    model, data_args, scaler = load_model(args.model)

    scenarios = []
    if args.split_type in ('mces',    'both'): scenarios.append('mces_10cv')
    if args.split_type in ('uniform', 'both'): scenarios.append('uniform_10cv')

    summary, all_results = [], {}

    for ds_name, ds_id, ds_prefix, gradient_len in DATASETS:
        meta_path = os.path.join(args.repo_root_folder, 'processed_data',
                                 ds_id, f'{ds_id}_metadata.yaml')
        if not os.path.exists(meta_path):
            print(f'SKIP {ds_name}: metadata not found'); continue

        for scenario in scenarios:
            folder_name = f'{ds_prefix}_{scenario}'
            folder = os.path.join(args.splits_folder, folder_name)
            if not os.path.isdir(folder):
                print(f'SKIP {folder_name}: folder not found'); continue

            folds = []
            for fold in range(10):
                tr = os.path.join(folder, f'{folder_name}_fold{fold}_train.csv')
                te = os.path.join(folder, f'{folder_name}_fold{fold}_test.csv')
                if not (os.path.exists(tr) and os.path.exists(te)): continue
                r = run_fold(model, data_args, scaler, meta_path,
                             pd.read_csv(tr), pd.read_csv(te),
                             args.batch_size, args.repo_root_folder)
                folds.append(r)
                print(f'  {ds_name} {scenario} fold{fold}: '
                      f'MAE={r["mae"]:.3f}  MedAE={r["medae"]:.3f}  RMSE={r["rmse"]:.3f} min')

            if not folds: continue
            mae       = [r['mae']   for r in folds]
            medae     = [r['medae'] for r in folds]
            rmse      = [r['rmse']  for r in folds]
            mae_norm  = [r['mae']  / gradient_len * 100 for r in folds]
            row = {'dataset': ds_name, 'scenario': scenario, 'folds': len(folds),
                   'MAE (min)':   f'{np.mean(mae):.3f} ± {np.std(mae):.3f}',
                   'MedAE (min)': f'{np.mean(medae):.3f} ± {np.std(medae):.3f}',
                   'RMSE (min)':  f'{np.mean(rmse):.3f} ± {np.std(rmse):.3f}',
                   'MAE norm (%)': f'{np.mean(mae_norm):.2f} ± {np.std(mae_norm):.2f}'}
            summary.append(row)
            all_results[f'{ds_name}/{scenario}'] = folds
            print(f'\n>>> {ds_name} [{scenario}]  '
                  f'MAE={np.mean(mae):.3f}±{np.std(mae):.3f} min  '
                  f'MAE norm={np.mean(mae_norm):.2f}±{np.std(mae_norm):.2f}%\n')

    print('\n=== Summary ===')
    print(pd.DataFrame(summary).to_string(index=False))

    if args.output:
        json.dump({'summary': summary, 'folds': all_results}, open(args.output, 'w'), indent=2)
        print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
