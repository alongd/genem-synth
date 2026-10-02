"""Frozen SA/SC scores and CheMeleon embeddings for the explicitly reduced pilot."""

import argparse
import contextlib
import importlib.metadata
import io
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from rdkit import Chem
import torch

from evaluation_features import REPO, atomic_json, canonical_ids, digest_file, outside_checkout


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regression', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    output = outside_checkout(args.output)
    if output.exists():
        raise ValueError('Refusing to overwrite frozen features')
    output.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    df = pd.read_csv(outside_checkout(args.regression), usecols=['smiles'])
    if df.smiles.duplicated().any():
        raise ValueError('Duplicate row identifiers')
    groups = canonical_ids(df.smiles.tolist())
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    sys.path.insert(0, str(REPO / 'scripts/tools/sascore'))
    sys.path.insert(0, str(REPO / 'scripts/tools/scscore/scscore'))
    sys.path.insert(0, str(REPO / 'scripts'))
    import sascorer
    from standalone_model_numpy import SCScorer
    from tools.chemeleon.chemeleon_fingerprint import CheMeleonFingerprint
    sc_weights = REPO / 'scripts/tools/scscore/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz'
    sc = SCScorer()
    with contextlib.redirect_stdout(io.StringIO()):
        sc.restore(str(sc_weights))
    sa_values, sc_values = [], []
    for value in df.smiles:
        sa_values.append(round(sascorer.calculateScore(Chem.MolFromSmiles(value)), 2))
        sc_values.append(round(sc.get_score_from_smi(value)[1], 2))
    df['sa'], df['sc'] = sa_values, sc_values
    score_seconds = perf_counter() - start
    weights = REPO / 'models/chemeleon_mp.pt'
    encoder = CheMeleonFingerprint(device='cpu', weights_path=weights)
    batches = []
    with torch.no_grad():
        for begin in range(0, len(df), 64):
            batches.append(encoder(df.smiles.iloc[begin:begin + 64].tolist()))
    embedded = np.concatenate(batches)
    if not np.isfinite(embedded).all() or not np.isfinite(df[['sa', 'sc']].to_numpy()).all():
        raise ValueError('Frozen scoring failed')
    result = pd.concat([df.reset_index(drop=True), pd.DataFrame(embedded,
                       columns=[f'fp_{i}' for i in range(embedded.shape[1])])], axis=1)
    result.to_csv(output, index=False)
    metadata = dict(
        label_free=True, profile='pilot_sa_sc_chemeleon',
        n_records=len(df), n_canonical_groups=len(set(groups)), dimensions=embedded.shape[1],
        features_sha256=digest_file(output), input_sha256=digest_file(args.regression),
        weights_sha256={'sc': digest_file(sc_weights), 'chemeleon': digest_file(weights)},
        script_sha256=digest_file(__file__),
        versions={p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'chemprop', 'rdkit']},
        score_seconds=score_seconds, total_seconds=perf_counter() - start,
        note='No annotation columns are read. SA and SC rounding matches the reproduction wrappers. '
             'The full sweep additionally needs frozen SYBA and RAscore values.')
    atomic_json(output.with_suffix('.manifest.json'), metadata)
    print(f"Frozen features: {len(df)} rows; {len(set(groups))} canonical groups; "
          f"{embedded.shape[1]} embedding dimensions; {metadata['total_seconds']:.1f} seconds", flush=True)


if __name__ == '__main__':
    main()
