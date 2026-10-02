"""Add label-free SYBA, RAscore, and pretrained FSscore to verified pilot features."""

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from evaluation_features import REPO, atomic_json, digest_file, outside_checkout


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--syba-package', required=True, help='Extracted official conda archive')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    start = perf_counter()
    source, output = outside_checkout(args.features), outside_checkout(args.output)
    if output.exists() or output.with_suffix('.manifest.json').exists():
        raise ValueError('Refusing to overwrite full frozen features')
    parent = json.loads(source.with_suffix('.manifest.json').read_text())
    if not parent['label_free'] or digest_file(source) != parent['features_sha256']:
        raise ValueError('Pilot frozen-feature provenance failed')
    with source.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    if any(c in columns for c in ('syba', 'ra', 'fs_base')):
        raise ValueError('Additional score columns already exist')
    ids = [r['smiles'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate row identifiers')
    package = Path(args.syba_package).resolve()
    sys.path.insert(0, str(package / 'site-packages'))
    sys.path.insert(0, str(REPO / 'scripts/tools/rascore'))
    sys.path.insert(0, str(REPO / 'fsscore/src'))
    import torch
    import xgboost
    from syba.syba import SybaClassifier
    import syba.syba as syba_module
    from RAscore_XGB import RAScorerXGB
    from fsscore.models.ranknet import LitRankNet
    from fsscore.score import Scorer
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    output.parent.mkdir(parents=True, exist_ok=True)
    timings = {}
    with threadpool_limits(limits=args.threads):
        begin = perf_counter()
        syba = SybaClassifier()
        syba.fitDefaultScore()
        syba_values = [round(float(syba.predict(smi=s)), 2) for s in ids]
        timings['syba_seconds'] = perf_counter() - begin
        del syba
        begin = perf_counter()
        ra_model = REPO / 'scripts/tools/rascore/models/XGB_chembl_ecfp_counts/model.json'
        ra = RAScorerXGB(model_path=str(ra_model))
        ra.xgb_model.set_param({'nthread': args.threads})
        ra_values = [round(float(ra.predict(s)), 4) for s in ids]
        timings['rascore_seconds'] = perf_counter() - begin
        begin = perf_counter()
        checkpoint = REPO / 'fsscore/models/pretrain_graph_GGLGGL_ep242_best_valloss.ckpt'
        model = LitRankNet.load_from_checkpoint(str(checkpoint), map_location='cpu')
        scorer = Scorer(model=model, num_workers=0, device='cpu', batch_size=128,
                        graph_datapath=str(output.parent / 'base-fsscore-query.pt'), keep_graphs=False)
        fs_values = [float(v) for v in scorer.score(ids)]
        timings['base_fsscore_seconds'] = perf_counter() - begin
    for values in (syba_values, ra_values, fs_values):
        if len(values) != len(rows) or not np.isfinite(values).all():
            raise ValueError('Missing or non-finite frozen scores')
    if not ((np.array(ra_values) >= 0) & (np.array(ra_values) <= 1)).all():
        raise ValueError('RAscore outside its probability range')
    # Preserve every original numeric token so the prior feature matrix is unchanged.
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns + ['syba', 'ra', 'fs_base'])
        writer.writeheader()
        for row, syba_value, ra_value, fs_value in zip(rows, syba_values, ra_values, fs_values):
            writer.writerow(dict(row, syba=syba_value, ra=ra_value, fs_base=fs_value))
    resources = sorted((Path(syba_module.__file__).parent / 'resources').glob('*'))
    weights = {str(p.relative_to(package)): digest_file(p) for p in resources if p.is_file()}
    weights.update(rascore=digest_file(ra_model), pretrained_fsscore=digest_file(checkpoint))
    manifest = dict(label_free=True, profile='full_frozen_tools_and_pretrained_fsscore',
                    n_records=len(rows), n_canonical_groups=parent['n_canonical_groups'],
                    dimensions=parent['dimensions'], features_sha256=digest_file(output),
                    parent_manifest=parent, parent_features_sha256=digest_file(source),
                    weights_sha256=weights,
                    source_sha256={'preparation': digest_file(__file__),
                        'syba': digest_file(syba_module.__file__),
                        'rascore': digest_file(REPO / 'scripts/tools/rascore/RAscore_XGB.py')},
                    syba_package=json.loads((package / 'info/index.json').read_text()),
                    versions={p: importlib.metadata.version(p) for p in
                              ('numpy', 'torch', 'rdkit-pypi', 'pytorch-lightning')},
                    xgboost_version=xgboost.__version__, timings=timings,
                    total_seconds=perf_counter() - start,
                    note='No panel labels read. SYBA/RA rounding follows reproduction wrappers; '
                         'pretrained FSscore is a baseline only, outside the fixed 186-grid.')
    atomic_json(output.with_suffix('.manifest.json'), manifest)
    print(json.dumps(dict(records=len(rows), timings=timings,
                          total_seconds=manifest['total_seconds']), indent=2))


if __name__ == '__main__':
    main()
