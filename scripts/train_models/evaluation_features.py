"""Training-scope-specific FSscore features and strict input/provenance checks."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np


REPO = Path(__file__).resolve().parents[2]


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def outside_checkout(path):
    path = Path(path).resolve()
    if path.is_relative_to(REPO):
        raise ValueError('Restricted inputs and outputs must stay outside the checkout')
    return path


def canonical_ids(smiles):
    from rdkit import Chem
    result = []
    for i, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f'Invalid molecular structure at row {i}')
        result.append(Chem.MolToSmiles(molecule, isomericSmiles=True))
    return result


def training_pair_rows(train, left, right):
    return np.flatnonzero(np.isin(left, train) & np.isin(right, train))


def feature_identity(ids, train, left, right, labels, specification):
    rows = training_pair_rows(train, left, right)
    payload = dict(ids=list(ids), train=list(map(int, train)),
                   pairs=[[int(left[i]), int(right[i]), float(labels[i])] for i in rows],
                   specification=specification)
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return key, rows, payload


class FSScoreFeatures:
    def __init__(self, ids, left, right, labels, cache_dir, checkpoint,
                 epochs=20, seed=42, max_fits=None):
        self.ids = list(ids)
        self.left, self.right = np.asarray(left), np.asarray(right)
        self.labels = np.asarray(labels, dtype=float)
        n = len(self.ids)
        if (len(self.left) != len(self.right) or len(self.labels) != len(self.left)
                or not len(self.labels) or not np.isin(self.labels, [0, 1]).all()
                or min(self.left.min(), self.right.min()) < 0
                or max(self.left.max(), self.right.max()) >= n):
            raise ValueError('Invalid pair endpoints or labels')
        self.cache_dir = outside_checkout(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = Path(checkpoint).resolve()
        self.epochs, self.seed, self.max_fits = epochs, seed, max_fits
        self.memory, self.records = {}, []
        self.fit_count = 0
        source_digest = hashlib.sha256()
        for source in sorted((REPO / 'fsscore/src/fsscore').rglob('*.py')):
            source_digest.update(str(source.relative_to(REPO)).encode())
            source_digest.update(source.read_bytes())
        versions = {}
        for package in ('torch', 'pytorch-lightning', 'torch-geometric', 'rdkit-pypi', 'numpy'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        self.specification = dict(
            epochs=epochs, seed=seed, lr=1e-4, batch_size=32,
            checkpoint_sha256=digest_file(checkpoint), versions=versions,
            python_version=platform.python_version(),
            vendored_source_sha256=source_digest.hexdigest(),
            provider_sha256=digest_file(__file__),
            finetuning_sha256=digest_file(REPO / 'fsscore/src/fsscore/finetuning.py'))

    def __call__(self, train):
        key, rows, identity = feature_identity(
            self.ids, train, self.left, self.right, self.labels, self.specification)
        if key in self.memory:
            return self.memory[key]
        if not len(rows):
            raise ValueError('No eligible pairwise labels within this fitting subset')
        directory = self.cache_dir / key
        directory.mkdir(exist_ok=True)
        scores_file, manifest_file = directory / 'scores.npy', directory / 'manifest.json'
        start = perf_counter()
        if scores_file.exists() and manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            if manifest['identity'] != identity or manifest['scores_sha256'] != digest_file(scores_file):
                raise ValueError('Feature cache provenance mismatch or corruption')
            scores = np.load(scores_file, allow_pickle=False)
            cached = True
        else:
            if self.max_fits is not None and self.fit_count >= self.max_fits:
                raise RuntimeError('Pilot learned-feature fit budget exceeded')
            self.fit_count += 1
            scores = self.fit_scores(directory, rows)
            if scores.shape != (len(self.ids),) or not np.isfinite(scores).all():
                raise ValueError('Invalid computed learned-feature vector')
            np.save(scores_file, scores, allow_pickle=False)
            atomic_json(manifest_file, dict(identity=identity, scores_sha256=digest_file(scores_file)))
            cached = False
        if scores.shape != (len(self.ids),) or not np.isfinite(scores).all():
            raise ValueError('Invalid cached learned-feature vector')
        self.records.append(dict(key=key, train=list(map(int, train)), pair_rows=rows.tolist(),
                                 pair_count=len(rows), cached=cached, seconds=perf_counter() - start))
        self.memory[key] = scores
        return scores

    def fit_scores(self, directory, rows):
        import random
        import sys
        sys.path.insert(0, str(REPO / 'fsscore/src'))
        import torch
        import fsscore.finetuning as ft
        from fsscore.models.ranknet import LitRankNet
        from fsscore.score import Scorer

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        os.environ['WANDB_DISABLED'] = 'true'
        graph_dir = directory / 'graphs'
        graph_dir.mkdir(exist_ok=True)
        old_processed = ft.PROCESSED_PATH
        # The vendored function sweeps temp_* in PROCESSED_PATH on exit. Confine it
        # to this fit so it cannot touch another run or write annotations in the repo.
        ft.PROCESSED_PATH = str(graph_dir)
        try:
            ft.finetune(
                smiles=[(self.ids[self.left[i]], self.ids[self.right[i]]) for i in rows],
                target=self.labels[rows].tolist(),
                graph_datapath=str(graph_dir / 'train.pt'), save_dir=str(directory / 'tuned'),
                model_path=str(self.checkpoint), featurizer='graph_2D',
                n_epochs=self.epochs, val_size=0.0, earlystopping=False, lr=1e-4,
                batch_size=32, num_workers=0, device='cpu', wandb_mode='disabled')
        finally:
            ft.PROCESSED_PATH = old_processed
        checkpoints = sorted((directory / 'tuned').rglob('*best*.ckpt'))
        checkpoints = checkpoints or sorted((directory / 'tuned').rglob('*.ckpt'))
        if not checkpoints:
            raise RuntimeError('Fine-tuning produced no checkpoint')
        model = LitRankNet.load_from_checkpoint(str(checkpoints[-1]), map_location='cpu')
        scorer = Scorer(model=model, num_workers=0, device='cpu', batch_size=128,
                        graph_datapath=str(graph_dir / 'query.pt'), keep_graphs=False)
        return np.asarray(scorer.score(self.ids), dtype=float)
