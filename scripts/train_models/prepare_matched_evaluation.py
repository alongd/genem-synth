"""Reproduce an approved protocol and separate fitting inputs from human outcomes."""
import argparse
import csv
import json
import os
import statistics
from pathlib import Path

from evaluation_features import atomic_json, canonical_ids, digest_file, outside_checkout
from matched_inputs import arm_inputs, digest, mutate, numerical, ratings, training_payload


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def prepare(protocol, raw, reference_path, output):
    protocol, output = Path(protocol), outside_checkout(output)
    pins = json.loads((protocol / 'prepared-sha256.json').read_text())
    # Accept the pinned artifact structure, never replace its approval-time hashes.
    if isinstance(pins, dict) and 'sha256' in pins:
        pins = pins['sha256']
    for name, expected in pins.items():
        if digest_file(protocol / name) != expected:
            raise ValueError(f'Approved protocol changed: {name}')
    plan = json.loads((protocol / 'protocol-plan.json').read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh preparation directory')
    output.mkdir(parents=True, exist_ok=True)
    rows, reference = read_csv(raw), read_csv(reference_path)
    ids = [r['smiles'] for r in reference]
    people = [k[:-5] for k in reference[0] if k.endswith('_mean')]
    assert len(people) == 3 and len(ids) == len(set(ids)) == 500 and len(rows) == 291
    import hashlib
    groups = [hashlib.sha256(s.encode()).hexdigest() for s in canonical_ids(ids)]
    assert groups == plan['group_hashes']
    average = ratings(rows, people, ids)
    primary = {i for i in range(len(ids)) if all((i, p) in average for p in people)}
    assert sorted(primary) == plan['primary_records']
    for i, ref in enumerate(reference):
        for person in people:
            expected, observed = numerical(ref[person + '_mean']), average.get((i, person))
            assert (expected is None) == (observed is None)
            assert expected is None or abs(expected - observed) < 1e-12
        assert abs(statistics.mean(average[i, p] for p in people if (i, p) in average)
                   - float(ref['consensus_score'])) < 1e-12
    truth = []
    for a, omitted in enumerate(people, 1):
        retained = [p for p in people if p != omitted]
        target, pairs = arm_inputs(rows, retained, primary, ids)
        approved = plan['arms'][a - 1]
        assert digest(target) == approved['target_sha256']
        assert digest(pairs) == approved['pairs_sha256']
        assert (target, pairs) == arm_inputs(mutate(rows, [omitted], ids), retained, primary, ids)
        for fold in plan['outer_folds']:
            test_groups = {groups[i] for i in fold['test'] + fold['secondary_query']}
            heldout = {i for i, g in enumerate(groups) if g in test_groups}
            changed_target, changed_pairs = arm_inputs(mutate(rows, retained, ids, heldout), retained, primary, ids)
            assert training_payload(target, pairs, fold['train']) == training_payload(
                changed_target, changed_pairs, fold['train'])
        for boundary in approved['boundaries']:
            assert digest(training_payload(target, pairs, boundary['train'])) == boundary['payload_sha256']
        atomic_json(output / f'arm-{a}.json', dict(chemist=a, ids=ids, target=target, pairs=pairs))
        for i in range(len(ids)):
            others = [average[i, p] for p in retained if (i, p) in average]
            truth.append(dict(chemist=a, row=i, primary=i in primary, human=average.get((i, omitted)),
                              target=statistics.mean(others), reference_raters=len(others)))
    atomic_json(output / 'plan.json', plan)
    evaluation = output / 'evaluation-only'
    evaluation.mkdir()
    atomic_json(evaluation / 'human-outcomes.json', truth)
    manifest = dict(status='passed', no_model_fitting=True, approved_protocol=pins,
                    source_inputs={str(Path(p).resolve()): digest_file(p) for p in (raw, reference_path)},
                    output_sha256={str(p.relative_to(output)): digest_file(p)
                                   for p in sorted(output.rglob('*.json'))},
                    preparation_source_sha256={p.name: digest_file(p) for p in
                                              [Path(__file__), Path(__file__).with_name('matched_inputs.py')]},
                    primary_records=425, secondary_records=75,
                    mutation_checks='All three omitted-rater and all 15 outer-test boundary checks passed')
    atomic_json(output / 'preparation-audit.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('status', 'no_model_fitting', 'primary_records',
                                             'secondary_records', 'mutation_checks')}))


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('protocol', 'raw', 'reference', 'output'):
        parser.add_argument('--' + arg, required=True)
    args = parser.parse_args()
    prepare(args.protocol, args.raw, args.reference, args.output)
