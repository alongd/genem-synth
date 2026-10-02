"""Pure arm-specific reconstruction; omitted outcomes are never returned to fitting."""
import copy
import hashlib
import json
import statistics
from collections import defaultdict


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def numerical(value):
    if value is None or str(value).strip() == '':
        return None
    result = float(value)
    if not 1 <= result <= 5:
        raise ValueError('Nonfinite or out-of-scale rating')
    return result


def ratings(rows, people, ids):
    lookup = {s: i for i, s in enumerate(ids)}
    collected = defaultdict(list)
    for row in rows:
        for endpoint in ('A', 'B'):
            index = lookup[row[f'Molecule {endpoint} SMILES']]
            for person in people:
                value = numerical(row[f'{person} - Score {endpoint}'])
                if value is not None:
                    collected[index, person].append(value)
    return {k: statistics.mean(v) for k, v in collected.items()}


def arm_inputs(rows, retained, primary, ids):
    if len(retained) != 2 or len(set(retained)) != 2:
        raise ValueError('Exactly two distinct retained chemists required')
    lookup = {s: i for i, s in enumerate(ids)}
    average = ratings(rows, retained, ids)
    target = {i: statistics.mean(average[i, person] for person in retained) for i in primary}
    pairs = []
    for index, row in enumerate(rows):
        left, right = [lookup[row[f'Molecule {e} SMILES']] for e in ('A', 'B')]
        if left not in primary or right not in primary:
            continue
        choices = [row[f'{p} - Choice'].strip().upper() for p in retained]
        if choices[0] not in ('A', 'B') or choices[0] != choices[1]:
            continue
        a, b = [[numerical(row[f'{p} - Score {e}']) for p in retained] for e in ('A', 'B')]
        if any(v is None for v in a + b):
            continue
        if (abs(statistics.mean(a) - statistics.mean(b)) > .5
                and (statistics.pstdev(a) + statistics.pstdev(b)) / 2 < 1.5):
            pairs.append([index, left, right, choices[0]])
    return target, pairs


def mutate(rows, people, ids, test_records=None):
    """Negative control, preserving availability; never used to generate production targets."""
    lookup = {s: i for i, s in enumerate(ids)}
    result = copy.deepcopy(rows)
    for row in result:
        ends = {e: lookup[row[f'Molecule {e} SMILES']] for e in ('A', 'B')}
        for p in people:
            for e, index in ends.items():
                key = f'{p} - Score {e}'
                value = numerical(row[key])
                if value is not None and (test_records is None or index in test_records):
                    row[key] = str(6 - value)
            if test_records is None or set(ends.values()) & test_records:
                key = f'{p} - Choice'
                value = row[key].strip().upper()
                if value in ('A', 'B'):
                    row[key] = 'B' if value == 'A' else 'A'
    return result


def training_payload(target, pairs, train):
    members = set(map(int, train))
    return dict(target=[[i, target[i]] for i in sorted(members)],
                pairs=[p for p in pairs if p[1] in members and p[2] in members])
