"""Publish shared-full-source prediction and source-group result tables."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np

from experiments.export_official_result_details import SCOPES, calculate, load_predictions


SEEDS = (20262020, 20262021, 20262022)
DATASETS = ('syncg', 'rf100', 'industrial')
FIELDS = ('dataset', 'seed', 'sample_id', 'group_id', 'condition', 'normalized_target',
          'normalized_prediction', 'normalized_absolute_error', 'status', 'relation_available',
          'candidate_base', 'candidate_polar', 'candidate_relational',
          'weight_base', 'weight_polar', 'weight_relational', 'original_geoprr_error', 'raw_full_error')
METRICS = ('nmae_pct_fs', 'acc_at_2_pct', 'acc_at_5_pct', 'coverage_pct', 'failures')
GROUP_FIELDS = ('dataset', 'method', 'seed', 'aggregation', 'scope', 'group_id', 'images', 'rows', *METRICS)
PAIR_FIELDS = ('dataset', 'scope', 'reference', 'group_id', 'rows_per_seed', 'seed_count',
               'shared_full_nmae_pct_fs', 'reference_nmae_pct_fs', 'candidate_minus_reference_pct_fs')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def write_compressed(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=6, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)


def export(root, destination, previous):
    root, destination = root.resolve(), destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    summary = read_json(root/'full_three_seed_summary.json')
    assert summary['status'] == 'complete' and summary['completed_seeds'] == list(SEEDS)
    for name in ('full_three_seed_summary.json', 'full_three_seed_per_seed_metrics.csv',
                 'full_three_seed_mean_sample_sd.csv', 'full_three_seed_paired_comparisons.csv'):
        (destination/name).write_text((root/name).read_text(encoding='utf-8-sig'), encoding='utf-8', newline='\n')
    (destination/'RESULTS.md').write_text((root/'full_three_seed_summary.md').read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    groups_out, pairs_out, counts, failures = [], [], {}, 0
    for dataset in DATASETS:
        ledgers, arrays = {}, {}
        for seed in SEEDS:
            path = root/f'full_training_seed_{seed}/evaluation/{dataset}/predictions.jsonl'
            ledgers[seed] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            arrays[seed] = load_predictions(path)
        first = arrays[SEEDS[0]]
        group_names, sample_names = sorted(set(first['group_id'])), sorted(set(first['sample_id']))
        group_alias = {name: f'{dataset}_group_{index:03d}' for index, name in enumerate(group_names, 1)}
        sample_alias = {name: f'industrial_{index:06d}' if dataset == 'industrial' else str(name)
                        for index, name in enumerate(sample_names, 1)}
        old = {}
        with gzip.open(previous/f'per_sample/{dataset}_main.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['method'] == 'geoprr' and int(row['seed']) == SEEDS[0]:
                    old[row['sample_id'], row['condition']] = row['group_id'], float(row['normalized_target'])
        public, cohort_groups = [], {}
        for seed in SEEDS:
            data = arrays[seed]
            for field in ('sample_id', 'group_id', 'condition', 'target'):
                assert np.array_equal(first[field], data[field]), (dataset, seed, field)
            expected_error = np.where(data['success'], np.abs(data['prediction']-data['target']), 1.)
            assert np.allclose(expected_error, data['normalized_absolute_error'], rtol=0, atol=1e-12)
            assert len(old) == len(ledgers[seed])
            failures += int((~data['success']).sum())
            for row in ledgers[seed]:
                sample, group = sample_alias[row['sample_id']], group_alias[row['group_id']]
                assert old[sample, row['condition']] == (group, row['normalized_target'])
                candidate = row['candidate_predictions']
                weights = row['routing_weights']
                assert len(candidate) == len(weights) == 3
                public.append({'dataset': dataset, 'seed': seed, 'sample_id': sample, 'group_id': group,
                    'condition': row['condition'], 'normalized_target': row['normalized_target'],
                    'normalized_prediction': row['normalized_prediction'] if row['status'] == 'success' else '',
                    'normalized_absolute_error': row['normalized_absolute_error'], 'status': row['status'],
                    'relation_available': row['relation_available'],
                    **dict(zip(('candidate_base', 'candidate_polar', 'candidate_relational'), candidate)),
                    **dict(zip(('weight_base', 'weight_polar', 'weight_relational'), weights)),
                    'original_geoprr_error': row['original_geoprr_error'], 'raw_full_error': row['raw_full_error']})
            for scope, conditions in SCOPES.items():
                selected = np.isin(data['condition'], conditions)
                metric = calculate(data, selected)
                expected = summary['per_seed'][str(seed)][dataset]['shared_full'][scope]
                for field in ('rows', 'images', *METRICS):
                    assert abs(metric[field]-expected[field]) < 1e-9, (dataset, seed, scope, field)
                for name in group_names:
                    group = group_alias[name]
                    metric = calculate(data, selected & (data['group_id'] == name))
                    output = dict(dataset=dataset, method='shared_full', seed=seed, aggregation='single_seed',
                                  scope=scope, group_id=group, **metric)
                    groups_out.append(output)
                    cohort_groups.setdefault((scope, group), []).append(output)
        for rows in cohort_groups.values():
            assert len(rows) == 3
            output = {**rows[0], 'seed': '', 'aggregation': 'mean_of_three_seed_metrics'}
            for field in METRICS:
                output[field] = float(np.mean([r[field] for r in rows]))
            groups_out.append(output)
        candidate = np.mean([arrays[s]['normalized_absolute_error'] for s in SEEDS], axis=0)
        references = {name: np.mean([[r[field] for r in ledgers[s]] for s in SEEDS], axis=0)
                      for name, field in [('original_geoprr', 'original_geoprr_error'), ('raw_full', 'raw_full_error')]}
        for scope, conditions in SCOPES.items():
            selected = np.isin(first['condition'], conditions)
            for reference, errors in references.items():
                if scope in ('all_conditions', 'clean'):
                    expected = summary['paired_comparisons'][dataset][scope][reference]['candidate_minus_reference_pct_fs']
                    assert abs(float((candidate[selected]-errors[selected]).mean()*100)-expected) < 1e-9
                for name in group_names:
                    chosen = selected & (first['group_id'] == name)
                    pairs_out.append(dict(dataset=dataset, scope=scope, reference=reference,
                        group_id=group_alias[name], rows_per_seed=int(chosen.sum()), seed_count=3,
                        shared_full_nmae_pct_fs=float(candidate[chosen].mean()*100),
                        reference_nmae_pct_fs=float(errors[chosen].mean()*100),
                        candidate_minus_reference_pct_fs=float((candidate[chosen]-errors[chosen]).mean()*100)))
        write_compressed(destination/f'per_sample/{dataset}.csv.gz', FIELDS, public)
        counts[dataset] = dict(images=len(sample_names), groups=len(group_names), prediction_rows=len(public))
    write_compressed(destination/'group_metrics.csv.gz', GROUP_FIELDS, groups_out)
    write_compressed(destination/'group_comparisons.csv.gz', PAIR_FIELDS, pairs_out)
    protocol = read_json(root/'full_training_seed_20262020/protocol.json')
    published = dict(experiment=protocol['experiment'], seeds=list(SEEDS),
        foundation={'architecture': 'EfficientNet-B0 scalar regressor', 'training_images': 16000,
                    'epochs': 60, 'selection': 'same-seed terminal checkpoint', 'reused': True},
        downstream_training={'remst': 16000, 'r2mt': 16000, 'polar': 16000, 'final_router': 16000},
        final_router_image_condition_rows=96000, warm_initialization='same saved A-trained initializer for all three seeds',
        warm_training_images=12818, budgets=protocol['budgets'], variants=['full'],
        final_checkpoint='fixed-epoch parameter EMA', evaluation_precision='float32',
        source_test_access_during_training=False, target_adaptation=False,
        training_overlap='foundation, candidates and final router share the full source-training pool',
        conditions=list(SCOPES['all_conditions']), scopes={name: list(value) for name, value in SCOPES.items()},
        cohorts=counts, original_comparator_package='official_syncg_fulltrain_20260908/details',
        group_aliases_match_original_comparator_package=True)
    write_json(destination/'protocol.json', published)
    validation = dict(status='complete', seeds=list(SEEDS), prediction_rows=sum(x['prediction_rows'] for x in counts.values()),
        failures=failures, group_metric_rows=len(groups_out), group_comparison_rows=len(pairs_out),
        checked_metric_scopes=81, per_row_errors_recomputed=True, summary_metrics_match=True,
        sample_target_group_alignment=True, aliases_match_previous_public_package=True)
    write_json(destination/'export_validation.json', validation)
    print(json.dumps(validation), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--previous-details', type=Path, required=True)
    args = parser.parse_args()
    export(args.run_root, args.output_dir, args.previous_details)
