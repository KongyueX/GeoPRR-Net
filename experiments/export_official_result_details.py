"""Export official prediction ledgers and source-group statistics for publication."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np


CONDITIONS = ('clean', 'blur_moderate', 'blur_severe', 'perspective_moderate',
              'perspective_severe', 'combined_severe')
SCOPES = {'all_conditions': CONDITIONS, **{c: (c,) for c in CONDITIONS},
          'perspective_pair': CONDITIONS[3:5], 'projective_three': CONDITIONS[3:]}
DATASETS = {'SyncG': 'syncg', 'RF100-VL': 'rf100', 'Industrial-1395': 'industrial'}
IDENTITY = ('family', 'method', 'dataset', 'seed', 'configuration', 'angle_degrees')
ROW_FIELDS = (*IDENTITY, 'sample_id', 'group_id', 'condition', 'normalized_target',
              'normalized_prediction', 'normalized_absolute_error', 'status', 'outer_fold')
METRICS = ('nmae_pct_fs', 'acc_at_2_pct', 'acc_at_5_pct', 'coverage_pct', 'failures')
GROUP_FIELDS = (*IDENTITY, 'aggregation', 'scope', 'group_id', 'images', 'rows', *METRICS)
PAIR_FIELDS = ('comparison', 'dataset', 'scope', 'group_id', 'rows_per_seed',
               'seed_count', 'candidate_nmae_pct_fs', 'reference_nmae_pct_fs',
               'candidate_minus_reference_pct_fs')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_arrays(path):
    with np.load(path) as data:
        return {key: data[key] for key in ('sample_id', 'group_id', 'condition', 'target',
                                         'prediction', 'normalized_absolute_error', 'success', 'outer_fold')}


def load_predictions(path):
    columns = defaultdict(list)
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = json.loads(line)
            for key in ('sample_id', 'group_id', 'condition'):
                columns[key].append(row[key])
            success = row['status'] == 'success'
            columns['target'].append(row['normalized_target'])
            columns['prediction'].append(row['normalized_prediction'] if success else np.nan)
            columns['normalized_absolute_error'].append(row['normalized_absolute_error'])
            columns['success'].append(success)
            columns['outer_fold'].append(row.get('outer_fold') if row.get('outer_fold') is not None else -1)
    return {key: np.asarray(value) for key, value in columns.items()}


def calculate(data, selected):
    errors = data['normalized_absolute_error'][selected]
    success = data['success'][selected]
    return dict(rows=int(selected.sum()), images=len(set(data['sample_id'][selected])),
                nmae_pct_fs=float(errors.mean() * 100),
                acc_at_2_pct=float(np.mean(success & (errors <= .02)) * 100),
                acc_at_5_pct=float(np.mean(success & (errors <= .05)) * 100),
                coverage_pct=float(success.mean() * 100), failures=int((~success).sum()))


def export_scan_group_comparisons(root, output):
    index = read_json(root / 'statistics/index.json')
    units = {(u['model'], u['seed'], u['angle_degrees']): u for u in index['perspective_scan_index']}
    records = read_json(root / 'perspective_scan/summary.json')['paired_group_comparisons']
    relative = 'scan_group_comparisons.csv.gz'
    rows = []
    for record in records:
        angle, reference = record['angle_degrees'], record['reference']
        candidate = [load_arrays(units['geoprr', seed, angle]['arrays']) for seed in (20262020, 20262021, 20262022)]
        other = [load_arrays(units[reference, seed, angle]['arrays']) for seed in (20262020, 20262021, 20262022)]
        for item in candidate[1:] + other:
            for field in ('sample_id', 'group_id', 'target'):
                if not np.array_equal(item[field], candidate[0][field]):
                    raise ValueError('scan comparison row alignment differs')
        left = np.mean([a['normalized_absolute_error'] for a in candidate], axis=0)
        right = np.mean([a['normalized_absolute_error'] for a in other], axis=0)
        if abs((left.mean() - right.mean()) * 100 - record['statistics']['candidate_minus_reference_pct_fs']) > 1e-9:
            raise ValueError('scan paired effect differs from the published summary')
        for i, group in enumerate(sorted(set(candidate[0]['group_id'])), 1):
            selected = candidate[0]['group_id'] == group
            rows.append(dict(comparison=f'geoprr_minus_{reference}', dataset='syncg', scope=f'angle_{angle}',
                             group_id=f'syncg_group_{i:03d}', rows_per_seed=int(selected.sum()), seed_count=3,
                             candidate_nmae_pct_fs=float(left[selected].mean() * 100),
                             reference_nmae_pct_fs=float(right[selected].mean() * 100),
                             candidate_minus_reference_pct_fs=float((left[selected].mean() - right[selected].mean()) * 100)))
    with (output / relative).open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=6, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDS, lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
    return {'file': relative, 'rows': len(rows), 'columns': list(PAIR_FIELDS), 'bytes': (output / relative).stat().st_size}


def export(run_root, output):
    root, output = run_root.resolve(), output.resolve()
    index = read_json(root / 'statistics/index.json')
    references = {(r['family'], r['method'], r['dataset'], str(r['seed']), r['scope']): r
                  for r in read_csv(root / 'statistics/metrics.csv')}
    group_alias, sample_alias = {}, {}
    populations = {}
    for dataset in ('syncg', 'rf100', 'industrial'):
        unit = next(u for u in index['prediction_units'] if u['family'] == 'main'
                    and u['method'] == 'geoprr' and u['dataset'] == dataset)
        data = load_arrays(unit['arrays'])
        groups, samples = sorted(set(data['group_id'])), sorted(set(data['sample_id']))
        group_alias[dataset] = {name: f'{dataset}_group_{i:03d}' for i, name in enumerate(groups, 1)}
        sample_alias[dataset] = {name: f'industrial_{i:06d}' if dataset == 'industrial' else str(name)
                                 for i, name in enumerate(samples, 1)}
        populations[dataset] = {(str(s), str(c)): (str(g), float(t))
                                for s, c, g, t in zip(data['sample_id'], data['condition'], data['group_id'], data['target'])}

    output.mkdir(parents=True, exist_ok=True)
    unit_manifest, files, writers, group_means = [], {}, {}, defaultdict(list)
    checked_scopes = 0
    with ExitStack() as stack:
        def writer(relative, fields):
            if relative not in writers:
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = stack.enter_context(path.open('wb'))
                zipped = stack.enter_context(gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=6, mtime=0))
                stream = stack.enter_context(io.TextIOWrapper(zipped, encoding='utf-8', newline=''))
                value = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
                value.writeheader()
                writers[relative] = value
                files[relative] = {'file': relative, 'rows': 0, 'columns': list(fields)}
            return writers[relative]

        def emit(relative, fields, row):
            writer(relative, fields).writerow(row)
            files[relative]['rows'] += 1

        def export_unit(meta, data, expected, source, duplicate_of=None):
            nonlocal checked_scopes
            dataset = meta['dataset']
            count = len(data['target'])
            keys = list(zip(data['sample_id'], data['condition']))
            if len(set(keys)) != count:
                raise ValueError(f'duplicate sample-condition keys: {source}')
            is_scan = meta['family'] == 'perspective_scan'
            for i, (sample, condition) in enumerate(keys):
                expected_group, expected_target = populations[dataset][sample, 'clean' if is_scan else condition]
                if data['group_id'][i] != expected_group or data['target'][i] != expected_target:
                    raise ValueError(f'population or target mismatch: {source}')
            wanted = len(populations[dataset]) // (6 if is_scan else 1)
            if count != wanted:
                raise ValueError(f'incomplete population: {source}: {count} != {wanted}')
            errors = np.where(data['success'], np.abs(data['prediction'] - data['target']), 1.)
            if not np.allclose(errors, data['normalized_absolute_error'], rtol=0, atol=1e-12):
                raise ValueError(f'prediction/error mismatch: {source}')
            if not np.isfinite(data['prediction'][data['success']]).all():
                raise ValueError(f'nonfinite successful prediction: {source}')
            scopes = {'all_conditions': tuple(set(data['condition']))} if is_scan else SCOPES
            for scope, conditions in scopes.items():
                selected = np.isin(data['condition'], conditions)
                actual = calculate(data, selected)
                ref = expected[scope]
                for field in ('rows', *METRICS):
                    if abs(actual[field] - float(ref[field])) > 1e-9:
                        raise ValueError(f'aggregate mismatch: {source}/{scope}/{field}: {actual[field]} != {ref[field]}')
                checked_scopes += 1
                for group in sorted(set(data['group_id'][selected])):
                    values = calculate(data, selected & (data['group_id'] == group))
                    row = {**meta, 'aggregation': 'ensemble' if str(meta['seed']) == 'ensemble' else 'single_seed',
                           'scope': scope, 'group_id': group_alias[dataset][group], **values}
                    emit('group_metrics.csv.gz', GROUP_FIELDS, row)
                    if meta['seed'] != 'ensemble':
                        key = tuple(meta[k] for k in IDENTITY if k != 'seed') + (scope, row['group_id'])
                        group_means[key].append(row)
            relative = f'per_sample/{dataset}_{meta["family"]}.csv.gz'
            for i in range(count):
                success = bool(data['success'][i])
                emit(relative, ROW_FIELDS, {**meta,
                     'sample_id': sample_alias[dataset][data['sample_id'][i]],
                     'group_id': group_alias[dataset][data['group_id'][i]],
                     'condition': str(data['condition'][i]),
                     'normalized_target': float(data['target'][i]),
                     'normalized_prediction': float(data['prediction'][i]) if success else '',
                     'normalized_absolute_error': float(errors[i]),
                     'status': 'success' if success else 'failure',
                     'outer_fold': int(data['outer_fold'][i]) if int(data['outer_fold'][i]) >= 0 else ''})
            unit_manifest.append({**meta, 'rows': count, 'file': relative,
                                  'source_artifact': str(Path(source).relative_to(root)).replace('\\', '/'),
                                  'duplicate_of': duplicate_of})
            if len(unit_manifest) % 24 == 0:
                print(json.dumps({'exported_units': len(unit_manifest), 'prediction_rows': sum(u['rows'] for u in unit_manifest)}), flush=True)

        for unit in index['prediction_units']:
            meta = {k: unit[k] for k in ('family', 'method', 'dataset', 'seed')}
            meta.update(configuration='', angle_degrees='')
            expected = {s: references[(unit['family'], unit['method'], unit['dataset'], str(unit['seed']), s)] for s in SCOPES}
            export_unit(meta, load_arrays(unit['arrays']), expected, unit['predictions'],
                        'main: same frozen predictions with outer-fold alignment' if unit['family'] == 'frozen_fold_aligned' else None)

        sensitivity = read_json(root / 'sensitivity/summary.json')
        for unit in sensitivity['units']:
            for config, record in unit['configurations'].items():
                meta = dict(family='sensitivity', method='geoprr', dataset=DATASETS[unit['dataset']],
                            seed=unit['seed'], configuration=config, angle_degrees='')
                export_unit(meta, load_predictions(record['predictions']), record['metrics'], record['predictions'],
                            'main/geoprr: default configuration replay' if config == sensitivity['main_configuration'] else None)

        scan = read_json(root / 'perspective_scan/summary.json')
        scan_refs = {(u['model'], u['seed'], u['angle_degrees']): u['metrics'] for u in scan['per_seed']}
        for unit in index['perspective_scan_index']:
            meta = dict(family='perspective_scan', method=unit['model'], dataset='syncg', seed=unit['seed'],
                        configuration='', angle_degrees=unit['angle_degrees'])
            export_unit(meta, load_arrays(unit['arrays']), {'all_conditions': scan_refs[unit['model'], unit['seed'], unit['angle_degrees']]}, unit['predictions'])

        for name, expected in read_json(root / 'statistics/ensemble_metrics.json').items():
            family, method = name.split('/')
            directory = root / 'statistics/ensembles' / method / family
            meta = dict(family=f'ensemble_{family}', method=method, dataset='industrial', seed='ensemble',
                        configuration='equal_three_source_seeds', angle_degrees='')
            export_unit(meta, load_arrays(directory / 'rows.npz'), expected, directory / 'predictions.jsonl')

        for rows in group_means.values():
            if len(rows) != 3 or {r['seed'] for r in rows} != {20262020, 20262021, 20262022}:
                raise ValueError('group metric requires the complete three-seed roster')
            row = {**rows[0], 'seed': '', 'aggregation': 'mean_of_three_seed_metrics'}
            for key in METRICS:
                row[key] = float(np.mean([r[key] for r in rows]))
            if any(r['rows'] != row['rows'] or r['images'] != row['images'] for r in rows):
                raise ValueError('group denominator differs across seeds')
            emit('group_metrics.csv.gz', GROUP_FIELDS, row)

        paired = read_json(root / 'statistics/paired_comparisons.json')
        for record in paired:
            with np.load(record['arrays']) as a, np.load(record['shared_draws']) as d:
                candidate, reference = a['candidate_group_error_sum'], a['reference_group_error_sum']
                groups, counts = d['group_order'], d['group_row_counts']
                for field, value in [('candidate_nmae_pct_fs', candidate.sum() / counts.sum() * 100),
                                     ('reference_nmae_pct_fs', reference.sum() / counts.sum() * 100)]:
                    if abs(value - record[field]) > 1e-9:
                        raise ValueError('paired group totals differ from the published comparison')
                for i, group in enumerate(groups):
                    emit('group_comparisons.csv.gz', PAIR_FIELDS,
                         {'comparison': record['comparison'], 'dataset': record['dataset'], 'scope': record['scope'],
                          'group_id': group_alias[record['dataset']][group], 'rows_per_seed': int(counts[i]),
                          'seed_count': len(a['seed_order']),
                          'candidate_nmae_pct_fs': float(candidate[i] / counts[i] * 100),
                          'reference_nmae_pct_fs': float(reference[i] / counts[i] * 100),
                          'candidate_minus_reference_pct_fs': float((candidate[i] - reference[i]) / counts[i] * 100)})

    for name, record in files.items():
        record['bytes'] = (output / name).stat().st_size
    scan_file = export_scan_group_comparisons(root, output)
    files[scan_file['file']] = scan_file
    manifest = dict(run=root.name, prediction_units=len(unit_manifest),
                    prediction_rows=sum(u['rows'] for u in unit_manifest),
                    verified_metric_scopes=checked_scopes, paired_comparisons=len(paired), scan_paired_comparisons=12,
                    cohort_sizes={d: {'samples': len(sample_alias[d]), 'groups': len(group_alias[d])} for d in populations},
                    files=list(files.values()), units=unit_manifest,
                    excluded=['source images', 'model weights', 'local absolute paths',
                              'Industrial source identities and physical readings', 'runtime caches'])
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: manifest[k] for k in ('prediction_units', 'prediction_rows', 'verified_metric_scopes', 'paired_comparisons')}, ensure_ascii=False), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    export(args.run_root, args.output_dir)
