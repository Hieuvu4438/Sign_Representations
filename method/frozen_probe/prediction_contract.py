"""Post-execution artifact integrity without computing any ground-truth score."""
import numpy as np

from confirmatory import METHODS, RECIPES, query_map
from temporal import window_plan
from window_budget import selected_indices


def verify_source_recipes(cohort, test, audit, schema, observations):
    indexed = query_map(cohort, test)
    ids = sorted(indexed)
    if (schema['split'] != 'test' or schema['query_ids'] != ids
            or schema['class_ids'] != list(range(len(cohort['classes'])))
            or schema['labels'] != [indexed[k]['class_id'] for k in ids]
            or schema['participant_ids'] != [indexed[k]['participant_id'] for k in ids]):
        raise ValueError('Prediction schema differs from held-out source identities/labels')
    videos = {s['sample_id']: s for s in audit['samples']}
    if len(videos) != len(audit['samples']) or set(videos) != set(ids) or not audit['passed']:
        raise ValueError('Native source audit incomplete')
    for key, source in indexed.items():
        native = videos[key]
        if any(native[field] != source[field] for field in ('split', 'source_sha256', 'participant_id', 'class_id')):
            raise ValueError('Audited source identity differs')
        plan = window_plan(native['pts_s'], 16, 2)
        if len(plan) != native['windows'] or len(native['pts_s']) != native['frames']:
            raise ValueError('Source native frame/window plan differs')
    seen = set()
    counts = {method: 0 for method in RECIPES}
    for row in observations:
        key, method = row['sample_id'], row['method']
        if key not in indexed or method not in RECIPES or (key, method) in seen:
            raise ValueError('Timing source/recipe coverage differs')
        seen.add((key, method))
        selected = selected_indices(method, videos[key]['windows'], indexed[key]['source_sha256'])
        if (row['native_window_indices'] != selected or row['encoded_windows'] != len(selected)
                or row['source_windows'] != videos[key]['windows']):
            raise ValueError('Actual encoder calls differ from source-only registered recipe')
        counts[method] += len(selected)
    if seen != {(key, method) for key in ids for method in RECIPES}:
        raise ValueError('Missing actual held-out query/recipe execution')
    return counts


def verify_pooled_arrays(arrays, expected_rows, dimension):
    if set(arrays) != set(METHODS):
        raise ValueError('Pooled method coverage differs')
    for matrix in arrays.values():
        matrix = np.asarray(matrix)
        if (matrix.dtype != np.float64 or matrix.shape != (expected_rows, dimension)
                or not np.isfinite(matrix).all()
                or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-10, rtol=1e-10)):
            raise ValueError('Pooled features violate dimension/dtype/finite/L2 contract')
