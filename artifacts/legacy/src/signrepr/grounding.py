"""Standard subsequence-DTW and one-to-one interval evaluation; no novelty."""
import numpy as np


def interval_overlap(a, b):
    intersection = max(0., min(a[1], b[1]) - max(a[0], b[0]))
    union = a[1]-a[0]+b[1]-b[0]-intersection
    if union <= 0:
        raise ValueError('Invalid interval duration')
    return intersection / union


def nms(candidates, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError('Invalid NMS IoU threshold')
    ordered = sorted(candidates, key=lambda r: (-r['score'], r['start_sec'], r['end_sec']))
    retained = []
    for candidate in ordered:
        if all(interval_overlap([candidate['start_sec'], candidate['end_sec']],
                                [r['start_sec'], r['end_sec']]) <= threshold for r in retained):
            retained.append(candidate)
    return retained


def subsequence_dtw(query, target, starts, ends, *, maximum_cost, min_duration, max_duration, nms_iou):
    """Minimize summed cosine distance with free target prefix/suffix.

    All query tokens are aligned. Ties prefer diagonal, then vertical, then
    horizontal. Score is minus winning summed cost divided by its path length;
    this does not optimize average cost. Duration filtering follows alignment.
    Source intervals remain in original time even after missing-token filtering.
    """
    query, target = np.asarray(query, float), np.asarray(target, float)
    starts, ends = np.asarray(starts, float), np.asarray(ends, float)
    if query.ndim != 2 or target.ndim != 2 or not len(query) or not query.shape[1] or query.shape[1] != target.shape[1]:
        raise ValueError('Query/target feature dimensions differ or query is empty')
    if starts.shape != (len(target),) or ends.shape != starts.shape or not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError('Invalid target source clock')
    if (ends <= starts).any() or not (np.diff(starts) > 0).all() or not (np.diff(ends) > 0).all():
        raise ValueError('Target source intervals must be positive and chronological')
    if not np.isfinite(query).all() or not np.isfinite(target).all() or not (0 <= maximum_cost <= 2 and 0 < min_duration <= max_duration):
        raise ValueError('Invalid feature values or locked search policy')
    if not 0 <= nms_iou <= 1 or not np.isfinite([maximum_cost, min_duration, max_duration]).all():
        raise ValueError('Invalid finite search policy or NMS threshold')
    qnorm, tnorm = np.linalg.norm(query, axis=1), np.linalg.norm(target, axis=1)
    if (qnorm <= 1e-10).any() or (tnorm <= 1e-10).any():
        raise ValueError('Zero feature cannot be normalized')
    if not len(target):
        return []
    local = 1 - np.clip((query/qnorm[:, None]) @ (target/tnorm[:, None]).T, -1, 1)
    nq, nt = local.shape
    distance = np.full((nq+1, nt+1), np.inf); distance[0] = 0
    length = np.zeros((nq+1, nt+1), dtype=np.int32)
    predecessor = np.zeros((nq+1, nt+1), dtype=np.uint8)
    for i in range(1, nq+1):
        for j in range(1, nt+1):
            choices = [(i-1, j-1), (i-1, j), (i, j-1)]
            direction = int(np.argmin([distance[a, b] for a, b in choices]))
            a, b = choices[direction]
            distance[i, j] = local[i-1, j-1] + distance[a, b]
            length[i, j] = length[a, b] + 1
            predecessor[i, j] = direction
    candidates = []
    for end in range(1, nt+1):
        cost = distance[nq, end] / length[nq, end]
        if cost > maximum_cost:
            continue
        i, j, path = nq, end, []
        while i > 0:
            if j <= 0:
                raise ValueError('DTW path lost query coverage')
            path.append((i-1, j-1))
            direction = predecessor[i, j]
            i, j = [(i-1, j-1), (i-1, j), (i, j-1)][direction]
        start_idx = min(b for _, b in path)
        start, stop = float(starts[start_idx]), float(ends[end-1])
        if min_duration <= stop-start <= max_duration:
            candidates.append({'start_sec': start, 'end_sec': stop, 'score': -float(cost),
                               'mean_path_cost': float(cost), 'path_length': len(path),
                               'query_tokens_covered': len({a for a, _ in path})})
    return nms(candidates, nms_iou)


def interpolated_ap(matches, positives):
    if not positives:
        return None
    correct = np.cumsum(matches)
    precision = correct / np.arange(1, len(matches)+1)
    mrec = np.concatenate([[0], correct / positives, [1]])
    mpre = np.maximum.accumulate(np.concatenate([[0], precision, [0]])[::-1])[::-1]
    changed = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(((mrec[changed+1]-mrec[changed])*mpre[changed+1]).sum())


def evaluate(predictions, gold, target_durations, queries, iou_threshold):
    """All-point interpolated AP, greedy score-ranked one-to-one matching.

    Every query is evaluated on every exhaustively reviewed target, including
    true negatives. Unannotated targets must never be passed as negatives.
    """
    if not 0 < iou_threshold <= 1 or not queries or len(set(queries)) != len(queries) or not target_durations or any(not np.isfinite(v) or v <= 0 for v in target_durations.values()):
        raise ValueError('Invalid registered metric coverage/durations')
    seen, lookup = set(), {q: {} for q in queries}
    for event in gold:
        q, t = event['query_id'], event['target_id']
        if q not in lookup or t not in target_durations:
            raise ValueError('Gold falls outside locked query/target coverage')
        interval = (event['start_sec'], event['end_sec'])
        if not np.isfinite(interval).all() or not 0 <= interval[0] < interval[1] <= target_durations[t]:
            raise ValueError('Gold interval outside target support')
        key = (q, t, *interval)
        if key in seen:
            raise ValueError('Duplicate gold occurrence')
        seen.add(key); lookup[q].setdefault(t, []).append(interval)
    for prediction in predictions:
        if prediction['query_id'] not in lookup or prediction['target_id'] not in target_durations:
            raise ValueError('Prediction coverage differs from locked targets')
        values = [prediction['score'], prediction['start_sec'], prediction['end_sec']]
        if not np.isfinite(values).all() or not 0 <= values[1] < values[2] <= target_durations[prediction['target_id']]:
            raise ValueError('Prediction interval outside target support')
    results, pooled, boundary_errors = {}, [], []
    for q in queries:
        candidates = sorted([r for r in predictions if r['query_id'] == q],
                            key=lambda r: (-r['score'], r['target_id'], r['start_sec'], r['end_sec']))
        used, matches, errors = set(), [], []
        negative_targets = set(target_durations) - set(lookup[q])
        negative_fp = 0
        for r in candidates:
            events = lookup[q].get(r['target_id'], [])
            eligible = [(interval_overlap([r['start_sec'], r['end_sec']], event), i) for i, event in enumerate(events)
                        if (r['target_id'], i) not in used]
            best = max(eligible, default=(0, -1))
            correct = best[0] >= iou_threshold
            if correct:
                used.add((r['target_id'], best[1]))
                errors.append([abs(r['start_sec']-events[best[1]][0]), abs(r['end_sec']-events[best[1]][1])])
            elif r['target_id'] in negative_targets:
                negative_fp += 1
            matches.append(int(correct))
            pooled.append((r, int(correct)))
        positives = sum(map(len, lookup[q].values()))
        ap = interpolated_ap(matches, positives)
        tp = int(sum(matches)); fp = len(matches)-tp
        boundary_errors.extend(errors)
        results[q] = {'AP': ap, 'positives': positives, 'predictions': len(matches), 'true_positives': tp, 'false_positives': fp,
                      'recall': tp/positives if positives else None,
                      'mean_matched_onset_offset_error_sec': np.mean(errors, axis=0).tolist() if errors else None,
                      'false_positives_per_minute': fp/(sum(target_durations.values())/60),
                      'verified_negative_targets': len(negative_targets), 'negative_target_false_positives': negative_fp,
                      'negative_target_false_positives_per_minute': negative_fp/(sum(target_durations[t] for t in negative_targets)/60) if negative_targets else None}
    available = [v['AP'] for v in results.values() if v['AP'] is not None]
    pooled.sort(key=lambda item: (-item[0]['score'], item[0]['query_id'], item[0]['target_id'], item[0]['start_sec'], item[0]['end_sec']))
    return {'macro_AP': float(np.mean(available)) if available else None, 'queries_with_positive_gold': len(available),
            'pooled_AP': interpolated_ap([hit for _, hit in pooled], len(gold)),
            'pooled_false_positives_per_query_target_minute': sum(v['false_positives'] for v in results.values())/(len(queries)*sum(target_durations.values())/60),
            'mean_matched_onset_offset_error_sec': np.mean(boundary_errors, axis=0).tolist() if boundary_errors else None,
            'per_query': results, 'iou_threshold': iou_threshold, 'AP_definition': 'All-point interpolated precision envelope; score-ranked greedy one-to-one occurrence matching.',
            'limitations': ['Point metrics only; recording-group bootstrap remains required for real dataset inference.',
                            'Queries with zero positive gold retain negative FP metrics but have undefined AP.']}


def recording_bootstrap(predictions, gold, targets, queries, iou_threshold, resamples=2000, seed=42):
    """Resample whole recordings, retaining every target and verified negative."""
    if resamples < 2000:
        raise ValueError('At least 2000 recording draws required')
    if len({r['target_id'] for r in targets}) != len(targets):
        raise ValueError('Duplicate target identity in recording bootstrap')
    evaluate(predictions, gold, {r['target_id']: r['duration_sec'] for r in targets}, queries, iou_threshold)
    groups = sorted({r['recording_group'] for r in targets})
    if len(groups) < 2:
        raise ValueError('At least two recording groups required for descriptive CI')
    target_by_group = {g: [r for r in targets if r['recording_group'] == g] for g in groups}
    pred_by_target = {r['target_id']: [] for r in targets}
    gold_by_target = {r['target_id']: [] for r in targets}
    for r in predictions:
        pred_by_target[r['target_id']].append(r)
    for r in gold:
        gold_by_target[r['target_id']].append(r)
    values, unavailable, missing_query_draws = [], 0, 0
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        prediction_draw, gold_draw, durations = [], [], {}
        for slot, group_index in enumerate(rng.integers(len(groups), size=len(groups))):
            for target in target_by_group[groups[group_index]]:
                tid = target['target_id']; replica = f'{slot}:{tid}'
                durations[replica] = target['duration_sec']
                prediction_draw.extend({**r, 'target_id': replica} for r in pred_by_target[tid])
                gold_draw.extend({**r, 'target_id': replica} for r in gold_by_target[tid])
        result = evaluate(prediction_draw, gold_draw, durations, queries, iou_threshold)
        missing_query_draws += result['queries_with_positive_gold'] < len(queries)
        if result['macro_AP'] is None:
            unavailable += 1
        else:
            values.append(result['macro_AP'])
    return {'resamples': resamples, 'bootstrap_seed': seed, 'recording_groups': len(groups),
            'macro_AP_ci95_on_supported_draws': np.quantile(values, [.025, .975]).tolist() if values else None,
            'draws_without_any_positive_query': unavailable, 'draws_missing_positive_query_support': missing_query_draws,
            'scope': 'Conditional fixed predictions and observed recordings. Unsupported all-negative draws are counted and omitted from descriptiveAPinterval; not population/retraining uncertainty.',
            'limitations': ['All query/target pairs and verified negatives retained in each sampled recording.',
                           'AP support can change by draw; few recording groups limit interval coverage.']}
