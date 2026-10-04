"""Native input coverage diagnostic, independent of labels/features/test outcomes."""
import math


def interval_coverage(intervals, lower, upper):
    if (not intervals or not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower
            or any(not math.isfinite(a) or not math.isfinite(b) or not lower <= a < b <= upper for a, b in intervals)):
        raise ValueError('Intervals outside the dense native domain')
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    covered = sum(end-start for start, end in merged)
    gaps = [merged[0][0]-lower, upper-merged[-1][1]]
    gaps.extend(b[0]-a[1] for a, b in zip(merged, merged[1:]))
    return dict(covered=covered, fraction=covered/(upper-lower), longest_uncovered_gap=max(gaps),
                disjoint_covered_intervals=len(merged), leading_uncovered=merged[0][0]-lower,
                trailing_uncovered=upper-merged[-1][1])


def native_coverage(plan, indices):
    if (not plan or not indices or indices != sorted(set(indices))
            or any(type(i) is not int or not 0 <= i < len(plan) for i in indices)):
        raise ValueError('Invalid selected native plan indices')
    frame_intervals = [(plan[i]['start_frame'], plan[i]['stop_frame_exclusive']) for i in indices]
    time_intervals = [(plan[i]['start_s'], plan[i]['end_s_exclusive']) for i in indices]
    frame = interval_coverage(frame_intervals, plan[0]['start_frame'], plan[-1]['stop_frame_exclusive'])
    time = interval_coverage(time_intervals, plan[0]['start_s'], plan[-1]['end_s_exclusive'])
    return dict(dense_windows=len(plan), encoded_windows=len(indices), frame=frame, time=time,
                any_selected_end_boundary_estimated=any(plan[i]['end_boundary_estimated'] for i in indices))
