"""Rational, ordinal-preserving dual-clock diagnostics; never repair timestamps."""
from fractions import Fraction
import re


def clock_comparison(raw, best, timebase, origin_tick, left, right):
    if len(raw) != len(best) or not raw:
        raise ValueError('Clock arrays require the same nonempty decoded-frame ordinals')
    if any(type(t) is not int for t in raw + best):
        raise ValueError('Only observed integer ticks are accepted')
    tb, lo, hi = Fraction(timebase), Fraction(str(left)), Fraction(str(right))
    if tb <= 0 or not lo < hi:
        raise ValueError('Invalid timebase/annotation interval')
    def selected(ticks):
        return [i for i, t in enumerate(ticks) if lo <= (t - origin_tick) * tb < hi]
    def bad(ticks):
        return [i for i in range(1, len(ticks)) if ticks[i] <= ticks[i-1]]
    a, b = selected(raw), selected(best)
    sa, sb = set(a), set(b)
    return dict(raw_selected_indices=a, best_selected_indices=b,
                raw_nonincreasing_indices=bad(raw), best_nonincreasing_indices=bad(best),
                changed_tick_indices=[i for i, (x, y) in enumerate(zip(raw, best)) if x != y],
                added_by_best_indices=[i for i in b if i not in sa],
                removed_by_best_indices=[i for i in a if i not in sb],
                maximum_absolute_clock_delta_seconds=float(max(abs(x-y) for x,y in zip(raw,best))*tb),
                selected_raw_nonincreasing_indices=bad([raw[i] for i in a]),
                selected_best_nonincreasing_indices=bad([best[i] for i in b]),
                selected_indices_identical=a == b,
                raw_complete_native16_stride2_windows=max(0, (len(a)-16)//2+1),
                best_complete_native16_stride2_windows=max(0, (len(b)-16)//2+1))


def parse_framehash(text):
    timebase = None
    rows = []
    for line in text.splitlines():
        if line.startswith('#tb 0:'):
            if timebase is not None:
                raise ValueError('Duplicate stream timebase')
            timebase = Fraction(line.split(':', 1)[1].strip())
        elif line and not line.startswith('#'):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) != 6 or parts[0] != '0' or not re.fullmatch('[0-9a-fA-F]{64}', parts[5]):
                raise ValueError('Malformed SHA256 framehash row or extra stream')
            rows.append(dict(dts=int(parts[1]), pts=int(parts[2]), duration=int(parts[3]),
                             bytes=int(parts[4]), sha256=parts[5].lower()))
    if timebase is None or timebase <= 0 or not rows or any(r['bytes'] <= 0 for r in rows):
        raise ValueError('Incomplete framehash')
    return dict(time_base=str(timebase), frames=rows)
