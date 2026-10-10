"""ASL-LEX 2.0 phonology joins for evaluation-only diagnostics (never used for training).

Minimal pairs: two glosses whose ASL-LEX codes differ in exactly one parameter group among
PARAMS and agree on all others (plus SignType), using initial-morpheme 2.0 codings.
Minimal-pair discrimination for a query of class A and pair (A,B): correct iff the
class-max score of A > that of B. This compares two specific lexical neighbours and so does
not suffer from probe leakage through sign identity.
"""
from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

ASLLEX = Path(__file__).resolve().parents[1] / "runtime" / "external" / "asllex" / "signdata.csv"
PARAMS = {
    "handshape": ["Handshape.2.0"],
    "location": ["MajorLocation.2.0", "MinorLocation.2.0"],
    "movement": ["Movement.2.0", "RepeatedMovement.2.0"],
}
CONTEXT = ["SignType.2.0"]


def load_asllex(path=ASLLEX):
    out = {}
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            out[r["Code"]] = r
    return out


def gloss_codes(clips):
    """gloss -> ASL-LEX code from clip rows (codes 'NA' dropped)."""
    m = {}
    for c in clips:
        if c.get("asllex") and c["asllex"] != "NA":
            m.setdefault(c["gloss"], c["asllex"])
    return m


def signature(row):
    sig = {k: tuple(row[c] for c in cols) for k, cols in PARAMS.items()}
    sig["context"] = tuple(row[c] for c in CONTEXT)
    return sig


def minimal_pairs(glosses, gcode, lex):
    """Return {param: [(gi, gj), ...]} over gloss indices (unordered pairs)."""
    sigs = {}
    for i, g in enumerate(glosses):
        code = gcode.get(g)
        if code in lex:
            row = lex[code]
            # cleaner pairs: single-morpheme, non-compound, non-fingerspelled, non-initialized signs
            if (row.get("NumberOfMorphemes.2.0") != "1" or row.get("Compound.2.0") == "1"
                    or row.get("FingerspelledLoanSign.2.0") == "1" or row.get("Initialized.2.0") == "1"):
                continue
            s = signature(row)
            if any(v in ("", "NA") for t in s.values() for v in t):
                continue
            sigs[i] = s
    pairs = defaultdict(list)
    # bucket by all-but-one parameter to avoid O(n^2) over 2.7k glosses
    keys = list(PARAMS)
    for p in keys:
        bucket = defaultdict(list)
        for i, s in sigs.items():
            bucket[tuple(s[k] for k in keys if k != p) + (s["context"],)].append(i)
        for members in bucket.values():
            for a, b in itertools.combinations(members, 2):
                if sigs[a][p] != sigs[b][p]:
                    pairs[p].append((a, b))
    return dict(pairs), sigs


def minimal_pair_accuracy(scores, q_labels, pairs):
    """scores [Nq,C] class scores; for every query of class a in a pair (a,b) (both directions)."""
    q_labels = np.asarray(q_labels)
    by_class = defaultdict(list)
    for qi, c in enumerate(q_labels):
        by_class[c].append(qi)
    res = {}
    for p, plist in pairs.items():
        hits, n, per_pair = 0, 0, []
        for a, b in plist:
            for x, y in ((a, b), (b, a)):
                qs = by_class.get(x, [])
                if not qs:
                    continue
                h = (scores[qs, x] > scores[qs, y]).astype(float)
                hits += h.sum()
                n += len(qs)
                per_pair.append(h.mean())
        res[p] = {"query_acc": hits / max(n, 1), "pair_macro_acc": float(np.mean(per_pair)) if per_pair else float("nan"),
                  "n_pairs": len(plist), "n_trials": n}
    return res


def error_attribute_sharing(scores, q_labels, sigs):
    """Among rank-1 errors, fraction where predicted class shares each parameter with truth,
    vs. the same fraction for a uniformly random wrong class (chance)."""
    pred = scores.argmax(1)
    keys = list(PARAMS)
    cls = np.array(sorted(sigs))
    share, chance, n = {k: 0 for k in keys}, {k: 0.0 for k in keys}, 0
    # chance per true class: fraction of other coded classes sharing parameter
    cache = {}
    for qi, (t, p) in enumerate(zip(q_labels, pred)):
        if t == p or t not in sigs or p not in sigs:
            continue
        n += 1
        if t not in cache:
            cache[t] = {k: np.mean([sigs[c][k] == sigs[t][k] for c in cls if c != t]) for k in keys}
        for k in keys:
            share[k] += sigs[p][k] == sigs[t][k]
            chance[k] += cache[t][k]
    return {k: {"error_share": share[k] / max(n, 1), "chance_share": chance[k] / max(n, 1)} for k in keys} | {"n_errors": n}
