"""One-example class galleries from a prevalidated native-eligible cohort."""
import numpy as np


def sample_galleries(eligible_rows, class_ids, seeds=(1701,1702,1703)):
    classes = list(class_ids)
    if (not classes or any(type(x) is not int or x < 0 for x in classes)
            or classes != sorted(set(classes))):
        raise ValueError('Vocabulary must be ascending unique nonnegative integers')
    if not seeds or any(type(x) is not int or x < 0 for x in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Fixed gallery seeds must be unique nonnegative integers')
    if not eligible_rows or len({r['sample_id'] for r in eligible_rows}) != len(eligible_rows):
        raise ValueError('Cohort sample IDs must be unique and nonempty')
    if any(not isinstance(r['sample_id'],str) or not r['sample_id'] or r['status']!='eligible'
           or r['split'] not in ('train','val') or type(r['class_id']) is not int or r['class_id'] not in classes
           or type(r['signer_id']) is not int or r['signer_id'] < 0
           or not isinstance(r['parent_id'],str) or not r['parent_id'] for r in eligible_rows):
        raise ValueError('Native eligibility/split/label/signer/source contract failed; no reselection')
    train = [r for r in eligible_rows if r['split']=='train']
    queries = [r for r in eligible_rows if r['split']=='val']
    if (set(r['signer_id'] for r in train)&set(r['signer_id'] for r in queries)
            or set(r['parent_id'] for r in train)&set(r['parent_id'] for r in queries)):
        raise ValueError('Train/query signer or source-parent overlap')
    if set(r['class_id'] for r in train)!=set(classes) or set(r['class_id'] for r in queries)!=set(classes):
        raise ValueError('All declared classes need native-eligible train and query support')
    pools = {c:sorted(r['sample_id'] for r in train if r['class_id']==c) for c in classes}
    supports = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        supports[str(seed)] = [pools[c][int(rng.integers(0,len(pools[c])))] for c in classes]
    return dict(class_ids=classes, support_seeds=list(seeds), supports=supports,
                query_ids=sorted(r['sample_id'] for r in queries),
                native_train_counts={str(c):len(pools[c]) for c in classes},
                sampling_rule='Per seed PCG64 default_rng; ascending classes; sample uniform integer from sample-ID-sorted train pool. One example/class; no query access or reselection. Reuse SAME saved supports for all methods.',
                numpy_version=np.__version__)
