"""Bounded numerical integration of semantic scoring, without pair gold.

Reuses one real source-captioned training clip and cached C04 cue streams.
Repeated identical text tests order/tie determinism. Blank-stream scoring tests
API/numerical behavior only; never reports semantic accuracy or absence gold.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

from _common import ROOT
from run_semantic_pairs import STREAM_SHAPES, load_model, score
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.semantic_pairs import matched_credit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        raise ValueError('Preserve attempts; use new project-local output')
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    state = {'status': 'RUNNING_NUMERICAL_ONLY', 'pid': os.getpid(), 'device': 'cpu'}
    write_json(args.output / 'status.json', state)
    try:
        import torch
        torch.set_num_threads(2)
        prior_path = ROOT / 'features/shubert_translation_smoke_attempt1/report.json'
        prior = json.loads(prior_path.read_text())
        if prior['status'] != 'PASS_LOCAL_PUBLIC_DEMO_NUMERICAL_SMOKE':
            raise ValueError('Successful real-clip numerical source required')
        row = next(r for _, r in read_jsonl(ROOT / 'data/external/ncslgr/diagnostic_v1/manifest.jsonl')
                   if r['utterance_sample_id'] == prior['sample_id'] and r['view_role'] == 'body')
        video = Path(row['source_root']) / row['relative_path']
        if sha256(video) != prior['video_sha256']:
            raise ValueError('Real source clip changed')
        cache_path = prior_path.parent / 'demo_streams.npz'
        with np.load(cache_path, allow_pickle=False) as data:
            frames = prior['sampled_frames']
            if not np.array_equal(data['sampled_indices'], prior['sampled_frame_indices']) or frames > 200:
                raise ValueError('Bounded C04 source frame identity differs')
            streams = {k: data[k].copy() for k in STREAM_SHAPES}
            if any(v.shape != (frames, STREAM_SHAPES[k]) or not np.isfinite(v).all() for k, v in streams.items()):
                raise ValueError('Cached C04 stream shapes/values differ')
        config = yaml.safe_load((ROOT / 'configs/experiments/A01.yaml').read_text())
        model, tokenizer, audit = load_model(config)
        # This direct backend test supplies one source caption twice. It does
        # not bypass the external runner's readiness gate for any pair dataset.
        repeated = score(model, tokenizer, streams, [prior['source_caption']] * 2, 256)
        for value in repeated:
            if value['token_count'] != prior['token_count'] or not np.isclose(value['mean_nll'], prior['mean_nll'], atol=1e-5, rtol=0):
                raise ValueError('Prior real-clip token/NLL result not reproduced')
        if repeated[0] != repeated[1] or matched_credit([r['mean_nll'] for r in repeated], 0) != .5:
            raise ValueError('Identical candidate determinism/tie accounting differs')
        blank = score(model, tokenizer, {k: np.zeros_like(v) for k, v in streams.items()}, [prior['source_caption']], 256)[0]
        result = {**state, 'status': 'PASS_SEMANTIC_BACKEND_NUMERICAL_ONLY', 'elapsed_seconds': time.perf_counter()-started,
                  'prior_report_sha256': sha256(prior_path), 'cached_stream_snapshot_sha256': sha256(cache_path),
                  'video_sha256': sha256(video), 'sample_id': prior['sample_id'], 'sampled_frames': frames,
                  'source_caption': prior['source_caption'], 'repeated_source_caption_scores': repeated,
                  'blank_stream_source_caption_score': blank, 'identical_text_tie_credit': .5, 'load_audit': audit,
                  'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT / 'scripts/run_semantic_pairs.py',
                                     ROOT / 'src/signrepr/semantic_pairs.py', ROOT / 'src/signrepr/scoring.py']},
                  'limits': ['One training-signer numerical input, no distinct semantic candidates or external pair gold.',
                             'Blank streams test numerical validity only; not absence or causal visual-reliance evidence.',
                             'Prior smoke did not pin cue-cache checksum; current snapshot hash plus source/frame/shape/NLL reproduction establish numerical linkage, not earlier byte identity.',
                             'External manifest/gate/cue pipeline integration still requires reviewed gold/media.'], 'semantic_accuracy_measured': False}
        write_json(args.output / 'report.json', result); state.update(status=result['status'], elapsed_seconds=result['elapsed_seconds'])
        write_json(args.output / 'status.json', state)
        print(json.dumps({k: result[k] for k in ['status', 'sampled_frames', 'elapsed_seconds', 'repeated_source_caption_scores', 'blank_stream_source_caption_score', 'semantic_accuracy_measured']}))
    except Exception as error:
        state.update(status='FAIL', failure_reason=type(error).__name__+': '+str(error), elapsed_seconds=time.perf_counter()-started)
        write_json(args.output / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
