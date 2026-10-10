"""Read-only corpus geometry, masks, activity-window and gradient audit."""
import argparse
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from signrep_baseline.cache import load_standardized, clip_targets, sha256, implementation_sha256
from signrep_baseline.losses import reconstruction_loss
from signrep_baseline.schema import PRIOR_SHAPES, BODY_SOURCE, BODY_DEST, RIGHT_TIPS, RIGHT_DEST, LEFT_TIPS, LEFT_DEST


def complete_windows(flags):
    return np.convolve(np.asarray(flags, dtype=np.int64), np.ones(16, dtype=np.int64), mode='valid') == 16


def check_close(actual, expected, mask):
    expected = np.where(mask, expected, 0)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.output.resolve()
    config = json.loads((run / 'run_config.json').read_text())
    exp = ROOT / 'experiments' / config['experiment']
    rows = [json.loads(line) for line in (run / 'corpus/pretrain.jsonl').read_text().splitlines() if line]
    torch.set_num_threads(2)
    valid_counts, total_counts, video_fraction_sums = {}, {}, {}
    joints = np.zeros(61, dtype=np.int64)
    frames = windows = reference_windows = 0
    side_counts = {side: dict(hand_detected_windows=0, anchor_visible_windows=0,
                             hand_and_anchor_windows=0, eligible_windows=0,
                             active_windows=0, inactive_windows=0, eligible_videos=0)
                   for side in ['left', 'right']}
    activity_video_fraction = 0.
    examples, first_clip = [], None
    for index, row in enumerate(rows):
        assert sha256(row['pose_cache']) == row['pose_cache_sha256']
        data, meta = load_standardized(row['pose_cache'], row['bone_statistics_sha256'], row['video_sha256'])
        assert meta['body_root_mode'] == 'canonical'
        assert all(np.isfinite(data[key]).all() for key in ['pose_3d', 'confidence', 'body_angles', 'right_angles', 'left_angles'])
        pose = data['pose_3d']; conf = data['confidence']; n = len(pose)
        observed = (conf > .5) & np.isfinite(pose).all(axis=-1)
        global_observed = observed.copy()
        global_observed[:, 19:40] &= observed[:, 2:3]
        global_observed[:, 40:61] &= observed[:, 5:6]
        safe = np.where(observed[..., None], pose, 0)
        targets, masks = clip_targets(data, np.arange(n))
        shapes = {**PRIOR_SHAPES, 'activity': (2,)}
        for key, shape in PRIOR_SHAPES.items():
            assert targets[key].shape == (n, int(np.prod(shape)))
            assert torch.isfinite(targets[key]).all()
            assert not targets[key][~masks[key]].any()
        assert targets['activity'].shape == masks['activity'].shape == (2,)
        check_close(targets['body_pose'].reshape(n, 61, 3).numpy(), safe,
                    global_observed[..., None])
        for side, start in [('right', 19), ('left', 40)]:
            expected = safe[:, start:start+21] - safe[:, start:start+1]
            mask = observed[:, start:start+21] & observed[:, start:start+1]
            check_close(targets[f'{side}_pose'].reshape(n, 21, 3).numpy(), expected, mask[..., None])
            assert not targets[f'{side}_pose'].reshape(n, 21, 3)[:, 0].any()
        for part, count in [('body', 22), ('right', 41), ('left', 41)]:
            key = f'full_{part}_angles'; a = data[f'{part}_angles']
            expected_mask = np.broadcast_to(data[f'{part}_detected'][:, None, None], (n, count, 2))
            assert np.array_equal(masks[key].reshape(n, count, 2).numpy(), expected_mask)
            check_close(targets[key].reshape(n, count, 2).numpy(),
                        np.stack([np.sin(a), np.cos(a)], -1), expected_mask)
        for key, src, dst in [('body_dist', BODY_SOURCE, BODY_DEST),
                              ('right_dist', RIGHT_TIPS, RIGHT_DEST), ('left_dist', LEFT_TIPS, LEFT_DEST)]:
            expected = safe[:, src, None, :] - safe[:, None, dst, :]
            obs = global_observed if key == 'body_dist' else observed
            mask = obs[:, src, None] & obs[:, None, dst]
            assert np.array_equal(masks[key].reshape(n, len(src), len(dst), 3).numpy(),
                                  np.broadcast_to(mask[..., None], expected.shape))
            check_close(targets[key].reshape(expected.shape).numpy(), expected, mask[..., None])
        starts = np.flatnonzero(complete_windows(data['body_detected']))
        assert len(starts)
        delta = np.zeros(n+1, dtype=np.int64)
        np.add.at(delta, starts, 1); np.add.at(delta, starts+16, -1)
        multiplicity = np.cumsum(delta[:-1])
        for key in PRIOR_SHAPES:
            count = int(masks[key].numpy().sum(axis=1) @ multiplicity)
            total = len(starts)*16*targets[key].shape[1]
            valid_counts[key] = valid_counts.get(key, 0) + count
            total_counts[key] = total_counts.get(key, 0) + total
            video_fraction_sums[key] = video_fraction_sums.get(key, 0.) + count/total
        ref = complete_windows(observed[:, [0, 3, 6, 7]].all(axis=1))[starts]
        reference_windows += int(ref.sum())
        eligible_here = 0
        for side, start, anchor, activity_index in [('left', 40, 5, 0), ('right', 19, 2, 1)]:
            hand = complete_windows(observed[:, start:start+21].all(axis=1))[starts]
            wrist = complete_windows(observed[:, anchor])[starts]
            eligible = hand & wrist & ref
            counts = side_counts[side]
            counts['hand_detected_windows'] += int(hand.sum())
            counts['anchor_visible_windows'] += int(wrist.sum())
            counts['hand_and_anchor_windows'] += int((hand & wrist).sum())
            counts['eligible_windows'] += int(eligible.sum())
            counts['eligible_videos'] += int(eligible.any())
            eligible_here += int(eligible.sum())
            for begin in starts[eligible]:
                clip = pose[begin:begin+16]; reference = clip[:, [0, 3, 6, 7], 1].mean(axis=1)
                below = bool((clip[:, start, 1] > reference).all())
                stationary = float(clip[:, start:start+21].astype(np.float64).std(axis=0, ddof=1).sum()) < .26
                value = int(not (below and stationary))
                t, m = clip_targets(data, np.arange(begin, begin+16))
                assert m['activity'][activity_index] and int(t['activity'][activity_index]) == value
                counts['active_windows' if value else 'inactive_windows'] += 1
                if len(examples) < 8:
                    examples.append({'video_id': row['video_id'], 'cache': row['pose_cache'],
                                     'start': int(begin), 'side': side, 'target': value})
        activity_video_fraction += eligible_here/(2*len(starts))
        if first_clip is None:
            first_clip = clip_targets(data, np.arange(starts[0], starts[0]+16))
        frames += n; windows += len(starts); joints += observed.sum(axis=0)
        if (index+1) % 500 == 0:
            print(json.dumps({'audited_videos': index+1, 'activity_eligible_windows_so_far': sum(c['eligible_windows'] for c in side_counts.values())}), flush=True)
    # Probe actual per-prior gradients through the released decoder architecture,
    # without using the backbone/GPU or changing the running model.
    from signrep_baseline.upstream_signrep.models.head_models.dict_head_v6_FINAL_thoughtful import ProjectionModel
    torch.manual_seed(config['seed'])
    decoder = ProjectionModel(latents_params={'hidden_dim':512,'out_dim':384},
                              projection_params={k:{'hidden_dim':None,'output_dim':int(np.prod(s))} for k,s in PRIOR_SHAPES.items()},
                              total_frames=16,in_dim=768)
    active = torch.nn.Sequential(torch.nn.Linear(768,128),torch.nn.GELU(),torch.nn.Linear(128,2))
    t, m = first_clip
    if examples:
        d,_ = load_standardized(examples[0]['cache'])
        t,m = clip_targets(d,np.arange(examples[0]['start'],examples[0]['start']+16))
    z = torch.randn(1,768,requires_grad=True)
    predictions = {**decoder(z), 'activity': active(z)}
    loss, terms = reconstruction_loss(predictions, {k:v.unsqueeze(0) for k,v in t.items()},
                                       {k:v.unsqueeze(0) for k,v in m.items()})
    gradients = {}
    for key, term in terms.items():
        gradient = torch.autograd.grad(term, predictions[key], retain_graph=True)[0]
        if m[key].any(): assert gradient.abs().sum() > 0
        else: assert not gradient.any()
        gradients[key] = {'valid_elements':int(m[key].sum()),'prediction_gradient_l1':float(gradient.abs().sum())}
    loss.backward(); assert z.grad is not None and torch.isfinite(z.grad).all()
    coverage = {k:valid_counts[k]/total_counts[k] for k in valid_counts}
    coverage['activity'] = sum(c['eligible_windows'] for c in side_counts.values())/(2*windows)
    balanced = {k:v/len(rows) for k,v in video_fraction_sums.items()}
    balanced['activity'] = activity_video_fraction/len(rows)
    report = {'status':'PASS_WITH_LIMITATIONS','checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'videos':len(rows),'decoded_frames':frames,'all_signer_present_16frame_windows':windows,
              'geometry_masks_and_cache_hashes':'PASS','implementation_sha256':implementation_sha256(),
              'coverage_all_windows_weighted':coverage,'expected_uniform_video_uniform_window_coverage':balanced,
              'reference_complete_windows':reference_windows,'reference_complete_window_fraction':reference_windows/windows,
              'joint_visibility_fraction':(joints/frames).tolist(),'activity_by_side':side_counts,
              'activity_examples':examples,'decoder_prediction_gradient_probe':gradients,
              'latent_gradient_finite':True,
              'limitations':['Corpus correctness and gradient probes do not establish exact equivalence with unpublished author training code.',
                             'Strict complete-hand/reference activity eligibility is an engineering choice; paper does not specify its missingness reduction.',
                             'No running training code, configuration, cache or checkpoint was modified.']}
    (exp/'pose_priors_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
