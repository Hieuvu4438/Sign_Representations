"""Prepare concrete local requests; never fabricate or send expert annotations."""
import csv
import json
from pathlib import Path

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json, write_jsonl


def main():
    output = ROOT / 'reports/ncslgr_expert_review_v1'
    if output.exists():
        raise ValueError('Preserve existing reviewer material and responses')
    output.mkdir()
    audit = ROOT / 'reports/ncslgr_interval_audit.json'
    selected = json.loads(audit.read_text())['fixed_error_ids']
    targets = {r['sample_id']: r for _, r in read_jsonl(ROOT / 'data/external/ncslgr/intervals_v1/targets.jsonl')}
    manifest = ROOT / 'data/external/ncslgr/diagnostic_v1/manifest.jsonl'
    media = {r['sample_id']: r for _, r in read_jsonl(manifest)}
    if len(selected) != 30 or len(set(selected)) != 30:
        raise ValueError('Use immutable30case selection, not a newly selected error subset')
    packets, template, requests = [], [], []
    for sid in selected:
        target = targets[sid]
        views = {}
        for role in ['body', 'face']:
            row = media[sid+':'+role]
            path = Path(row['source_root']) / row['relative_path']
            metadata = json.loads(Path(row['metadata_source']).read_text())
            if sha256(path) != metadata['prepared_sha256'] or not metadata['roi_lossless_verified']:
                raise ValueError('Prepared review video provenance differs')
            views[role] = {'path': str(path), 'sha256': metadata['prepared_sha256'],
                           'frames': metadata['frames'], 'fps': metadata['fps'], 'camera_id': row['source_camera_id']}
        if views['body']['frames'] != views['face']['frames'] or views['body']['fps'] != views['face']['fps']:
            raise ValueError('Paired review source clocks differ')
        duration = views['body']['frames']/views['body']['fps']
        if target['split'] != 'test' or not target['scope_supervised']:
            raise ValueError('Fixed selected review case is not source-positive single-interval test')
        packets.append({'canonical_id': sid, 'views': views, 'duration_sec': duration,
            'utterance_support_sec': target['utterance_support_sec'], 'recording_group': target['recording_group'],
            'source_signer': target['signer'], 'source_event_id': target['event_id'],
            'source_reference_only': {'label': target['label'], 'functional_core_interval_sec': target['recorded_functional_interval_sec']},
            'review_status': 'PENDING_ASL_EXPERT', 'sampling': 'Immutable original interval audit fixed error IDs, SHA25642selection; post-test diagnostic, not representative population sample.',
            'instruction': 'Researcher provenance packet includes source reference. Keep source labels, model outputs and metrics hidden from first-pass reviewers; use separate blank template and videos.'})
        template.append({'canonical_id': sid, 'body_video_path': views['body']['path'], 'face_video_path': views['face']['path'],
            'duration_sec': duration, 'fps': views['body']['fps'], 'reviewer_id': '', 'review_status': 'PENDING_ASL_EXPERT',
            'video_identity_verified': '', 'function_labels': '', 'marker_intervals_json': '',
            'anatomical_phases_json': '', 'grammatical_scope_intervals_json': '', 'absence_coverage_reviewed': '',
            'uncertain_or_unannotated_intervals_json': '', 'manual_semantic_match_review': '', 'notes': ''})
        requests.append({'canonical_id': sid, 'duration_sec': duration, 'target_phenomenon': target['label']+' (source-recorded reference)',
            'requested_label_or_scope': 'Independent function review; marker/anatomical phases/full grammatical scope separately; exhaustive absence only if explicitly reviewed',
            'source_license': 'ASLLRP source research/education terms; no media redistribution or external upload',
            'ambiguity': 'Source functional core is not independently adjudicated full grammatical scope; test-error selection is not representative; unknown is not absence',
            'reviewer_requirement': 'ASL-qualified reviewer; independent dual review of fixed first10cases and adjudication of disagreements',
            'required_artifact': 'reports/ncslgr_expert_review_v1/adjudicated.jsonl plus reviewer identities, source hashes and interval/label agreement',
            'status': 'PENDING_ASL_EXPERT'})
    write_jsonl(output / 'source_packets.jsonl', packets)
    for name, rows in [('review_template.csv', template), ('annotation_request.csv', requests)]:
        with (output / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output / 'manifest.json', {'status': 'REVIEW_MATERIAL_PREPARED_EXPERT_REVIEW_NOT_COMPLETED',
        'case_count': 30, 'expert_reviews_completed': 0, 'selected_ids': selected,
        'fixed_dual_review_ids': selected[:10], 'input_manifest_sha256': sha256(manifest),
        'input_audit_sha256': sha256(audit), 'script_sha256': sha256(Path(__file__)),
        'packet_sha256': sha256(output / 'source_packets.jsonl'), 'template_sha256': sha256(output / 'review_template.csv'),
        'source_clock': 'Paired lossless source RGB ROI, original30FPS frame-start clocks; half-open interval seconds/frame indices.',
        'limits': ['No expert responses synthesized or sent.', 'Selected held-out error cases cannot become adaptation train/validation data.',
                   'Dual-review subset is a proposed request, not completed agreement evidence. Expert must validate annotation guideline.',
                   'Full independent semantic/scope gold remains unavailable. This packet supports error review only.']})
    print(json.dumps({'status': 'REVIEW_MATERIAL_PREPARED_EXPERT_REVIEW_NOT_COMPLETED', 'cases': 30}))


if __name__ == '__main__':
    main()
