"""Recount completed reference diagnostics and retain exact terminal bindings.

Never rereads native source videos or restarts/audits the live full-cohort job.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    records=Path('method/research_lab/records')
    names=['MSNATIVESCANTEST001','MSNATIVESCANTEST002','MSNATIVECOST001','MSNATIVECOST002']
    results={}
    for name in names:
        reg=json.loads((records/f'experiments/{name}_registration.json').read_text())
        root=Path(reg['output_dir']);c=json.loads(Path(reg['config']).read_text());s=json.loads((root/'status.json').read_text())
        assert sha(reg['config'])==reg['config_sha256']==s['config_sha256']
        assert s['run_id']==reg['run_id']
        files={str(p.relative_to(root)):p for p in root.rglob('*') if p.is_file()}
        caps={**c['artifact_file_caps'],'artifact_manifest.json':c['artifact_manifest_cap']}
        assert set(files)<=set(caps)
        assert all(p.stat().st_size<=caps[n] for n,p in files.items())
        size=sum(p.stat().st_size for p in files.values());assert size<=c['artifact_ceiling_bytes']
        for n,h in s['output_hashes'].items():assert sha(root/n)==h
        if s['status']=='completed':
            assert set(files)==set(caps)
            manifest=json.loads((root/'artifact_manifest.json').read_text())
            # Manifest structure is validated by its writer; additionally every
            # terminal status output is rehashed above, with actual file caps.
            assert isinstance(manifest,dict)
        else:
            assert name=='MSNATIVECOST001' and s['status']=='failed'
        results[name]=dict(run_id=s['run_id'],status=s['status'],pid=s['pid'],wall_seconds=s['wall_seconds'],
                           GPU_hours=s['GPU_hours'],actual_logical_bytes=size,config_sha256=s['config_sha256'],
                           status_sha256=sha(root/'status.json'),caps_passed=True)
    reg=json.loads((records/'experiments/MSNATIVECOST002_registration.json').read_text());root=Path(reg['output_dir'])
    c=json.loads((root/'config.json').read_text());rows=json.loads((root/'native_scan_records.json').read_text());summary=json.loads((root/'native_scan_summary.json').read_text())
    assert [x['sample_id'] for x in rows]==c['selected_sample_ids'] and len(set(c['selected_sample_ids']))==11
    assert all(x['status']=='eligible' and x['independent_reference']['integer_pts_exact'] for x in rows)
    assert len([x for x in rows if x['class_id']>=100])==4
    assert sum(x['native']['selected_native_frames'] for x in rows)==summary['selected_native_frames']==1155
    assert sum(x['native']['complete_stride2_windows'] for x in rows)==summary['complete_stride2_windows']
    assert abs(sum(x['wall_seconds'] for x in rows)-summary['captured_row_wall_seconds'])<1e-12
    source=json.loads((root/'source_parents.json').read_text())
    assert len(source)==summary['source_parents']==9
    assert sum(x['bytes'] for x in source.values())==summary['unique_source_hashed_bytes']==985888623
    old=Path(c['prior_native_runtime']);prior=json.loads((old/'native_audit.json').read_text())
    import numpy as np
    for x in prior['samples']:
        row=next(r for r in rows if r['sample_id']==x['sample_id'])
        arr=np.load(old/(x['sample_id']+'_first_native_window.npy'),allow_pickle=False)
        pts=np.load(old/(x['sample_id']+'_native_pts.npy'),allow_pickle=False)
        assert hashlib.sha256(arr.tobytes()).hexdigest()==row['native']['first_native16_RGB_sha256']
        assert len(pts)==row['native']['selected_native_frames']
        assert float(pts[0])==row['native']['first_selected_relative_pts']
        assert float(pts[-1])==row['native']['last_selected_relative_pts']
        assert row['old_native_reference_equal']
    report=dict(verified_utc=datetime.now(timezone.utc).isoformat(),passed=True,terminal_runs=results,
                actual_terminal_wall_seconds=sum(x['wall_seconds'] for x in results.values()),
                actual_terminal_logical_bytes=sum(x['actual_logical_bytes'] for x in results.values()),GPU_hours=0,
                reference_rows=11,reference_controls=4,target_rows=7,integer_pts_frames=1155,
                scope='Same-agent independent arithmetic/hash/cap/native-reference consistency; no new native decode/scientific confirmation. Failed individual rows missing retained explicitly. Full3425 native audit is a separate live job, not covered by these checks.',
                verifier_source_sha256=sha(__file__))
    with (records/'inventory/MSNATIVESCAN_BINDING_VERIFICATION.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ['terminal_runs','scope']},indent=2))


if __name__=='__main__':main()
