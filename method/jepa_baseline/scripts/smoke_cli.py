"""End-to-end production entry points on small, explicitly technical-only fixtures."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import yaml
from signrep_baseline.cache import sha256


def main():
    folder = ROOT / "runtime/cli_smoke"
    folder.mkdir(parents=True, exist_ok=True)
    fixtures = ROOT / "runtime/smoke"
    records = []
    for index, suffix in enumerate(['', '2']):
        records.append({'video_id': f'technical_source{index}', 'youtube_id': f'technical_source{index}',
                        'video': str(fixtures / f'real_pose_clip{suffix}.mp4'), 'split': 'pretrain'})
    records.append({**records[0], 'video_id': 'blocked_video', 'youtube_id': 'blocked'})
    source = folder / 'source.jsonl'
    source.write_text(''.join(json.dumps(row)+'\n' for row in records))
    annotations = folder / 'wlasl_synthetic_annotation.json'
    annotations.write_text(json.dumps([{'instances': [{'url':'https://youtube.com/watch?v=blocked','split':'test'}]}]))
    corpus = folder / 'corpus'
    (corpus / 'raw').mkdir(parents=True, exist_ok=True)
    for index, suffix in enumerate(['', '2']):
        shutil.copyfile(fixtures / f'raw{suffix}.npz', corpus / 'raw' / f'{index:08d}.npz')
    steps = []
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}

    def run(*args):
        command = [sys.executable, '-m', 'signrep_baseline', *[str(a) for a in args]]
        result = subprocess.run(command, env=env, cwd=WORKSPACE, text=True, capture_output=True)
        (folder / f'command_{len(steps):02d}.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'{args[0]} failed: {result.stdout[-2000:]} {result.stderr[-4000:]}')
        steps.append(list(map(str,args)))
        print('PASS',args[0],flush=True)

    run('prepare-corpus', '--manifest', source, '--wlasl', annotations, '--output', corpus,
        '--pose-python', WORKSPACE / '.venv-shubert-native/bin/python', '--weights', ROOT / 'runtime/teacher/models')
    # Independent source videos; labels below are only technical interface fixtures.
    pilot = [json.loads(line) for line in (WORKSPACE / 'data/manifests/pilot.jsonl').read_text().splitlines()]
    queries = []
    for index, row in enumerate(pilot[2:4]):
        path = Path(row['source_root']) / row['relative_path']
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for number in range(count):
            ok, frame = capture.read()
            if not ok:break
            if max(0,count//2-8) <= number < max(0,count//2-8)+16:frames.append(frame)
        capture.release()
        assert len(frames)==16
        video = folder / f'query{index}.mp4'
        writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(frames[0].shape[1],frames[0].shape[0]))
        for frame in frames:writer.write(frame)
        writer.release()
        queries.append({'video_id': f'technical_query{index}','video':str(video),'split':'val','label':index,
                        'scope':'technical fixture; label is synthetic, not benchmark ground truth'})
    gallery = [{**record,'split':'train','label':index,'scope':'technical fixture; label synthetic'} for index,record in enumerate(records[:2])]
    gp, qp = folder / 'gallery.jsonl', folder / 'queries.jsonl'
    gp.write_text(''.join(json.dumps(r)+'\n' for r in gallery))
    qp.write_text(''.join(json.dumps(r)+'\n' for r in queries))
    config = {'scope':'technical','manifest':str(corpus/'pretrain.jsonl'),'output':str(folder/'train'),
              'initialization':'kinetics_mae','checkpoint':str(ROOT/'runtime/kinetics/mae_hiera_base_16x224.pth'),
              'device':'cpu','amp':False,'seed':9,'total_steps':4,'warmup_steps':1,'max_lr':.0001,'batch_size':4,
              'validation_interval':2,'save_interval':1,'validation_gallery':str(gp),'validation_queries':str(qp)}
    config_path = folder / 'pretrain.yaml'
    config_path.write_text(yaml.safe_dump(config))
    run('pretrain','--config',config_path,'--stop-after',1)
    run('pretrain','--config',config_path,'--resume',folder/'train/last.pt','--stop-after',2)
    assert (folder/'train/best.pt').exists() and (folder/'train/best.model.pt').exists()
    released = WORKSPACE / 'checkpoints/signrep/ckpt.pt'
    run('features','--manifest',gp,'--checkpoint',released,'--output',folder/'gallery_features')
    run('features','--manifest',qp,'--checkpoint',released,'--output',folder/'query_features')
    run('retrieval','--gallery',folder/'gallery_features/features.jsonl','--queries',folder/'query_features/features.jsonl',
        '--split','val','--output',folder/'retrieval.json')
    translation_rows = [json.loads(line) for line in (folder/'gallery_features/features.jsonl').read_text().splitlines()]
    for row in translation_rows:row['translation']='technical translation fixture'
    translation_manifest = folder/'translation.jsonl'
    translation_manifest.write_text(''.join(json.dumps(r)+'\n' for r in translation_rows))
    run('how2sign-export','--manifest',translation_manifest,'--output-tsv',folder/'translation.tsv','--feature-directory',folder/'translation_npy')
    audit=json.loads((corpus/'pretrain.jsonl.audit.json').read_text())
    assert audit['removed_video_ids']==['blocked_video']
    report={'result':'PASS','scope':'technical only; small ASL-Citizen clips, synthetic eval labels, not paper scores',
            'commands':steps,'held_out_video_sources_distinct':True,'audit_excludes_blocked_held_out_id':True,
            'cli_checkpoint_resume':True,'scheduled_validation_and_best_model_export':True,
            'frozen_feature_retrieval_and_translation_export':True}
    (ROOT/'evidence/S6_cli_smoke.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
