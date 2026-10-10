"""Execute the torchrun CLI boundary on a synthetic CPU cohort."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from signcoord.config import load_config
from signcoord.engine.checkpoint import load_checkpoint
from signcoord.engine.train import make_dataset
from signcoord.provenance import atomic_json,execution_stamp,sha256_file
from signcoord.targets.statistics import fit_target_statistics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(PROJECT/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    fixture=root/'fixture'
    with (root/'fixture.log').open('w') as log:
        subprocess.run([sys.executable,str(PROJECT/'tools/generate_synthetic_fixture.py'),'--output',str(fixture)],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    config=load_config(fixture/'fixture_smoke.yaml')
    config['run_dir']=str(root/'train')
    config['train'].update(microbatch_per_gpu=2,accumulation_steps=2)
    fit_target_statistics(make_dataset(config),config['data']['ssl_train_manifest'],
                          config['data']['target_stats_path'],config['data']['phase_calibration_path'])
    path=root/'config.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
             '--module','signcoord.cli','train','--config',str(path)]
    atomic_json(root/'command.json',command)
    with (root/'torchrun.log').open('w') as log:
        result=subprocess.run(command,env=env,cwd=PROJECT.parents[1],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('torchrun failed; inspect torchrun.log')
    checkpoint=root/'train/checkpoints/final.pt';saved=load_checkpoint(checkpoint)
    assert saved['world_size']==2 and saved['optimizer_update']==8
    assert saved['compute_ledger']['stats_batch']==4 and saved['compute_ledger']['optimizer_batch']==8
    assert len(saved['rng_states_by_rank'])==2
    assert all(state['cuda'] is None and state['cuda_device_rng'] is None for state in saved['rng_states_by_rank'])
    records=[json.loads(line) for line in (root/'train/metrics.jsonl').read_text().splitlines()]
    assert len(records)==8
    assert any(m['relation_weight']>0 and m['relation']['active'] for record in records
               for rank in record['rank_microsteps'] for m in rank)
    report={**execution_stamp(),'status':'passed','data_kind':'synthetic','world_size':2,'backend':'gloo',
            'optimizer_updates':8,'stats_batch':4,'optimizer_batch':8,'accumulation_steps':2,
            'checkpoint_sha256':sha256_file(checkpoint),'checkpoint':str(checkpoint),
            'config_sha256':sha256_file(path),'command':command,'compute_ledger':saved['compute_ledger'],
            'relation_active_supervision':True,'cpu_checkpoint_cuda_rng_queries':False,
            'scope':'actual launcher/CLI/optimizer engineering check; no scientific transfer result'}
    atomic_json(root/'ddp_cli_smoke_report.json',report)
    atomic_json(PROJECT/'docs/evidence/ddp_cli_smoke_report.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
