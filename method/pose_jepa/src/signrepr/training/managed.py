"""Signal-safe single-GPU training entrypoint used by the durable job supervisor."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import time
import torch
from ..config import load_config
from ..data.pose_dataset import PoseDataset
from ..data.schema import sha256_file
from ..evaluation.loso import validate_loso
from ..evaluation.dictionary import validate_dictionary
from .distributed import initialize_distributed, world_size
from .pretrain import SSLTrainer
from .provenance import provenance
from .stream import SampleStream
from .checkpoints import save_training
from .job_support import atomic_json, durable_checkpoint, resume_latest


def run(config_path, job_root, device='cuda', cpu_threads=4):
    device = initialize_distributed(device)
    if world_size() != 1: raise ValueError('The managed job is configured for one GPU')
    torch.set_num_threads(cpu_threads)
    config = load_config(config_path)
    if device.type == 'cuda' and config['optim'].get('cuda_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(config['optim']['cuda_memory_fraction'], device)
    root = Path(config['run']['output_dir']); root.mkdir(parents=True, exist_ok=True)
    job_root = Path(job_root)
    selection = config['data']['validation_manifest']
    metadata = provenance(config, config['data']['train_manifest'])
    metadata['selection_protocol_sha256'] = sha256_file(selection)
    validation_protocol = config['evaluation'].get('selection_protocol', 'loso')
    if validation_protocol not in ('loso', 'dictionary'): raise ValueError('Unknown selection protocol')
    metadata['selection_protocol'] = ('raw_dictionary_official_train_gallery_val_queries' if validation_protocol == 'dictionary'
                                      else 'raw_loso_on_official_val_excluding_aslc300_queries')
    if validation_protocol == 'dictionary':
        from ..data.schema import read_manifest
        import hashlib
        inventory = []
        for row in read_manifest(selection):
            digest = sha256_file(row['pose_path'])
            if digest != row['pose_sha256']: raise ValueError('Validation pose checksum mismatch')
            inventory.append((row['sample_id'], digest))
        metadata['selection_pose_inventory_sha256'] = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
    dataset = PoseDataset(config['data']['train_manifest'], config['data'], sampling='isolated')
    trainer = SSLTrainer(config, device)
    stream = SampleStream(dataset, config['optim']['per_rank_batch'], config['run']['seed']+1,
                          num_workers=config['data']['num_workers'], pin_memory=device.type == 'cuda',
                          persistent_workers=config['data']['persistent_workers'])
    metadata, recovery = resume_latest(trainer, stream, root, metadata)
    best = metadata.get('best_validation_metric')
    stale = metadata.get('stale_validations', 0)
    elapsed = metadata.get('active_training_seconds', 0.)
    selection_count = metadata.get('validation_count', 0)
    if recovery: atomic_json(root/'resume.json', recovery)
    config_copy = root/'config.yaml'
    if not config_copy.exists(): config_copy.write_text(Path(config_path).read_text())
    atomic_json(root/'provenance.json', metadata)
    stop = {'requested': False, 'signal': None}
    def request_stop(number, frame):
        stop.update(requested=True, signal=number)
    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    attempts = 0
    last_saved = trainer.step
    status = 'running'
    run_started = time.time()
    def progress(extra=None):
        record = {'phase': 'train', 'status': status, 'pid': os.getpid(), 'step': trainer.step,
                  'max_steps': config['optim']['max_steps'], 'last_checkpoint_step': last_saved,
                  'best_validation_R1': best, 'stale_validations': stale,
                  'active_training_seconds': elapsed, 'updated_unix': time.time(),
                  'run_started_unix': run_started, 'recovery': recovery, **(extra or {})}
        atomic_json(job_root/'progress.json', record)
        text = (f"{config['run']['name']} | {status}\nStep: {trainer.step}/{config['optim']['max_steps']}\n"
                f"Checkpoint: {last_saved}\nBest validation R@1: {best}\n"
                f"Active training: {elapsed/3600:.2f} hours\nUpdated: {time.ctime()}\n")
        (job_root/'progress.txt').write_text(text)
    def validation():
        nonlocal best, stale, selection_count, last_saved
        from .checkpoints import rng_state, restore_rng
        rng = rng_state(); progress({'activity': 'validation'})
        try:
            if validation_protocol == 'dictionary':
                result = validate_dictionary(trainer.teacher, config['data']['train_manifest'], selection, config,
                    progress_callback=lambda split,done,total: progress({'activity':'validation', 'validation_split':split,
                                                                        'validation_completed':done, 'validation_total':total}),
                    should_stop=lambda: stop['requested'] or (job_root/'pause.request').exists())
            else:
                result = validate_loso(trainer.teacher, selection, config)
        except InterruptedError:
            stop['requested'] = True
            return
        finally:
            restore_rng(rng)
        score = result['metrics']['R@1']; selection_count += 1
        atomic_json(root/f'validation_{trainer.step:08d}.json', result)
        improved = best is None or score > best
        if improved: best, stale = score, 0
        else: stale += 1
        metadata.update(best_validation_metric=best, stale_validations=stale, validation_count=selection_count,
                        last_validation_step=trainer.step)
        if improved: save_training(trainer, root/'best_validation.pt', stream.state_dict(), metadata)
        durable_checkpoint(trainer, stream, root, metadata); last_saved = trainer.step
        progress({'activity': 'training'})
    try:
        if not recovery:
            durable_checkpoint(trainer, stream, root, metadata)
        progress()
        if (root/'complete.json').exists():
            complete = json.loads((root/'complete.json').read_text())
            if complete['checkpoint_sha256'] != sha256_file(root/'last.pt'):
                raise ValueError('Completed training checkpoint changed')
            return complete
        interval = config['evaluation']['frozen_validation_interval_steps']
        # A crash during validation leaves a complete pre-validation checkpoint.
        # Repeat that validation before consuming a new training batch, including
        # the final update where the loop below would otherwise never execute.
        pending_validation = trainer.step > metadata.get('last_validation_step', 0) and (trainer.step % interval == 0 or trainer.step == config['optim']['max_steps'])
        if pending_validation and not stop['requested'] and not (job_root/'pause.request').exists(): validation()
        while trainer.step < config['optim']['max_steps']:
            if stop['requested'] or (job_root/'pause.request').exists(): break
            if stale >= config['evaluation'].get('early_stopping_patience', 5):
                status = 'early_stopped'; break
            started = time.monotonic()
            batches = [stream.next() for _ in range(config['optim']['gradient_accumulation_steps'])]
            report = trainer.update(batches)
            elapsed += time.monotonic()-started
            attempts += 1
            metadata['active_training_seconds'] = elapsed
            with (root/'train.jsonl').open('a') as log:
                log.write(json.dumps({**report, 'attempt_unix': time.time()})+'\n')
            if report['successful'] and (trainer.step == 1 or trainer.step % config['evaluation']['checkpoint_interval_steps'] == 0):
                durable_checkpoint(trainer, stream, root, metadata); last_saved = trainer.step
            progress({'last_update': report})
            if stop['requested'] or (job_root/'pause.request').exists(): break
            if report['successful'] and (trainer.step % interval == 0 or trainer.step == config['optim']['max_steps']):
                validation()
                if stale >= config['evaluation'].get('early_stopping_patience', 5):
                    status = 'early_stopped'; break
            if attempts > max(100, config['optim']['max_steps']*10):
                raise RuntimeError('Repeated unsupported/nonfinite updates; refusing an endless restart')
        durable_checkpoint(trainer, stream, root, metadata); last_saved = trainer.step
        if stop['requested'] or (job_root/'pause.request').exists():
            status = 'paused'; progress({'stop_signal': stop['signal']})
            return {'status': status, 'step': trainer.step}
        if status != 'early_stopped': status = 'complete'
        result = {'status': status, 'step': trainer.step, 'best_validation_R1': best,
                  'checkpoint_sha256': sha256_file(root/'last.pt'), 'finished_unix': time.time()}
        atomic_json(root/'complete.json', result); progress()
        return result
    finally:
        stream.close()
        for number, handler in old_handlers.items(): signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--job-root', required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.job_root, args.device, args.cpu_threads)), flush=True)


if __name__ == '__main__': main()
