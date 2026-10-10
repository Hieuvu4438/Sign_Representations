"""Real-source 100-step head cost and numerical gate, no heldout quality."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_gpu_context import BoundedGPUContext
from bounded_artifacts import verify_store
from limited_gpu_resources import configure_gpu_2gib
from resources import scratch_preflight, gpu_preflight
from rejection_heads import make_heads, episode_inputs, balanced_logit_loss
from paired_rejection_episodes import paired_inputs


def read(path): return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    for name in ('config', 'run-id', 'output-dir'): p.add_argument('--' + name, required=True)
    p.add_argument('--seed', type=int, required=True); a = p.parse_args()
    with BoundedGPUContext(a.output_dir, a.run_id, a.config, a.seed) as run:
        c = run.config; resource = scratch_preflight('/dev/shm', 512 << 20)
        root = Path(c['pool_runtime']); verify_store(root); assert read(root / 'pool_summary.json')['passed']
        vectors = np.load(root / 'fit_vectors.npy', allow_pickle=False)
        episodes = np.load(root / 'training_plan.npy', allow_pickle=False)
        labels = np.array([r['class_id'] for r in read(root / 'source_roster.json')])
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0, 'method/stream_reliability'); from gpu_accounting import inspect_effective_budget
        budget = inspect_effective_budget(c)
        observation = gpu_preflight(c['min_free_mib'], c['max_gpu_utilization'])
        assert observation['uuid'] == c['granted_uuid']
        cpu_heads = make_heads(a.seed)
        run.start_gpu(budget); device = configure_gpu_2gib(observation)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        heads = [copy.deepcopy(head).to(device) for head in cpu_heads]
        inputs = paired_inputs if c.get('paired_episodes') else episode_inputs
        residual, similarity, targets = inputs(vectors, labels, episodes[0])
        parities = []
        for cpu, gpu in zip(cpu_heads, heads):
            left = cpu(torch.from_numpy(residual)); right = gpu(torch.from_numpy(residual).to(device))
            assert torch.allclose(left, right.detach().cpu(), atol=1e-4, rtol=1e-4)
            loss_cpu = balanced_logit_loss(left, torch.from_numpy(similarity), torch.from_numpy(targets))
            loss_gpu = balanced_logit_loss(right, torch.from_numpy(similarity).to(device), torch.from_numpy(targets).to(device))
            assert loss_cpu is not None and loss_gpu is not None
            loss_cpu.backward(); loss_gpu.backward()
            maximum = 0.
            for x, y in zip(cpu.parameters(), gpu.parameters()):
                maximum = max(maximum, float((x.grad - y.grad.cpu()).abs().max()))
                assert torch.allclose(x.grad, y.grad.cpu(), atol=1e-4, rtol=1e-4)
            parities.append(dict(maximum_forward_error=float((left - right.detach().cpu()).abs().max()),
                                 maximum_gradient_error=maximum, loss_cpu=float(loss_cpu.detach()), loss_gpu=float(loss_gpu.detach())))
        optimizers = [torch.optim.Adam(head.parameters(), lr=.001, weight_decay=.0001) for head in heads]
        losses = []; updates = 0
        torch.cuda.synchronize(); start = time.monotonic()
        for episode in episodes[:100]:
            residual, similarity, targets = inputs(vectors, labels, episode)
            x = torch.from_numpy(residual).to(device)
            sim = torch.from_numpy(similarity).to(device); target = torch.from_numpy(targets).to(device)
            row = []
            for head, optimizer in zip(heads, optimizers):
                optimizer.zero_grad(set_to_none=True)
                loss = balanced_logit_loss(head(x), sim, target)
                if loss is None: row.append(0.); continue
                loss.backward(); optimizer.step(); row.append(float(loss.detach().cpu()))
            losses.append(row)
            if loss is not None: updates += 1
        torch.cuda.synchronize(); seconds = time.monotonic() - start
        estimate = 10 * seconds + 30  # full training plus prospective eval/context allowance
        memory = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        assert memory['peak_reserved_bytes'] <= 2 << 30
        run.save_array('cost_losses.npy', np.asarray(losses, dtype=np.float64), c['array_specs']['cost_losses.npy'])
        run.save('head_cost.json', dict(passed=estimate <= 120, source_episode_attempts=100, source_updates=updates,
            source_pipeline_seconds=seconds, estimated_full_training_eval_seconds=estimate,
            CPU_GPU_parities=parities, memory=memory, resource=resource, device_observation=observation,
            torch_version=torch.__version__, TF32=False, discarded_cost_weights=True,
            new_encoder_calls=0, heldout_quality_access='none',
            scope='Registered source-fit-only cost/numerical gate; no calibration/dev/test score or method utility.'))
        assert estimate <= 120, 'Prospective training/evaluation estimate exceeds120s per-seed gate'


if __name__ == '__main__': main()
