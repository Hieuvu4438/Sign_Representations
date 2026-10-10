"""Pinned component parity plus independent loss/gradient/sampling oracles."""
import ast
from math import log
from pathlib import Path
import numpy as np
import torch
from rejection_heads import FRDiscMLP, ResidualLinear, balanced_logit_loss, selected_bce_indices, training_plan, validate_training_plan, episode_inputs, flat_parameters


def run_tests(reference):
    tree = ast.parse(Path(reference).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)
             and n.name in ('BinaryClassificationModelSAFSAR', 'OpenSetLoss')]
    assert len(nodes) == 2
    scope = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), reference, 'exec'), scope)
    torch.set_num_threads(1)
    own = FRDiscMLP().double(); author = scope['BinaryClassificationModelSAFSAR'](768).double()
    author.load_state_dict(own.state_dict())
    assert sum(p.numel() for p in own.parameters()) == 2410881
    assert sum(p.numel() for p in ResidualLinear().parameters()) == 769
    rng = np.random.default_rng(1701); errors = []
    for _ in range(10):
        x = torch.from_numpy(rng.normal(size=(7, 768)))
        a, b = own(x).sigmoid(), author(x).squeeze(-1)
        errors.append(float((a - b).abs().max())); assert torch.equal(a, b)
    sim = torch.tensor([[.9, .1], [.1, .9], [.7, .3], [.5, .5]], dtype=torch.float64)
    target = torch.tensor([0, 0, -1, -1])
    z = torch.tensor([0., 17., log(3), -17.], dtype=torch.float64, requires_grad=True)
    loss = balanced_logit_loss(z, sim, target)
    assert abs(float(loss.detach()) - (log(2) + log(4)) / 2) < 1e-12
    loss.backward(); assert torch.allclose(z.grad, torch.tensor([-.25, 0., .375, 0.], dtype=torch.float64), rtol=0, atol=1e-12)
    selected, _ = selected_bce_indices(sim, target); assert selected.tolist() == [2, 0]
    ref = scope['OpenSetLoss']('discriminator')
    # Author explicitly makes FP32 labels; use its native FP32 loss inputs.
    ref_loss = ref.discriminator_loss({'disc_prob': z.detach().float().sigmoid()[:, None]}, target, sim.float())['os_loss']
    assert abs(float(ref_loss) - float(loss.detach())) < 1e-6
    assert balanced_logit_loss(z, sim, torch.tensor([1, 0, -1, -1])) is None
    # Swapping the unknown rows changes which negative is selected, as declared.
    altered = z.detach().clone(); altered[2], altered[3] = z.detach()[3], z.detach()[2]
    assert balanced_logit_loss(altered, sim, target) < loss.detach()
    rows = [dict(sample_id=f'{c:02d}-{p}', class_id=c, signer=p, source_group=f'{c:02d}-{p}')
            for c in range(5) for p in ('A', 'B', 'C')]
    plan = training_plan(rows, steps=30, size=2, queries=3, seed=1701)
    assert validate_training_plan(rows, plan, size=2, queries=3)
    assert np.array_equal(plan, training_plan(rows, steps=30, size=2, queries=3, seed=1701))
    bad = plan.copy(); bad[0, 1] = bad[0, 0]
    try: validate_training_plan(rows, bad, size=2, queries=3)
    except ValueError: pass
    else: raise AssertionError('Duplicate support accepted')
    labels = np.array([0, 1, 2, 0, 2])
    vectors = np.array([[1., 0.], [0., 1.], [-1., 0.], [1., 0.], [0., -1.]])
    residual, scores, targets = episode_inputs(vectors, labels, np.array([0, 1, 3, 4]), size=2)
    assert np.array_equal(scores, [[1., 0.], [0., -1.]])
    assert np.array_equal(residual, [[0., 0.], [-1., -1.]]) and targets.tolist() == [0, -1]
    parameters, schema = flat_parameters([ResidualLinear(), FRDiscMLP()])
    assert parameters.dtype == np.float32 and parameters.size == 2411650
    assert schema[-1]['stop'] == len(parameters) and all(a['stop'] == b['start'] for a, b in zip(schema, schema[1:]))
    return dict(passed=True, groups=['author_architecture_ten_FP64_sigmoid_parities',
        'manual_loss_gradient_wrong_known_and_first_unknown_selection',
        'no_correct_known_skip', 'thirty_fixed_signer_matched_episode_contracts',
        'manual_signed_residual_targets_and_flat_weight_schema'],
        maximum_author_score_error=max(errors), MLP_parameters=2410881, linear_parameters=769,
        scope='Selected AST components only; no upstream module import/full training or task quality.')
