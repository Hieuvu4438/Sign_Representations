"""Exercise real WLASL poses and pretrained mT5; no mocks or synthetic data."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import utils
from config import train_label_paths
from datasets import S2T_Dataset
from models.loop import Sign_Loop_Hyperbolic
from loop_training import recursive_to_float_cuda


def check(model, src, tgt, mode, num_loop):
    model.args.loop = model.loop_type = mode
    model.args.num_loop = num_loop
    model.enable_hyp_alignment = 'Hyperbolic' in mode
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(src, tgt)
    assert torch.isfinite(out['loss']), (mode, 'nonfinite loss')
    out['loss'].backward()
    gradients = {}
    for name in ('pose_proj.weight', 'mt5_model.encoder.block.0.layer.0.SelfAttention.q.weight'):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, (mode, name)
        gradients[name] = grad.norm().item()
    if model.enable_hyp_alignment:
        grad = model.hyp_proj_body.linear.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, mode
        assert model.loss_alpha_logit.grad is not None, mode
        gradients['hyp_proj_body'] = grad.norm().item()
    loss = out['loss'].item()
    del out
    model.zero_grad(set_to_none=True)
    model.eval()
    seen_refined = []
    hook = model.mt5_model.decoder.register_forward_pre_hook(
        lambda module, args, kwargs: seen_refined.append(kwargs.get('inputs_embeds') is not None), with_kwargs=True)
    with torch.no_grad():
        out = model(src, tgt)
        seen_refined.clear()
        generated = model.generate(out, max_new_tokens=3, num_beams=2)
        assert generated.shape[0] == len(tgt['gt_sentence'])
        if 'SignDec' in mode and num_loop:
            assert len(seen_refined) > num_loop and seen_refined[num_loop], (mode, 'refined decoder prefix discarded')
    hook.remove()
    result = {'mode': mode, 'loops': num_loop, 'loss': loss, 'gradients': gradients,
              'generation_shape': list(generated.shape), 'manifold': model.args.manifold}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/runtime_checks.json')
    opts = parser.parse_args()
    args = utils.get_args_parser().parse_args([])
    args.device = torch.device('cuda')
    args.n_layers, args.num_loop, args.max_length = 6, 2, 16
    args.hyp_dim = 128  # Catch silent assumptions that geometry dimension == GCN dimension.
    utils.set_seed(42)
    ds = S2T_Dataset(train_label_paths['WLASL2000'], args, 'train')
    src, tgt = ds.collate_fn([ds[0], ds[1]])
    src = recursive_to_float_cuda(src, args.device)
    results = []
    start = time.time()
    model = Sign_Loop_Hyperbolic(args).to(args.device)
    for mode in ['SignEc', 'SignDec', 'SignEcDec', 'SignEcHyperbolic', 'SignDecHyperbolic', 'SignEcDecHyperbolic']:
        results.append(check(model, src, tgt, mode, 2))
    results.append(check(model, src, tgt, 'SignEcDecHyperbolic', 0))
    args.use_dynamic_loops = True
    results.append(check(model, src, tgt, 'SignEcDecHyperbolic', 2))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    args.use_dynamic_loops = False
    for manifold in ['Euclidean', 'PoincareBall', 'Lorentz', 'AdaptiveLorentz']:
        args.manifold = manifold
        args.loop = 'SignEcDecHyperbolic'
        model = Sign_Loop_Hyperbolic(args).to(args.device)
        results.append(check(model, src, tgt, args.loop, 2))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    path = Path(opts.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'seconds': time.time() - start, 'checks': results}, indent=2))


if __name__ == '__main__':
    main()
