"""Pinned original Sign2GPT temporal/XGLM components with real SignRep features.

Run in runtime/translation_env. Uses a tiny RANDOM local language model solely to
verify feature integration, backward, and original generation; not translation scores.
"""
import json
import sys
import importlib.metadata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor/sign2gpt"))
import numpy as np
import torch
from transformers import XGLMConfig
from models.huggingface.modeling_xglm import XGLMForCausalLM
from models.trial_models.test_stage2_model import Model


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    if not torch.cuda.is_available():
        raise RuntimeError("Original Sign2GPT xformers attention requires CUDA; no substitute kernel")
    checkpoint = ROOT / "runtime/smoke/tiny_xglm"
    config = XGLMConfig(vocab_size=32, d_model=64, num_layers=2, attention_heads=4,
                        ffn_dim=128, max_position_embeddings=64, dropout=0., attention_dropout=0.,
                        pad_token_id=1, bos_token_id=0, eos_token_id=2)
    XGLMForCausalLM(config).save_pretrained(checkpoint)
    model = Model(stage1_name="signrep_baseline.translation_sign2gpt", stage1_params={}, stage1_ckpt=None,
                  post_name="models.post_models.linear_pos_head", post_params={"pos_type": "sine", "pre_pos": True},
                  lang_backbone_name="models.huggingface.modeling_xglm", llm_name=str(checkpoint),
                  adaptor_params={"adapt_layers": [0, 1], "lora_layers": [0, 1], "w_lora_ff": False,
                                  "lora_rank": 4, "lora_drop": .1, "gate_type": "clamp", "lora_a": 4., "adapt_tokens": False},
                  pretext_length=1, freeze=False).cuda().train()
    with np.load(ROOT / "runtime/smoke/features.npz") as file:
        features = torch.from_numpy(file["features"].copy()).cuda()
    sequences = [features[:8], features[:7]]
    inputs = torch.tensor([[0, 4, 5], [0, 6, 7]], device="cuda")
    target = torch.tensor([[4, 5, 2], [6, 7, 2]], device="cuda")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=.001)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype):
        result = model(inputs, torch.ones_like(inputs, dtype=torch.bool), sequences,
                       torch.ones(2, 16, dtype=torch.bool, device="cuda"), max_len=16)
    assert result["logits"].shape == (2, 3, 32)
    loss = torch.nn.functional.cross_entropy(result["logits"].float().reshape(-1, 32), target.reshape(-1), label_smoothing=.1)
    loss.backward()
    assert model.sign_model.lin.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None for n, p in model.lang_model.named_parameters() if "adaptor" in n)
    assert not features.requires_grad
    optimizer.step()
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        generated = model(inputs[:, :1], torch.ones(2, 1, dtype=torch.bool, device="cuda"), sequences,
                          torch.ones(2, 16, dtype=torch.bool, device="cuda"), max_len=16, generate=True,
                          gen_params={"max_length": 4, "num_beams": 1})
    report = {"result": "PASS", "scope": "technical integration; random tiny XGLM, no BLEU reproduction",
              "feature_shape": list(features.shape), "original_temporal_hidden_dim": model.sign_model.dim,
              "original_XGLM_adapter_forward_backward": True, "original_generation": True,
              "generated_shape": list(generated["output_ids"].shape), "cross_entropy": float(loss.detach()),
              "pseudo_gloss_stage_omitted": True, "frozen_input_features": True,
              "versions": {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'xformers']}}
    (ROOT / "evidence/S5_sign2gpt_smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
