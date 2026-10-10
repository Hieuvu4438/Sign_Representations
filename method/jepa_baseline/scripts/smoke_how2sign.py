"""Original Tarrés transformer/Fairseq loader plus registered 768-dim input adapter."""
import importlib.metadata
import json
from pathlib import Path
import sys
from argparse import Namespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "integrations"))
sys.path.insert(0, str(ROOT / "vendor/how2sign_translation"))
import numpy as np
import torch
import yaml
from fairseq.data import Dictionary
from fairseq.data.sign_language import SignFeatsDataset
from fairseq.models.sign_to_text.sign2text_transformer import Sign2TextTransformerConfig
from how2sign_user import SignRepSign2TextTransformer
from signrep_baseline.translation import export_how2sign


def main():
    torch.set_num_threads(2)
    torch.manual_seed(27)
    folder = ROOT / "runtime/smoke"
    manifest = folder / "how2sign_smoke.jsonl"
    manifest.write_text(json.dumps({"video_id": "technical1", "features": str(folder / "features.npz"),
                                    "split": "train", "translation": "technical sample"}) + '\n')
    tsv = folder / "cvpr23.fairseq.i3d.train.how2sign.tsv"
    export_how2sign(manifest, tsv, folder / "how2sign_features")
    dataset = SignFeatsDataset.from_manifest_file(tsv)
    sample = dataset[0]
    assert sample["source"].shape == (11, 768)
    cfg = Sign2TextTransformerConfig()
    source_config = ROOT / "vendor/how2sign_translation/examples/sign_language/config/wicv_cvpr23/i3d_best/baseline_6_3_dp03_wd_2.yaml"
    recipe = yaml.safe_load(source_config.read_text())
    for key, value in recipe["model"].items():
        if key != "_name":
            setattr(cfg, key, value)
    cfg.max_source_positions, cfg.max_target_positions, cfg.feats_type = 1024, 1024, "i3d"
    dictionary = Dictionary()
    for i in range(28):
        dictionary.add_symbol(f"token{i}")
    model = SignRepSign2TextTransformer.build_model(cfg, Namespace(target_dictionary=dictionary)).train()
    assert model.encoder.feat_proj.in_features == 768
    assert len(model.encoder.transformer_layers) == 6 and len(model.decoder.layers) == 3
    features = sample["source"][None]
    padding = torch.zeros(1, 11, dtype=torch.bool)
    previous = torch.tensor([[2, 4, 5]])
    targets = torch.tensor([[4, 5, 2]])
    output = model(src_tokens=features, encoder_padding_mask=padding, prev_output_tokens=previous)
    logits = output[0]
    assert logits.shape == (1, 3, len(dictionary))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(dictionary)), targets.reshape(-1), label_smoothing=.1)
    loss.backward()
    assert model.encoder.feat_proj.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .98), eps=1e-8, weight_decay=.1)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    model.eval()
    from fairseq.sequence_generator import SequenceGenerator
    generator = SequenceGenerator([model], dictionary, beam_size=2, max_len_b=8)
    with torch.inference_mode():
        hypotheses = generator.generate([model], {"net_input": {"src_tokens": features, "encoder_padding_mask": padding}})
    assert hypotheses and len(hypotheses[0]) > 0
    report = {"result": "PASS", "scope": "technical integration with random translation weights; no BLEU reproduction",
              "original_Fairseq_feature_loader": True, "feature_shape": list(sample["source"].shape),
              "original_encoder_layers": 6, "original_decoder_layers": 3, "original_hidden_dim": 256,
              "only_architecture_change": "input projection width 1024->768", "original_generation": True,
              "generated_tokens": len(hypotheses[0][0]["tokens"]), "cross_entropy": float(loss.detach()),
              "versions": {name: importlib.metadata.version(name) for name in ["torch", "omegaconf", "hydra-core", "pose-format"]}}
    (ROOT / "evidence/S5_how2sign_smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
