# Frozen SignRep → original translation trainers

Source pins, immutable code and licenses are in `vendor/sign2gpt`,
`vendor/how2sign_translation` and `evidence/translation_sources.json`. The default
recipes are the public author recipes; their historical SignRep integration revisions
are unknown (U10). Pre-extract every complete 16-frame window with stride2; retain
**unpooled T×768 features**, without augmentation, LoRA on SignRep, or pose inputs.

## Sign2GPT: Phoenix14T / CSL-Daily

Use `features` to export each official partition. Feature manifests must include
`video_id`, `split` (`train`, `val`, `test`), `video`, `translation`, `features`.
Relative feature/video paths resolve against the manifest directory. Every partition
must use the same encoder checkpoint; train/val/test source videos must be disjoint.

Write a JSON configuration with absolute paths:

```json
{
  "dataset": "phoenix14t",
  "train_manifest": "/path/train/features.jsonl",
  "val_manifest": "/path/val/features.jsonl",
  "test_manifest": "/path/test/features.jsonl",
  "output": "/path/translation_run",
  "llm": "facebook/xglm-1.7B",
  "max_seq_len": 512
}
```

For CSL-Daily use `dataset: "csl_daily"`; character BLEU splitting is enabled. Supply
the original `replacement_pickle` if applying the author's text replacements. Their
historical replacement mapping is not supplied by SignRep. Original defaults: batch
8 Phoenix / 4 CSL, AdamW lr3e-4 weight decay .001, warmup5 epochs, 100 epochs, CE
smoothing .1, LoRA rank4, gated adapters at all24 XGLM layers, beam4/max length64.
The trainable 768→512 linear+BatchNorm and original 512-dim MetaFormer replace the
original DinoV2 input; **no stage1/pseudo-gloss config or checkpoint is loaded**.

```bash
python -m signrep_baseline sign2gpt-train \
  --python method/jepa_baseline/runtime/translation_env/bin/python \
  --config /path/translation.json --inspect
# Remove --inspect to invoke the actual original stage2 Trainer.
```

`scripts/run_sign2gpt.py` isolates generic upstream imports and uses local text
logging. The upstream callback's missing personal `environment_variables` module is
provided as an empty compatibility module. Hugging Face cache is inside `runtime/`.
The original trainer reports teacher-forcing `obleu` each epoch and generated `ableu`
every10 epochs; report generated scores for a translation benchmark.

Verified components: original MetaFormer + original modified XGLM/adapters forward,
backward and generation with **real SignRep features and a random tiny local XGLM**.
Evidence `S5_sign2gpt_smoke.json` verifies integration, not BLEU or pretrained LLM quality.

## How2Sign: Tarrés Fairseq fork

Export each split independently:

```bash
python -m signrep_baseline how2sign-export \
  --manifest /path/train/features.jsonl \
  --output-tsv /path/how2sign/cvpr23.fairseq.i3d.train.how2sign.tsv \
  --feature-directory /path/how2sign/train
```

The original TSV loader recognizes the transport tag `i3d`; the arrays contain SignRep
features and provenance explicitly identifies them as such. Do not pad to1024 or
substitute actual I3D features. `integrations/how2sign_user` registers a subclass that
sets **only input projection width to768**, retaining the original 6×3-layer 256-dim
transformer. The pinned author's best YAML is:
`vendor/how2sign_translation/examples/sign_language/config/wicv_cvpr23/i3d_best/baseline_6_3_dp03_wd_2.yaml`.

Set `SAVE_DIR`, `WANDB_NAME`, `WANDB_PROJECT`, `I3D_DIR` according to that original YAML;
provide its trained lowercased unigram7000 tokenizer/dictionary or train the tokenizer
on **training text only** using the original scripts. Then launch from its repo:

```bash
python fairseq_cli/hydra_train.py \
  --config-dir examples/sign_language/config/wicv_cvpr23/i3d_best \
  --config-name baseline_6_3_dp03_wd_2 \
  common.user_dir=/ABS/method/jepa_baseline/integrations/how2sign_user \
  model._name=signrep_sign2text_transformer
```

Use the original generation/tokenizer/blacklist scripts for BLEU, reduced BLEU and
ROUGE/chrf metrics. Preserve beam5, smoothing.1, Adam(.9,.98), dropout.3, lr.001,
warmup2000 and original cosine periods. Original selected YAML weight decay is .1;
keep the actual YAML value despite its filename ending `wd_2`.

Verified: real exported features load through original `SignFeatsDataset`, original
6/3-layer network executes backward/Adam update and original Fairseq beam generation.
Evidence `S5_how2sign_smoke.json` uses random translation weights and synthetic text,
so does not establish benchmark scores.

## Environments

Original environment specifications remain in both vendor repos. Our integration
checks ran in isolated Python3.10, Torch2.1.1/Transformers4.31/Xformers.0.23; Fairseq
uses OmegaConf2.0.6/Hydra1.0.7/Pose-format.0.1.0. This differs from the original
Sign2GPT Docker Torch2.3/Xformers.0.26 and How2Sign Torch1.11/Python3.8. Record the
chosen environment and rerun the provided smoke scripts after changing it. Exact
historical environment equivalence is unverified. Separate processes are required
for the two original repositories' imports; never import their generic modules into
the RGB/teacher process.
