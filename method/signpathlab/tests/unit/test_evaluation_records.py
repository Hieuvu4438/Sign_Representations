from dataclasses import replace
import numpy as np
import pytest
import torch
from signpath.evaluation.bootstrap import classification_bootstrap, paired_classification_bootstrap
from signpath.evaluation.phonology import fit_phonology_probe, macro_f1
from signpath.evaluation.records import quality_summary, write_jsonl
from signpath.evaluation.translation_metrics import translation_metrics


def test_quality_proxy_and_unknown_phonology_are_not_fabricated(bundle):
    assert quality_summary(bundle)["bucket"] == "high"
    changed = replace(bundle, visual_valid=np.zeros_like(bundle.visual_valid), pose_coordinate_valid=np.zeros_like(bundle.pose_coordinate_valid))
    assert quality_summary(changed)["bucket"] == "low"
    result = fit_phonology_probe(np.zeros((2, 3)), [-100, -100], np.zeros((2, 3)), [-100, -100])
    assert result["status"] == "not_run" and "accuracy" not in result


def test_macro_f1_hand_calculation_and_source_bootstrap():
    assert macro_f1([0, 0, 1, 1], [0, 0, 0, 1], [0, 1]) == pytest.approx((.8 + 2 / 3) / 2)
    result = classification_bootstrap([1, 1, 1, 1], [0, 0, 1, 1], ["a", "a", "b", "b"], samples=20, resampling_unit="official_signer_id")
    assert result["per_class_top1_ci95"] == [1, 1] and result["resampling_unit"] == "official_signer_id"
    assert classification_bootstrap([1], [0], ["only"], samples=20)["status"] == "insufficient_independent_sources"


def test_paired_macro_difference_uses_labels_and_whole_groups():
    result = paired_classification_bootstrap([0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1],
        ["signer_a"] * 3 + ["signer_b"], samples=100, resampling_unit="signer_id")
    assert result["per_instance_difference"] == .75
    assert result["per_class_difference"] == .5
    assert result["group_count"] == 2 and result["resampling_unit"] == "signer_id"
    assert result["per_class_difference_ci95"] == [0., 1.]
    tied = paired_classification_bootstrap([0, 1, 1], [0, 1, 1], [0, 1, 2], ["a", "b", "c"], samples=20)
    assert tied["per_class_difference_ci95"] == [0., 0.]


def test_paired_prediction_comparison_checks_reference_alignment(tmp_path):
    import json
    from signpath.experiments import paired_prediction_intervals
    row = {"sample_id": "sample", "reference_label_id": 0, "prediction_label_id": 0,
           "signer_id": None, "source_group_id": "source"}
    first, second = tmp_path / "a", tmp_path / "b"
    for directory in (first, second):
        directory.mkdir()
        (directory / "predictions.val.jsonl").write_text(json.dumps(row) + "\n")
    assert paired_prediction_intervals(first, second)["status"] == "insufficient_independent_groups"
    (second / "predictions.val.jsonl").write_text(json.dumps({**row, "reference_label_id": 1}) + "\n")
    with pytest.raises(ValueError, match="alignment"):
        paired_prediction_intervals(first, second)


def test_verified_annotation_probe_masks_unknowns():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train = np.array([[1., 0], [0, 1], [99, 99]], dtype=np.float32)
        result = fit_phonology_probe(train, [0, 1, -100], train[:2], [0, 1],
            annotation_provenance={"source_sha256": "a" * 64, "scope": "synthetic_correctness_only"}, epochs=10)
        assert result["status"] == "completed" and result["train_known"] == result["val_known"] == 2
    finally:
        torch.set_num_threads(previous)


def test_translation_metric_signature_and_exact_match():
    pytest.importorskip("sacrebleu")
    text = ["this is a sufficiently long exact sentence"]
    result = translation_metrics(text, text)
    assert result["bleu"] == pytest.approx(100)
    assert result["rouge_l_f1"] == 1 and result["wer"]["wer"] == 0
    assert "tok:13a" in result["bleu_signature"] and "case:mixed" in result["bleu_signature"]
