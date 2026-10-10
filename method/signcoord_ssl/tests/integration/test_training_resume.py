import copy
import json
from pathlib import Path
import subprocess
import sys
import shutil
import numpy as np

import pytest
import torch

from signcoord.config import load_config,relation_weight
from signcoord.engine.checkpoint import load_checkpoint
from signcoord.engine.train import make_dataset,train
from signcoord.targets.statistics import fit_target_statistics,read_statistics

PROJECT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def fixture_config(tmp_path_factory):
    root=tmp_path_factory.mktemp("fixture_parent")/"fixture"
    subprocess.run([sys.executable,str(PROJECT/"tools/generate_synthetic_fixture.py"),"--output",str(root)],check=True,capture_output=True)
    c=load_config(root/"fixture_smoke.yaml")
    fit_target_statistics(make_dataset(c),c["data"]["ssl_train_manifest"],c["data"]["target_stats_path"],c["data"]["phase_calibration_path"])
    return c


def test_exact_optimizer_boundary_resume(fixture_config,tmp_path):
    torch.set_num_threads(2)
    full=copy.deepcopy(fixture_config); full["run_dir"]=str(tmp_path/"full")
    interrupted=copy.deepcopy(fixture_config); interrupted["run_dir"]=str(tmp_path/"interrupted")
    train(full)
    train(interrupted,stop_after=4)
    checkpoint=tmp_path/"interrupted/checkpoints/last.pt"
    assert load_checkpoint(checkpoint)["optimizer_update"] == 4
    train(interrupted,resume=checkpoint)
    a=load_checkpoint(tmp_path/"full/checkpoints/final.pt")
    b=load_checkpoint(tmp_path/"interrupted/checkpoints/final.pt")
    assert a["sampler_state"] == b["sampler_state"]
    assert torch.equal(a["rng_state"]["torch"],b["rng_state"]["torch"])
    for key in a["model_state"]:
        assert torch.equal(a["model_state"][key],b["model_state"][key]),key
    assert a["scheduler_state"] == b["scheduler_state"]
    log_a=[json.loads(r) for r in (tmp_path/"full/metrics.jsonl").read_text().splitlines()]
    log_b=[json.loads(r) for r in (tmp_path/"interrupted/metrics.jsonl").read_text().splitlines()]
    assert log_a == log_b
    assert any(m["relation_weight"]>0 and m["relation"]["active"] for record in log_a for m in record["microsteps"])


@pytest.mark.parametrize("arm",["A1_inspired","A2","A3","A5","A6","A8","A9"])
def test_matched_controls_fixture(fixture_config,tmp_path,arm):
    c=load_config(Path(fixture_config["data"]["ssl_train_manifest"]).parents[1]/"fixture_smoke.yaml",arm=arm)
    c["run_dir"]=str(tmp_path/arm)
    if arm in ("A1_inspired","A2","A3","A6"):
        c["data"]["phase_calibration_path"]=None
    report=train(c,stop_after=4)
    assert report["optimizer_updates"] == 4 and not report["scientific_evidence"]
    if arm == "A9":
        records=[json.loads(line) for line in (tmp_path/arm/"metrics.jsonl").read_text().splitlines()]
        for record in records:
            for micro in record["microsteps"]:
                for own,other in zip(micro["sample_ids"],micro["relation_shuffle"]["target_sample_ids"]):
                    assert other is None or own != other


def test_warmup_scientific_recipe():
    c=load_config(PROJECT/"configs/base.yaml")
    assert relation_weight(c,0)==0 and relation_weight(c,2000)==0
    assert relation_weight(c,3500)==pytest.approx(.025)
    assert relation_weight(c,5000)==.05


def test_unknown_config_and_invalid_geometry():
    with pytest.raises(ValueError,match="unknown config"):
        load_config(PROJECT/"configs/base.yaml",["model.fake=1"])
    with pytest.raises(ValueError,match="match encoder"):
        load_config(PROJECT/"configs/base.yaml",["model.input_frames=64"])
    with pytest.raises(ValueError,match="complete velocity windows"):
        load_config(PROJECT/"configs/base.yaml",["loss.relation.window=32"])


def test_pose_mutation_invalidates_live_reads_and_train_statistics(fixture_config,tmp_path):
    current=copy.deepcopy(fixture_config)
    shutil.copytree(current['data']['pose_cache_root'],tmp_path/'pose')
    current['data']['pose_cache_root']=str(tmp_path/'pose')
    dataset=make_dataset(current)
    read_statistics(current['data']['target_stats_path'],current['data']['ssl_train_manifest'],dataset)
    dataset[0]
    path=dataset.pose_path(dataset.records[0]['pose_cache_key'])
    with np.load(path,allow_pickle=False) as source:
        arrays={key:source[key].copy() for key in source.files}
    arrays['confidence'][0,9]=.8
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='changed'):
        dataset[0]
    changed=make_dataset(current)
    with pytest.raises(ValueError,match='pose contents changed'):
        read_statistics(current['data']['target_stats_path'],current['data']['ssl_train_manifest'],changed)


def test_statistics_reject_changed_normalization_recipe(fixture_config):
    current=copy.deepcopy(fixture_config)
    current['data']['normalization']['min_scale_fraction']=.03
    with pytest.raises(ValueError,match='recipe changed'):
        read_statistics(current['data']['target_stats_path'],current['data']['ssl_train_manifest'],make_dataset(current))


@pytest.mark.parametrize('setting',['train.optimizer=sgd','train.scheduler=constant','train.precision=fake',
    'model.encoder_taps=[early, final]','data.sampler.resize=direct_resize',
    'data.normalization.origin=per_frame_center','model.mask_semantics=true_means_drop'])
def test_unsupported_recipes_are_rejected_instead_of_silently_ignored(setting):
    with pytest.raises(ValueError):load_config(PROJECT/'configs/base.yaml',[setting])
