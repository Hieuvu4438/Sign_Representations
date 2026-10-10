from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from signcoord.config import relation_weight
from signcoord.data.collate import collate_ssl
from signcoord.data.dataset import SSLVideoDataset
from signcoord.losses.distributed_reduce import weighted_mean
from signcoord.losses.geometry import geometry_loss
from signcoord.losses.motion import velocity_loss
from signcoord.losses.regularization import variance_covariance
from signcoord.models.backbone_adapter import TinyEncoderAdapter
from signcoord.models.hiera_adapter import HieraAdapter
from signcoord.models.generic_hiera_adapter import GenericHieraAdapter
from signcoord.models.representation import SignCoordModel
from signcoord.provenance import atomic_json, execution_stamp, object_hash, sha256_file
from signcoord.targets.lag import lag_loss_terms
from signcoord.targets.motion import interval_velocity
from signcoord.targets.magnitude import magnitude_loss_terms
from signcoord.targets.shuffle import shuffled_relational_targets
from signcoord.targets.schema import ANCHORS, SCHEMA_HASH
from signcoord.targets.spectral import phase_loss_terms
from signcoord.targets.statistics import invert_standardization, read_statistics
from .checkpoint import atomic_checkpoint, load_checkpoint, restore_rng, rng_state
from .distributed import TrainingContext, unwrap, check_prediction_finite
from .selection import FrozenRetrievalSelection,selection_inputs


def make_dataset(c, dev=False):
    data=c["data"]; sampler=data["sampler"]
    return SSLVideoDataset(data["ssl_dev_manifest" if dev else "ssl_train_manifest"],data["pose_cache_root"],
             sampler["frames"],sampler["grid_fps"],sampler["spatial_size"],
             max_repeat_fraction=sampler["max_repeat_fraction_spectral"],normalization=data["normalization"],
             color_jitter=0. if dev else data["augmentation"]["color_jitter"],sampling_seed=c["seed"])


def make_model(c, initialize=True):
    m=c["model"]
    if m["backbone"] == "tiny_fixture":
        encoder=TinyEncoderAdapter(m["input_frames"],c["data"]["sampler"]["spatial_size"],m["tiny_dim"])
    elif m["backbone"] == "signrep_hiera_adapter":
        encoder=HieraAdapter(m["vendor_root"],m["input_frames"],m["initialization_checkpoint"] if initialize else None,
                            m["initialization_mode"] if initialize else "random_init",m.get("attention_kernel","vendor_5d"))
    elif m["backbone"] == "hiera_base_video_adapter":
        encoder=GenericHieraAdapter(m["generic_vendor_root"],m["input_frames"],
                            m["initialization_checkpoint"] if initialize else None,
                            m["initialization_mode"] if initialize else "random_init",m.get("attention_kernel","vendor_5d"))
    else:
        raise ValueError(f"backbone not implemented: {m['backbone']}")
    decoder={k:m["decoder"][k] for k in ("width","blocks","heads","mlp_ratio","dropout")}
    return SignCoordModel(encoder,decoder,m["regularization_projector"],
                          pooled_prior_frames=m["input_frames"] if c["arm"] == "A1_inspired" else None)


def optimizer_groups(model,c):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        encoder=name.startswith("encoder.")
        decay=not (p.ndim <= 1 or any(s in name for s in ("norm","pos_embed","embedding","query_base")))
        key=(encoder,decay)
        groups.setdefault(key,[]).append(p)
    return [{"params":params,"lr":c["train"]["encoder_lr" if enc else "new_modules_lr"],
             "weight_decay":c["train"]["weight_decay"] if decay else 0.,
             "group_name":("encoder" if enc else "new_modules")+("_decay" if decay else "_no_decay")}
            for (enc,decay),params in groups.items()]


def scientific_config_hash(c):
    stable=copy.deepcopy(c); stable.pop("run_dir")
    return object_hash(stable)


def paired_indices(length,batch,microstep,seed):
    batches=length//batch
    if batches == 0:
        raise ValueError("dataset smaller than a full microbatch")
    epoch,local=divmod(microstep,batches)
    generator=torch.Generator().manual_seed(seed+epoch)
    return torch.randperm(length,generator=generator)[local*batch:(local+1)*batch].tolist(),epoch


def batch_to_device(batch,device):
    return {"student_inputs":{k:v.to(device) for k,v in batch["student_inputs"].items()},
            "targets":{k:v.to(device) for k,v in batch["targets"].items()},"audit":batch["audit"]}


def validate_training_batch(batch):
    inputs,target=batch['student_inputs'],batch['targets']
    reliability=target['reliability']
    if not torch.isfinite(inputs['rgb']).all() or not torch.isfinite(inputs['times']).all():
        raise ValueError('nonfinite RGB/times before model forward')
    if not (torch.diff(inputs['times'],dim=1)>0).all():
        raise ValueError('training times must increase strictly')
    if not torch.isfinite(reliability).all() or ((reliability<0)|(reliability>1)).any():
        raise ValueError('invalid target reliability')
    if not torch.isfinite(target['global'][reliability>0]).all():
        raise ValueError('nonfinite reliable target coordinates')


def paired_keep_mask(encoder,sample_ids,c,update,device):
    seeds=[int(object_hash({'seed':c['seed'],'update':update,'sample_id':sample_id,
                           'policy':'per_sample_mask_independent_of_rank_and_module_initialization_v2'})[:16],16)
           for sample_id in sample_ids]
    masks=[encoder.random_mask(1,c['model']['mask_ratio'],device,
                                torch.Generator(device=device).manual_seed(seed)) for seed in seeds]
    return torch.cat(masks),seeds


def compute_loss(model,batch,c,stats,calibration,update,context=None):
    inputs,target=batch["student_inputs"],batch["targets"]
    ids=[a['sample_id'] for a in batch['audit']]
    core=unwrap(model)
    keep,mask_seeds=paired_keep_mask(core.encoder,ids,c,update,inputs['rgb'].device)
    if context is None:
        out=model(inputs["rgb"],keep,inputs["frame_valid"],inputs["times"])
    else:
        out=context.checked("model_forward",lambda:model(inputs["rgb"],keep,inputs["frame_valid"],inputs["times"]))
    check_prediction_finite(out,context)
    physical=invert_standardization(out["pred_standardized"].float(),stats["groups"])
    geom,glog=geometry_loss(physical,target["global"],target["reliability"],stats["groups"],c["loss"]["geometry"])
    velocity,vlog=velocity_loss(physical,target["global"],inputs["times"],target["reliability"],stats["groups"])
    var,cov=variance_covariance(out["projected"],c["loss"]["variance_target_std"])
    relation=physical.sum()*0; rlog={"denominator":0.,"mean":0.,"active":False}; diag={}
    kind=c["loss"]["relation"]["kind"]; rc=c["loss"]["relation"]; shuffle_log=None
    if kind != "off":
        pv,_=interval_velocity(physical[...,ANCHORS,:],inputs["times"],target["reliability"][...,ANCHORS])
        tv,rel=interval_velocity(target["global"][...,ANCHORS,:],inputs["times"],target["reliability"][...,ANCHORS])
        timeline=target["spectral_frame_valid"]
        eligibility=timeline[:,1:] & timeline[:,:-1]
        rel=rel*eligibility[...,None]
        # Missing coordinates never enter an FFT. Their zero reliability still
        # excludes complete windows; finite zero is only a numerical placeholder.
        tv=torch.where(rel[...,None]>0,tv,torch.zeros_like(tv))
        def terms(function):
            return function() if context is None else context.checked("relation_terms",function)
        if kind == "lag":
            n,d,diag=terms(lambda:lag_loss_terms(pv,tv,rel,rc["window"],rc["hop"],rc["lag_min"],rc["lag_max"],
                                   rc["energy_floor"],rc["min_confidence"]))
        elif kind == "magnitude":
            n,d,diag=terms(lambda:magnitude_loss_terms(pv,tv,rel,calibration["magnitude"],rc["window"],rc["hop"]))
        else:
            thresholds={k:pv.new_tensor(calibration["thresholds"][k]) for k in ("tau_p","tau_q","tau_cross")}
            basic=None
            if kind == "phase_basic":
                jv=target["joint_valid"][...,ANCHORS]
                basic=jv[:,1:] & jv[:,:-1] & eligibility[...,None]
            if kind == "phase_shuffle":
                jv=target["joint_valid"][...,ANCHORS]
                support=jv[:,1:] & jv[:,:-1] & eligibility[...,None]
                tv,rel,shuffle_log=shuffled_relational_targets(tv,rel,support,inputs["times"],
                             [a["sample_id"] for a in batch["audit"]],c["seed"],update)
            n,d,diag=terms(lambda:phase_loss_terms(pv,tv,rel,**thresholds,window=rc["window"],hop=rc["hop"],
                                     min_conf=rc["min_confidence"],basic_valid=basic))
        relation,rlog=weighted_mean(n,d)
    weight=relation_weight(c,update)
    total=geom+c["loss"]["velocity_weight"]*velocity+weight*relation+c["loss"]["variance_weight"]*var+c["loss"]["covariance_weight"]*cov
    log={"total":total.detach().item(),"geometry":geom.detach().item(),"velocity":velocity.detach().item(),
         "variance":var.detach().item(),"covariance":cov.detach().item(),"relation":rlog,
         "relation_weight":weight,"geometry_groups":glog,"velocity_groups":vlog,
         "actual_mask_ratio":1-float(keep.sum().item()/keep.numel()),
         'mask_sample_seeds':mask_seeds,
         "sample_ids":[a["sample_id"] for a in batch["audit"]],
         "mean_repeat_fraction":sum(a["repeat_fraction"] for a in batch["audit"])/len(batch["audit"])}
    if diag:
        log["relation_active_bins"]=int((diag["gate"]>0).sum().item())
        log["relation_mean_gate"]=diag["gate"].mean().item()
    if shuffle_log is not None: log["relation_shuffle"]=shuffle_log
    return total,log


def train(c,resume=None,stop_after=None):
    if c["arm"] in ("A0","A1_ref"):
        raise ValueError("reference arms are frozen; use init-reference or extract-signrep-reference")
    context=TrainingContext(c["train"]["device"])
    try:
        return _train(c,resume,stop_after,context)
    finally:
        context.close()


def _train(c,resume,stop_after,context):
    run=Path(c["run_dir"])
    def prepare_run():
        if resume is None and run.exists() and any(run.iterdir()):
            raise ValueError("run_dir exists; use explicit --resume or a new experiment path")
        run.mkdir(parents=True,exist_ok=True)
    context.rank_zero("prepare_run",prepare_run)
    train_cfg=c["train"]; device=context.device
    seed=c["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    dataset=context.checked("dataset",lambda:make_dataset(c))
    context.check_error("training_split",ValueError("SSL training cannot read downstream or SSL dev data")
                        if any(r["split"] != "ssl_train" for r in dataset.records) else None)
    stats=context.checked("target_statistics",lambda:read_statistics(c["data"]["target_stats_path"],c["data"]["ssl_train_manifest"],dataset))
    rc=c["loss"]["relation"]
    needs_calibration=rc["kind"] in ("phase","phase_basic","phase_shuffle","magnitude")
    def load_calibration():
        calibration=read_statistics(c["data"]["phase_calibration_path"],c["data"]["ssl_train_manifest"],dataset) if needs_calibration else None
        if needs_calibration:
            if calibration["window"] != rc["window"] or calibration["hop"] != rc["hop"]:
                raise ValueError("relation calibration window/hop mismatch")
            if calibration.get("min_confidence",.30) != rc["min_confidence"]:
                raise ValueError("relation calibration confidence policy mismatch")
            if rc["kind"] == "magnitude" and "magnitude" not in calibration:
                raise ValueError("A5 requires train-only magnitude calibration; refit statistics")
        return calibration
    calibration=context.checked("relation_calibration",load_calibration)
    manifest_hash,stats_hash,phase_hash=context.checked("input_hashes",lambda:(
        sha256_file(c["data"]["ssl_train_manifest"]),sha256_file(c["data"]["target_stats_path"]),
        sha256_file(c["data"]["phase_calibration_path"]) if needs_calibration else None))
    selected_inputs=context.checked('selection_inputs',lambda:selection_inputs(c))
    selector=FrozenRetrievalSelection(c,run,selected_inputs) if selected_inputs is not None else None
    selection_protocol_hash=selected_inputs['protocol_hash'] if selected_inputs is not None else None
    stamp=execution_stamp()
    protocol={"scientific_config_hash":scientific_config_hash(c),"manifest":manifest_hash,"stats":stats_hash,
              "calibration":phase_hash,"implementation":stamp["implementation_hash"],
              "run_dir":str(run.resolve()),"resume":str(Path(resume).resolve()) if resume else None,
              "stop_after":stop_after,'selection_protocol_hash':selection_protocol_hash}
    protocols=context.all_objects(protocol)
    if any(p != protocols[0] for p in protocols):
        raise ValueError("ranks disagree on scientific inputs or run/resume protocol")
    model=context.checked("model_initialization",lambda:make_model(c).to(device).train())
    optimizer=torch.optim.AdamW(optimizer_groups(model,c),betas=tuple(train_cfg["betas"]))
    def schedule(u):
        warm=train_cfg["warmup_updates"]; maximum=train_cfg["max_updates"]
        if u < warm: return (u+1)/max(warm,1)
        return .5*(1+math.cos(math.pi*min(1.,(u-warm)/max(maximum-warm,1))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    precision=train_cfg["precision"]
    context.check_error("precision",ValueError("CPU engineering runs require explicit fp32")
                        if device.type == "cpu" and precision != "fp32" else None)
    dtype=torch.bfloat16 if precision == "bf16" or (precision == "bf16_if_supported_else_fp16" and torch.cuda.is_bf16_supported()) else torch.float16
    amp=device.type == "cuda" and precision != "fp32"
    scaler=torch.amp.GradScaler(device.type,enabled=amp and dtype == torch.float16)
    start,microstep,elapsed_prior,evaluation_prior=0,0,0.,0.
    def read_resume():
        saved=load_checkpoint(resume)
        if saved["implementation_hash"] != stamp["implementation_hash"]:
            raise ValueError("exact resume requires the same implementation source hash")
        if saved["scientific_config_hash"] != scientific_config_hash(c):
            raise ValueError("resume scientific configuration differs from checkpoint")
        if saved["world_size"] != context.world:
            raise ValueError("exact resume requires the same world size")
        if saved.get('selection_protocol_hash')!=selection_protocol_hash:
            raise ValueError('resume selection inputs changed')
        if any(saved[k] != v for k,v in (("ssl_manifest_hash",manifest_hash),("target_stats_hash",stats_hash),("phase_calibration_hash",phase_hash))):
            raise ValueError("resume input hashes changed")
        model.load_state_dict(saved["model_state"],strict=True); optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"]); scaler.load_state_dict(saved["scaler_state"])
        if (saved.get('selection_state') or {}).get('best_checkpoint_self_reference'):
            if saved['selection_state']['best_update']!=saved['optimizer_update']:
                raise ValueError('best-dev checkpoint self-reference update mismatch')
            saved['selection_state']['best_checkpoint_path']=str(Path(resume).resolve())
            saved['selection_state']['best_checkpoint_sha256']=sha256_file(resume)
            saved['selection_state'].pop('best_checkpoint_self_reference')
        return saved
    saved=context.checked("resume",read_resume) if resume is not None else None
    if saved is not None:
        start=saved["optimizer_update"]; microstep=saved["sampler_state"]["next_microstep"]
        elapsed_prior=saved["compute_ledger"]["elapsed_training_seconds"]
        evaluation_prior=saved['compute_ledger'].get('elapsed_dev_selection_seconds',0.)
        if selector is not None:
            def restore_selection():
                selector.restore(saved.get('selection_state'))
                return selector.state
            selector.state=context.rank_zero_long('selection_restore',restore_selection)
    core=model
    model=context.wrap(core)
    # DDP constructor may consume RNG. Restore after construction at a saved boundary.
    if saved is not None:
        context.checked("rank_rng_restore",lambda:restore_rng(saved["rng_states_by_rank"][context.rank]
                        if "rng_states_by_rank" in saved else saved["rng_state"]))
    else:
        rank_seed=seed+context.rank*1000003
        random.seed(rank_seed); np.random.seed(rank_seed % (2**32)); torch.manual_seed(rank_seed)
        if device.type == "cuda":torch.cuda.manual_seed_all(rank_seed)
    context.rank_zero("resolved_config",lambda:atomic_json(run/"resolved_config.json",c))
    context.rank_zero("provenance",lambda:atomic_json(run/"provenance.json",{**stamp,"config_hash":object_hash(c),"scientific_config_hash":scientific_config_hash(c),
       "ssl_manifest_hash":manifest_hash,"target_stats_hash":stats_hash,"phase_calibration_hash":phase_hash,
       "schema_hash":SCHEMA_HASH,"environment":{"torch":torch.__version__,"numpy":np.__version__,"python":os.sys.version},
       "data_kind":"synthetic" if c["model"]["backbone"] == "tiny_fixture" else "real",
       "checkpoint_load_report":getattr(core.encoder,"load_report",{}),"world_size":context.world,
       "distributed_backend":torch.distributed.get_backend() if context.world>1 else None,
       "checkpoint_selection":'fixed_budget_final_primary_dev_selected_secondary' if selector is not None else 'fixed_budget_final',
       'selection_protocol_hash':selection_protocol_hash}))
    batch=train_cfg["microbatch_per_gpu"]; accum=train_cfg["accumulation_steps"]
    maximum=train_cfg["max_updates"] if stop_after is None else min(stop_after,train_cfg["max_updates"])
    if device.type == "cuda": torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    begin=time.perf_counter(); update=start; any_active_supervision=False; evaluation_wait=0.
    logfile=context.checked("open_metrics",lambda:(run/"metrics.jsonl").open("a" if resume else "w") if context.rank==0 else None)
    try:
        while update < maximum:
            optimizer.zero_grad(set_to_none=True); logs=[]
            for accumulation_index in range(accum):
                indices,epoch=paired_indices(len(dataset),batch*context.world,microstep,seed)
                indices=indices[context.rank*batch:(context.rank+1)*batch]
                dataset.sample_epoch=epoch
                b=context.checked("load_microbatch",lambda:batch_to_device(collate_ssl([dataset[i] for i in indices]),device))
                context.checked("validate_microbatch",lambda:validate_training_batch(b))
                reliable_count=(b["targets"]["reliability"]>0).sum()
                if context.world>1:torch.distributed.all_reduce(reliable_count)
                if not reliable_count.item():
                    raise ValueError("all-invalid geometry microbatch; rejecting optimizer step")
                with context.accumulation_context(model,accumulation_index==accum-1):
                    with torch.autocast(device.type,dtype=dtype,enabled=amp):
                        loss,log=compute_loss(model,b,c,stats,calibration,update,context)
                    context.check_error("loss",ValueError("nonfinite loss; run terminated") if not torch.isfinite(loss) else None)
                    scaler.scale(loss/accum).backward()
                microstep+=1; logs.append(log)
                any_active_supervision |= log['relation_weight']>0 and log['relation']['active']
            scaler.unscale_(optimizer)
            norm=context.checked("gradient_clip",lambda:torch.nn.utils.clip_grad_norm_(core.parameters(),train_cfg["grad_clip_norm"],error_if_nonfinite=True))
            previous_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
            context.check_error("optimizer_step",ValueError("AMP skipped update; diagnostic/resume required")
                                if scaler.get_scale()<previous_scale else None)
            scheduler.step(); update+=1
            rank_logs=context.all_objects(logs)
            record={"optimizer_update":update,"grad_norm_before_clip":norm.item(),"microsteps":logs,
                    "learning_rates":scheduler.get_last_lr()}
            if context.world>1:
                record['rank_microsteps']=rank_logs
                record['global_mean_loss']=sum(m['total'] for rank in rank_logs for m in rank)/(context.world*accum)
                record['microstep_loss_scope']='rank_scaled_contribution; rank_average_is_global_objective'
            def write_record():
                logfile.write(json.dumps(record,allow_nan=False)+"\n"); logfile.flush()
            context.rank_zero("write_metrics",write_record)
            evaluation_due=selector is not None and (update%train_cfg['eval_every_updates']==0 or update==train_cfg['max_updates'])
            improved=False
            if evaluation_due:
                evaluation_begin=time.perf_counter()
                result=context.rank_zero_long('frozen_dev_selection',lambda:selector.evaluate(core,update,device))
                evaluation_wait+=time.perf_counter()-evaluation_begin
                improved=selector.record(result)
            if update % train_cfg["save_every_updates"] == 0 or update == maximum or evaluation_due:
                if device.type == "cuda": torch.cuda.synchronize(device)
                rank_rng=context.all_objects(rng_state(device))
                rank_elapsed=context.all_objects(elapsed_prior+time.perf_counter()-begin-evaluation_wait)
                rank_evaluation=context.all_objects(evaluation_prior+evaluation_wait)
                rank_peak=context.all_objects(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
                elapsed=max(rank_elapsed)
                ledger={"elapsed_training_seconds":elapsed,"device":str(device),"world_size":context.world,
                        "stats_batch":batch*context.world,"optimizer_batch":batch*context.world*accum,
                        "accumulation_contract":"mean_of_microstep_global_weighted_means",
                        "training_gpu_hours":elapsed*context.world/3600 if device.type == "cuda" else 0.,
                        'elapsed_dev_selection_seconds':max(rank_evaluation),
                        'dev_selection_gpu_hours':rank_evaluation[0]/3600 if device.type=='cuda' else 0.,
                        'elapsed_scope':'training excludes frozen-dev evaluation/wait; peak allocation includes both',
                        "peak_allocated_bytes":max(rank_peak) if device.type == "cuda" else None,
                        "peak_allocated_bytes_by_rank":rank_peak}
                content={"schema_version":"signcoord_checkpoint_v1","model_state":core.state_dict(),
                   "optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"scaler_state":scaler.state_dict(),
                   "optimizer_update":update,"microstep_boundary":0,"epoch":epoch,
                   "sampler_state":{"next_microstep":microstep,"sample_plan_seed":seed,
                                    "world_size":context.world,"global_microbatch":batch*context.world},
                   "rng_state":rank_rng[0],"rng_states_by_rank":rank_rng,
                   "resolved_config":c,"config_hash":object_hash(c),"scientific_config_hash":scientific_config_hash(c),
                   **stamp,"ssl_manifest_hash":manifest_hash,"target_stats_hash":stats_hash,"phase_calibration_hash":phase_hash,
                   "pose_schema_hash":SCHEMA_HASH,"precision":precision,"world_size":context.world,"compute_ledger":ledger,
                   "initialization_checkpoint_hash":getattr(core.encoder,"load_report",{}).get("checkpoint_sha256"),
                   "load_report":getattr(core.encoder,"load_report",{}),
                   'selection_protocol_hash':selection_protocol_hash,
                   'selection_state':copy.deepcopy(selector.state) if selector is not None else None,
                   "best_dev_metric":selector.state['best_metric'] if selector is not None else None,
                   "selection_rule":"fixed_budget_final"}
                def save():
                    if improved:
                        best=copy.copy(content);best['selection_rule']='dev_rec1_max_earliest_tie_secondary'
                        best['selection_state']=copy.deepcopy(selector.state)
                        best['selection_state']['best_checkpoint_sha256']=None
                        best['selection_state']['best_checkpoint_self_reference']=True
                        atomic_checkpoint(run/'checkpoints/best_dev.pt',best)
                        selector.state['best_checkpoint_sha256']=sha256_file(run/'checkpoints/best_dev.pt')
                    if selector is not None:
                        content['selection_state']=copy.deepcopy(selector.state)
                        selector.save_history()
                    atomic_checkpoint(run/"checkpoints/last.pt",content)
                    if update == train_cfg["max_updates"]: atomic_checkpoint(run/"checkpoints/final.pt",content)
                    atomic_json(run/"compute_ledger.json",ledger)
                context.rank_zero("checkpoint_write",save)
                if selector is not None:
                    selector.state=context.all_objects(selector.state)[0]
    finally:
        if logfile is not None:logfile.close()
    report={**stamp,"status":"passed","optimizer_updates":update,"arm":c["arm"],"run_dir":str(run),
            "scientific_evidence":False, "real_data_training":c["model"]["backbone"] != "tiny_fixture", "config_hash":object_hash(c),
            "manifest_hash":manifest_hash,"checkpoint_sha256":sha256_file(run/"checkpoints/last.pt"),
            "relation_activated":any(m["relation_weight"] > 0 for m in logs) if update > start else False}
    report['relation_active_supervision']=bool(any_active_supervision)
    report['world_size']=context.world
    report['selection_protocol_hash']=selection_protocol_hash
    report['best_dev_metric']=selector.state['best_metric'] if selector is not None else None
    context.rank_zero("training_report",lambda:atomic_json(run/"training_report.json",report))
    return report
