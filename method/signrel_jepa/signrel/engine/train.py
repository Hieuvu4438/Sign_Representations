import contextlib
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from signrel.data.manifest import read_manifest,sha256
from signrel.data.leakage import assert_disjoint
from signrel.data.video_dataset import SSLVideoDataset,ssl_collate
from signrel.data.sampler import SeededDataset,ResumableBatchSampler
from signrel.models.signrel_jepa import SignRelJEPA
from signrel.models.ema import ema_momentum
from .checkpoint import save_checkpoint,load_checkpoint,rng_state
from .logging import record_run,source_hash,command_output
from signrel.eval.frozen import frozen_validation,validate_evaluation

ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed,deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def dataset_asset_hash(rows):
    h = hashlib.sha256()
    paths = sorted({r[k] for r in rows for k in ("video_path","landmark_path") if r.get(k)})
    for path in paths:
        h.update(str(Path(path).resolve()).encode())
        h.update(sha256(path).encode())
    return h.hexdigest()


def train(config,resume=None,stop_after=None):
    dc,tc = config["data"],config["trainer"]
    rank,world = int(os.getenv("RANK","0")),int(os.getenv("WORLD_SIZE","1"))
    local_rank = int(os.getenv("LOCAL_RANK","0"))
    device_name = tc.get("device","cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{local_rank}" if device_name=="cuda" else device_name)
    if device.type=="cuda":
        torch.cuda.set_device(device)
    if world>1:
        torch.distributed.init_process_group("nccl" if device.type=="cuda" else "gloo")
    seed = config.get("seed",42)
    seed_everything(seed,tc.get("deterministic",False))
    rows = read_manifest(dc["manifest"])
    eval_paths = list(dc.get("heldout_manifests",[]))
    evaluation_cfg = config.get("evaluation",{})
    eval_every = int(tc.get("eval_every",0))
    if eval_every < 0:
        raise ValueError("eval_every must be nonnegative")
    if eval_every:
        eval_paths = sorted(set(eval_paths+validate_evaluation(evaluation_cfg)))
    if not eval_paths:
        raise ValueError("Declare heldout_manifests explicitly; leakage audit is mandatory")
    evaluation = [r for p in eval_paths for r in read_manifest(p)]
    audit = assert_disjoint(rows,evaluation,dc.get("audit_hash_files",False))
    manifest_hash = sha256(dc["manifest"])
    expected = {"world_size":world,"source_hash":source_hash(ROOT),
                "dataset_asset_hash":dataset_asset_hash(rows),
                "evaluation_manifest_hashes":{p:sha256(p) for p in eval_paths}}
    if eval_every:
        expected["evaluation_config_sha256"] = sha256(evaluation_cfg["config"])
        import yaml
        eval_config = yaml.safe_load(Path(evaluation_cfg["config"]).read_text())
        if eval_config.get("phonology_metadata"):
            expected["evaluation_phonology_metadata_sha256"] = sha256(eval_config["phonology_metadata"])
    model = SignRelJEPA(config).to(device)
    oc = config.get("optimizer",{})
    accum = tc.get("accumulate_steps",1)
    batch_size = dc.get("batch_size_per_gpu",2)
    effective_batch = batch_size*world*accum
    lr = oc.get("lr",1e-4)
    if oc.get("scale_lr",False):
        lr *= effective_batch/oc.get("reference_batch",256)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,
                                  weight_decay=oc.get("weight_decay",0.04),betas=tuple(oc.get("betas",[0.9,0.95])))
    max_steps = int(tc["max_steps"])
    warmup = int(max_steps*config.get("scheduler",{}).get("warmup_fraction",0.05))
    def lr_factor(step):
        if step<warmup:
            return (step+1)/max(warmup,1)
        return 0.5*(1+math.cos(math.pi*min((step-warmup)/max(max_steps-warmup,1),1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    precision = tc.get("precision","fp32")
    if precision not in {"bf16","fp16","fp32"}:
        raise ValueError("Unknown precision")
    if precision=="fp16" and device.type!="cuda":
        raise ValueError("fp16 training requires CUDA")
    if precision=="bf16" and device.type=="cpu":
        # oneDNN BF16 convolution backward is unavailable on this AVX2 host.
        # PyTorch reference kernels support BF16; retain actual mixed precision in CPU DDP tests.
        torch.backends.mkldnn.enabled = False
    scaler = torch.amp.GradScaler("cuda",enabled=precision=="fp16")
    start_step = 0
    resume_state = {}
    if resume:
        state = load_checkpoint(resume,model,optimizer,scheduler,scaler,config,manifest_hash,rank,expected)
        start_step = state["global_step"]
        resume_state = state
    # Student initialization must match across ranks; per-rank stochastic state after model creation.
    if not resume:
        seed_everything(seed+rank,tc.get("deterministic",False))
    ddp = DistributedDataParallel(model,device_ids=[local_rank] if device.type=="cuda" else None,
                                  find_unused_parameters=True) if world>1 else model
    dataset = SSLVideoDataset(dc["manifest"],num_frames=dc.get("num_frames",16),
                              sampling_stride=dc.get("sampling_stride",2),crop_size=dc.get("crop_size",224),
                              augment=dc.get("augment",True),require_landmarks=dc.get("require_landmarks",True))
    end_step = min(max_steps,stop_after) if stop_after is not None else max_steps
    sampler = ResumableBatchSampler(len(dataset),batch_size,seed,start_batch=start_step*accum,
                                    total_batches=end_step*accum,rank=rank,world_size=world)
    generator = torch.Generator().manual_seed(seed+rank)
    loader = DataLoader(SeededDataset(dataset,seed),batch_sampler=sampler,collate_fn=ssl_collate,
                        num_workers=dc.get("num_workers",0),generator=generator,pin_memory=device.type=="cuda")
    run_dir = Path(tc["run_dir"])
    if rank==0:
        if not resume and (run_dir/"metrics.jsonl").exists():
            raise FileExistsError("Run already exists; resume explicitly or choose another run_dir")
        if not resume:
            record_run(run_dir,config,dc["manifest"],model.student_encoder.upstream_init,ROOT)
        else:
            (run_dir/f"resume_{start_step}.json").write_text(json.dumps({"checkpoint":str(resume),"stop_after":stop_after},indent=2))
        (run_dir/"leakage.json").write_text(json.dumps(audit,indent=2))
        (run_dir/"input_contract.json").write_text(json.dumps(expected,indent=2))
        print(json.dumps({"effective_batch":effective_batch,"world_size":world,"device":str(device),
                          "precision":precision,"steps":max_steps,"start_step":start_step}),flush=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    begin = time.monotonic()
    totals = {}
    unhealthy = int(resume_state.get("collapse_alarm_count",0))
    best_evaluation = resume_state.get("best_evaluation")
    iterator = iter(loader)
    for step in range(start_step,end_step):
        step_start = time.monotonic()
        for micro in range(accum):
            batch = next(iterator)
            batch["video"] = batch["video"].to(device,non_blocking=True)
            batch["articulator_meta"] = {k:v.to(device,non_blocking=True) for k,v in batch["articulator_meta"].items()}
            sync = ddp.no_sync() if world>1 and micro<accum-1 else contextlib.nullcontext()
            dtype = torch.bfloat16 if precision=="bf16" else torch.float16
            with sync:
                with torch.autocast(device.type,dtype=dtype,enabled=precision!="fp32"):
                    output = ddp(batch)
                    loss = output["loss"]/accum
                scaler.scale(loss).backward()
            for k,v in output["metrics"].items():
                totals[k] = totals.get(k,0.)+v/accum
        for module in (model.teacher_encoder,model.teacher_relation_projector):
            if any(p.grad is not None or p.requires_grad for p in module.parameters()):
                raise AssertionError("Teacher gradient invariant violated")
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),tc.get("grad_clip_norm",1.),error_if_nonfinite=True)
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale()<before:
            raise FloatingPointError("AMP skipped optimizer step; run stopped before EMA/scheduler update")
        ec = config.get("ema",{})
        momentum = ema_momentum(step,max_steps,**ec)
        model.ema_update(momentum)
        applied_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        # Alarms are diagnostic; don't silently add objectives to rescue a failed relation target.
        health = tc.get("collapse_alarm",{})
        threshold = health.get("std_threshold",1e-5)
        small = totals.get("student_embedding_std",1)<threshold or totals.get("teacher_embedding_std",1)<threshold
        if config.get("loss",{}).get("relation_weight",0)>0:
            small = small or totals.get("teacher_target_std",1)<threshold
        small = small or totals.get("between_sample_cosine",0)>health.get("cosine_threshold",0.99999)
        unhealthy = unhealthy+1 if small else 0
        if unhealthy>=health.get("patience",100):
            raise RuntimeError("Sustained representation collapse alarm")
        totals.update(global_step=step+1,ema_momentum=momentum,grad_norm=float(grad),
                      learning_rate=applied_lr,next_learning_rate=optimizer.param_groups[0]["lr"],weight_decay=optimizer.param_groups[0]["weight_decay"],
                      amp_scaler=scaler.get_scale(),samples_per_second=effective_batch/(time.monotonic()-step_start),
                      gpu_memory_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else 0.,
                      elapsed_sec=time.monotonic()-begin,collapse_alarm_count=unhealthy)
        if world>1:
            keys = sorted(k for k in totals if k not in {"global_step","elapsed_sec"})
            data = torch.tensor([totals[k] for k in keys],device=device,dtype=torch.float64)
            torch.distributed.all_reduce(data)
            for k,v in zip(keys,(data/world).tolist()):
                totals[k] = v
        if rank==0:
            with (run_dir/"metrics.jsonl").open("a") as f:
                f.write(json.dumps(totals,allow_nan=False)+"\n")
            if (step+1)%tc.get("log_every",20)==0 or step==start_step:
                print(json.dumps(totals,allow_nan=False),flush=True)
        is_best = False
        if eval_every and ((step+1)%eval_every==0 or step+1==max_steps):
            # Every rank reaches the same boundary. Only rank 0 evaluates, and
            # broadcasts failures so other ranks cannot wait indefinitely.
            payload = [None]
            if world>1:
                torch.distributed.barrier()
            if rank==0:
                try:
                    result = frozen_validation(model.student_encoder,evaluation_cfg,dc,device,precision,run_dir,step+1)
                    value = result["selection_value"]
                    mode = evaluation_cfg.get("selection_mode","max")
                    is_best = best_evaluation is None or (value>best_evaluation["selection_value"] if mode=="max" else value<best_evaluation["selection_value"])
                    if is_best:
                        best_evaluation = result
                    payload[0] = {"result":result,"best_evaluation":best_evaluation,"is_best":is_best}
                    with (run_dir/"frozen_validation.jsonl").open("a") as f:
                        f.write(json.dumps({**result,"is_best":is_best},allow_nan=False)+"\n")
                except Exception as error:
                    payload[0] = {"error":f"{type(error).__name__}: {error}"}
            if world>1:
                torch.distributed.broadcast_object_list(payload,src=0)
            if "error" in payload[0]:
                raise RuntimeError("Periodic frozen evaluation failed: "+payload[0]["error"])
            best_evaluation,is_best = payload[0]["best_evaluation"],payload[0]["is_best"]
        if (step+1)%tc.get("checkpoint_every",2000)==0 or step+1==end_step or is_best:
            rngs = [rng_state()]
            if world>1:
                local = rng_state()
                rngs = [None]*world
                torch.distributed.all_gather_object(rngs,local)
            if rank==0:
                extra = {**expected,"git_commit":command_output(["git","rev-parse","HEAD"],ROOT).strip(),
                         "epoch":((step+1)*accum)//sampler.per_epoch,
                         "collapse_alarm_count":unhealthy,"best_evaluation":best_evaluation}
                save_checkpoint(run_dir/"checkpoints/latest.pt",model,optimizer,scheduler,scaler,step+1,
                                config,manifest_hash,rngs,extra)
                if is_best:
                    # latest.pt is atomically replaced, so a hard-linked best.pt
                    # retains its old inode when a later latest.pt is written.
                    import shutil
                    best_path = run_dir/"checkpoints/best.pt"
                    temporary = best_path.with_suffix(".pt.tmp")
                    temporary.unlink(missing_ok=True)
                    try:
                        os.link(run_dir/"checkpoints/latest.pt",temporary)
                    except OSError:
                        shutil.copyfile(run_dir/"checkpoints/latest.pt",temporary)
                    temporary.replace(best_path)
        totals = {}
    if world>1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return model
