import copy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch


def contract_hash(config):
    c = copy.deepcopy(config)
    trainer = c.get("trainer",{})
    for k in ("run_dir","log_every","checkpoint_every","keep_checkpoints"):
        trainer.pop(k,None)
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path,model,optimizer,scheduler,scaler,step,config,manifest_hash,
                    rng_states=None,extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    state = {"format_version":1,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
             "lr_scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
             "global_step":step,"epoch":(extra or {}).get("epoch",0),
             "rng_state":rng_states or [rng_state()],"config":config,"config_hash":contract_hash(config),
             "dataset_manifest_hash":manifest_hash,"upstream_init":model.student_encoder.upstream_init,
             **(extra or {})}
    # Named module views satisfy the explicit component contract without duplicating tensor storage.
    for name in ("student_encoder","teacher_encoder","student_relation_projector","teacher_relation_projector",
                 "relation_predictor","articulator_predictor","generic_predictor"):
        state[name] = getattr(model,name).state_dict()
    tmp = path.with_suffix(path.suffix+".tmp")
    torch.save(state,tmp)
    tmp.replace(path)


def load_checkpoint(path,model,optimizer,scheduler,scaler,config,manifest_hash,rank=0,expected=None):
    # Locally generated run checkpoints include Python/NumPy RNG objects.
    state = torch.load(path,map_location="cpu",weights_only=False)
    if state["config_hash"]!=contract_hash(config):
        raise AssertionError("Resume config mismatch")
    if state["dataset_manifest_hash"]!=manifest_hash:
        raise AssertionError("Resume dataset manifest mismatch")
    for key,value in (expected or {}).items():
        if state.get(key)!=value:
            raise AssertionError(f"Resume {key} mismatch")
    if rank>=len(state["rng_state"]):
        raise AssertionError("Resume distributed world size mismatch")
    model.load_state_dict(state["model"],strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["lr_scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng_state"][rank])
    return state
