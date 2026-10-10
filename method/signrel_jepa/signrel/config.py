from pathlib import Path
import copy
import yaml


def merge(base,override):
    result=copy.deepcopy(base)
    for k,v in override.items():
        if isinstance(v,dict) and isinstance(result.get(k),dict) and not (
            "provider" in v and v["provider"]!=result[k].get("provider")):
            result[k]=merge(result[k],v)
        else:
            result[k]=copy.deepcopy(v)
    return result


def load_config(path,seen=None):
    path=Path(path).resolve()
    seen=set() if seen is None else seen
    if path in seen:
        raise ValueError("Cyclic config inheritance")
    seen.add(path)
    config=yaml.safe_load(path.read_text())
    if not isinstance(config,dict):
        raise ValueError("Config must be a mapping")
    parent=config.pop("extends",None)
    if parent:
        config=merge(load_config(path.parent/parent,seen),config)
    return config
