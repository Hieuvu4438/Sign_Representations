"""Session-wide measured resource accounting, shared by all sequential jobs."""
import json
from pathlib import Path
from .utils import atomic_json

LEDGER = Path('artifacts/budget_ledger.json')


def remaining(resource):
    if not LEDGER.exists(): return float('inf')
    value = json.loads(LEDGER.read_text())
    caps = value['global_caps']
    limits = {'download_bytes':caps['download_gb']*1e9,
              'new_cache_bytes':caps['new_cache_gb']*1e9,
              'gpu_seconds':caps['gpu_hours']*3600,
              'wall_training_seconds':caps['wall_training_minutes']*60}
    return limits[resource]-value['used'].get(resource,0)


def charge(event_id,kind,**resources):
    if not LEDGER.exists(): return
    value = json.loads(LEDGER.read_text())
    prior = next((r for r in value['runs'] if r['event_id']==event_id),None)
    previous = prior.get('resources',{}) if prior else {}
    if kind not in ('training','download'):
        # Profile/debug/re-extraction calls consume additional time, while each
        # cache key charges its maximum stored footprint rather than double it.
        resources={name:(previous.get(name,0)+amount if name.endswith('seconds') else max(previous.get(name,0),amount))
                   for name,amount in resources.items()}
    for name,amount in resources.items():
        if amount<previous.get(name,0): raise ValueError('Budget usage cannot decrease')
        value['used'][name] = value['used'].get(name,0)+amount-previous.get(name,0)
    event = {'event_id':event_id,'kind':kind,'resources':resources}
    if prior: value['runs'][value['runs'].index(prior)] = event
    else: value['runs'].append(event)
    atomic_json(LEDGER,value)
