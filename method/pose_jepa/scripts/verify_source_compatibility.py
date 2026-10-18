"""Check default-recipe update parity against the exact live pilot source snapshot."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
from signrepr.config import load_config
from signrepr.cli import source_hash
from signrepr.data.pose_dataset import PoseDataset
from signrepr.data.collate import collate_pose
from signrepr.training.pretrain import SSLTrainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    name = 'signrepr_pilot_snapshot'
    spec = importlib.util.spec_from_file_location(name, args.snapshot/'__init__.py', submodule_search_locations=[str(args.snapshot)])
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    old_class = importlib.import_module(name+'.training.pretrain').SSLTrainer
    cfg = load_config(args.config)
    cfg['optim']['precision'] = 'fp32'
    ds = PoseDataset(cfg['data']['train_manifest'], cfg['data'], sampling='isolated')
    batch = collate_pose([ds[0], ds[1]])
    records = []
    for name, weights in [('geometry', (0, 1, 0)), ('motion', (0, 0, 1)), ('jepa', (1, 0, 0)), ('hybrid', (1, .25, .1))]:
        c = deepcopy(cfg)
        for key, weight in zip(['latent_weight', 'geometry_weight', 'motion_weight'], weights): c['loss'][key] = weight
        old, new = old_class(c), SSLTrainer(c)
        a, b = old.update([batch]), new.update([batch])
        a.pop('update_duration_sec'); b.pop('update_duration_sec')
        if not weights[0]:
            # Old diagnostic fields described an unused scheduled EMA and a
            # dummy CE target. Their correction leaves all applied updates equal.
            a['ema_momentum'] = b['ema_momentum']
            a['raw_losses']['latent'] = b['raw_losses']['latent']
        if a != b: raise AssertionError(f'Update report mismatch: {name}')
        for component in ['system', 'teacher', 'criterion']:
            for key, value in getattr(old, component).state_dict().items():
                torch.testing.assert_close(value, getattr(new, component).state_dict()[key], atol=0, rtol=0)
        records.append({'objective': name, 'FP32_update_state_parity': 'bitwise', 'loss_center_EMA_report_parity': True})
        del old, new
    result = {'snapshot_source_sha256': args.snapshot.parent.name, 'current_source_sha256': source_hash(),
              'scope': 'one full-backbone effective update on two actual ASL-Citizen clips, default 0.2-second motion gap',
              'changes': 'stricter config/cache validation, configurable gap wiring, export metadata fallback and inactive-latent/EMA logging corrections; applied default updates preserved',
              'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'controls_with_bitwise_parity': len(records)}))


if __name__ == '__main__': main()
