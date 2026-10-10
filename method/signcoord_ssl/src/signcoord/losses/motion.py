from signcoord.targets.motion import interval_velocity
from signcoord.targets.schema import PART_SLICES
from .masked import masked_huber


def velocity_loss(pred, target, times, reliability, stats):
    pv, rel = interval_velocity(pred, times, reliability)
    tv, _ = interval_velocity(target, times, reliability)
    total, logs = pred.sum()*0, {}
    for part, s in PART_SLICES.items():
        scale = stats[part]["velocity_scale"]
        loss, logs[part] = masked_huber(pv[..., s, :]/scale, tv[..., s, :]/scale, rel[..., s])
        total = total + 0.25*loss
    return total, logs
