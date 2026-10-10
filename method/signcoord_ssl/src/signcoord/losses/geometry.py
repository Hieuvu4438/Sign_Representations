from signcoord.targets.geometry import local_hands
from signcoord.targets.schema import PART_SLICES
from .masked import masked_huber


def geometry_loss(pred_global, target_global, reliability, stats, config=None):
    config=config or {"global_part_weights":dict.fromkeys(PART_SLICES,.25),
                      "local_total_weight":.5,"local_hand_weights":{"L":.5,"R":.5},"huber_delta":1.}
    losses, logs = {}, {}
    for part, s in PART_SLICES.items():
        mean = pred_global.new_tensor(stats[part]["mean"])
        scale = stats[part]["scale"]
        losses[part], logs["global_"+part] = masked_huber((pred_global[..., s, :]-mean)/scale,
                                                      (target_global[..., s, :]-mean)/scale, reliability[..., s],config["huber_delta"])
    local_pred, local_rel = local_hands(pred_global, reliability)
    local_target, _ = local_hands(target_global, reliability)
    local = pred_global.sum()*0
    for part in ("L", "R"):
        term, logs["local_"+part] = masked_huber(local_pred[part]/stats[part]["scale"],
                                              local_target[part]/stats[part]["scale"], local_rel[part],config["huber_delta"])
        local = local + config["local_hand_weights"][part]*term
    return sum(config["global_part_weights"][part]*loss for part,loss in losses.items()) + config["local_total_weight"]*local, logs
