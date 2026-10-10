import torch
from signrel.models.signrel_jepa import SignRelJEPA


def test_multiscale_backprop_and_queries(cfg,batch):
    cfg["model"]["temporal_scales"]=[{"name":"short","kernel":1,"weight":1.},
                                      {"name":"medium","kernel":2,"weight":0.5},
                                      {"name":"long","kernel":4,"weight":0.25}]
    model=SignRelJEPA(cfg)
    output=model(batch)
    output["loss"].backward()
    assert output["target_relation_0"].shape[1]==4
    assert output["target_relation_1"].shape[1]==3
    assert output["target_relation_2"].shape[1]==1
    for i in range(3):
        assert not output[f"target_relation_{i}"].requires_grad
        assert output[f"query_valid_{i}"].any()
    assert all(p.grad is None for p in model.teacher_relation_projector.parameters())
