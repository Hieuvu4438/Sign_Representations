import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def cfg():
    return {"seed":42,"model":{"backbone":{"provider":"tiny","dim":32,"depth":1,"heads":4,
                                             "patch":8,"tubelet":2},"relation_dim":16,
                                "predictor":{"width":32,"heads":4,"depth":1}},
            "masking":{"mixture":{"relation_break":1.},"span_fraction":0.5,"min_visible_ratio":0.25},
            "loss":{"generic_weight":0.,"relation_weight":1.,"articulator_weight":0.25},
            "trainer":{"run_dir":"unused","max_steps":8}}


@pytest.fixture
def batch():
    boxes = torch.tensor([[0.,0.5,0.25,1.],[0.75,0.5,1.,1.],
                          [0.25,0.,0.75,0.25],[0.25,0.25,0.75,1.]])
    return {"video":torch.randn(2,3,8,32,32),"articulator_meta":{
        "boxes":boxes[None,None].repeat(2,8,1,1),"confidence":torch.ones(2,8,4),
        "motion":torch.rand(2,8,4)},"sample_id":["a","b"]}
