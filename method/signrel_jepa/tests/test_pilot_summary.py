
from signrel.eval.pilot_summary import summarize

NAMES = ["A_generic","B_articulator","C_relation","D_combined"]


def values(value):
    return {k:value for k in ("Recall@1","Recall@5","mAP_instance","phonology_macro_f1","signer_accuracy")}


def test_completed_baseline_cannot_produce_negative_incomplete_verdict():
    grouped = {"A_generic":{s:values(.1) for s in (42,43,44)},"C_relation":{42:values(.05)}}
    report = summarize(grouped,[42,43,44],NAMES,audit_verified=True)
    assert report["decision"]["status"] == "insufficient_evidence"
    assert len(report["decision"]["missing"]) == 8


def test_beating_generic_only_is_not_relation_superiority():
    grouped = {name:{s:values(v) for s in (42,43,44)} for name,v in zip(NAMES,[.1,.3,.2,.2])}
    report = summarize(grouped,[42,43,44],NAMES,audit_verified=True)
    assert report["decision"]["status"] == "relation_benefit_not_supported_by_this_pilot"
    assert report["decision"]["comparisons"]["C_relation_vs_A_generic"]["consistent_two_metric_pilot_benefit"]
    assert not report["decision"]["comparisons"]["C_relation_vs_B_articulator"]["consistent_two_metric_pilot_benefit"]


def test_unaudited_and_test_results_cannot_authorize_phase2():
    grouped = {name:{s:values(v) for s in (42,43,44)} for name,v in zip(NAMES,[.1,.2,.3,.4])}
    report = summarize(grouped,[42,43,44],NAMES)
    assert report["decision"]["status"] == "insufficient_evidence"
    report = summarize(grouped,[42,43,44],NAMES,split="test",audit_verified=True)
    assert report["decision"]["status"] == "promising_pilot_requires_broader_confirmation"
    assert not report["decision"]["phase2_allowed"]
