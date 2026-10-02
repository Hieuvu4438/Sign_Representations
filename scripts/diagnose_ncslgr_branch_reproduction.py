"""Read-only numerical diagnosis of the failed training-clip reproduction guard."""
import json
import time
from pathlib import Path

from _common import ROOT
from extract_shubert_pilot import NativeSHuBERT, load_native_dino, dino_features, torch, np
from signrepr.io import read_jsonl, sha256, write_json


def main():
    import yaml
    output=ROOT/'reports/ncslgr_branch_reproduction_diagnostic_v1.json'
    if output.exists():
        raise ValueError('Preserve numerical diagnosis attempt')
    sid='ncslgr:ncslgr10p:11:body'
    row=next(r for _,r in read_jsonl(ROOT/'data/manifests/ncslgr_native_body_v1.jsonl') if r['sample_id']==sid)
    if row['official_split']!='train':
        raise ValueError('Diagnosis restricted to failed training clip')
    item=next(r for r in json.loads((ROOT/'features/shubert_native_ncslgr_v1_preprocessing/report.json').read_text())['results'] if r['sample_id']==sid)
    entry=next(r for _,r in read_jsonl(ROOT/'features/shubert_native_ncslgr_v1/index.jsonl') if r['sample_id']==sid)
    assert sha256(entry['shard'])==entry['shard_sha256']
    assert all(sha256(text)==item['stream_sha256'][k] for k,text in item['streams'].items())
    config=yaml.safe_load((ROOT/'configs/shubert.yaml').read_text());started=time.perf_counter()
    encoder=NativeSHuBERT(ROOT,config,device='cpu');face,fa=load_native_dino(ROOT,config,'W03',device='cpu');hand,ha=load_native_dino(ROOT,config,'W04',device='cpu')
    def inputs(threads):
        torch.set_num_threads(threads)
        return {k:np.load(text,allow_pickle=False).astype(np.float32) if k=='body_posture' else dino_features(face if k=='face' else hand,Path(text)) for k,text in item['streams'].items()}
    a=inputs(4);b=inputs(4)
    stream_repeat={k:float(np.abs(a[k]-b[k]).max()) for k in a}
    torch.set_num_threads(4);x=encoder.forward(a);repeated=encoder.forward(b)
    torch.set_num_threads(1);single_same_inputs=encoder.forward(a)
    c=inputs(1);single_all=encoder.forward(c)
    with np.load(entry['shard'],allow_pickle=False) as cached:
        reference=cached['embeddings'].copy();keep=cached['stream_observed_mask'].all(1)
    def compare(value):
        old=reference.astype(np.float64);new=value.astype(np.float64)
        result=dict(max_abs_difference=float(np.abs(old-new).max()),rmse=float(np.sqrt(np.mean((old-new)**2))),
                    allclose_atol1e_5_rtol1e_5=bool(np.allclose(new,old,atol=1e-5,rtol=1e-5)))
        if keep.any():
            v=old[keep].mean(0);w=new[keep].mean(0);v/=np.linalg.norm(v);w/=np.linalg.norm(w)
            result.update(observed_pool_cosine=float(v@w),observed_pool_max_abs_difference=float(np.abs(v-w).max()))
        return result
    report=dict(status='PASS_NUMERICAL_DIAGNOSIS_NO_SCIENTIFIC_METRICS',source_split='train',sample_id=sid,
        source_sha256=item['source_sha256'],reference_shard_sha256=entry['shard_sha256'],frames=len(reference),
        torch=torch.__version__,numpy=np.__version__,mkldnn_enabled=torch.backends.mkldnn.enabled,
        stream_repeat_4_threads_max_abs_difference=stream_repeat,
        encoder_repeat_4_threads_max_abs_difference=float(np.abs(x-repeated).max()),
        four_thread_reproduction=compare(x),encoder_one_thread_same_inputs=compare(single_same_inputs),
        all_one_thread_reproduction=compare(single_all),
        dino_four_vs_one_thread_max_abs_difference={k:float(np.abs(a[k]-c[k]).max()) for k in a},
        source_sha256_map={p:sha256(ROOT/p) for p in ['scripts/diagnose_ncslgr_branch_reproduction.py','scripts/extract_shubert_pilot.py','src/signrepr/shubert.py']},
        elapsed_seconds=time.perf_counter()-started,limits=['One failed training clip only; no val/test labels or grammar metrics examined.',
        'Original absolute-tolerance failure preserved. Diagnostic alone does not authorize widening locked tolerance or establish source/checkpoint mismatch.'])
    write_json(output,report);print(json.dumps(report))


if __name__=='__main__':main()
