"""Self-contained official ASL Citizen comparison from preserved measurements."""
from pathlib import Path
import csv,hashlib,json,shutil,statistics
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;BUNDLE=OUT/'source_results';BUNDLE.mkdir(exist_ok=True)
ARCHIVE=ROOT/'artifacts/phonopath_jepa_results_20261019';sources={};rows=[]

def source(path,name):
    p=ROOT/path;target=BUNDLE/name
    if p.exists():
        if p.resolve()!=target.resolve():shutil.copy2(p,target)
    elif not target.exists():raise FileNotFoundError(p)
    data=target.read_bytes();sources[name]=dict(original=path,sha256=hashlib.sha256(data).hexdigest())
    return json.loads(data) if target.suffix=='.json' else data.decode('utf-8')

def add(group,name,readout,budget,m,src,primary=True,**notes):
    rows.append(dict(group=group,method=name,readout=readout,primary=primary,budget=budget,dataset='ASL Citizen',split='test',gallery=40154,queries=32941,
                     R1=m.get('R1'),R5=m.get('R5'),R10=m.get('R10'),MRR=m.get('MRR'),DCG_x100=m.get('DCG_x100'),source_bundle=src,notes=notes))

p='method/phonojepa/runtime/eval/PJ_PLctx_FULL_160ep_dictionary/aslc/results.json';s='PLctx160_test.json';r=source(p,s)['readouts']['_grp']
add('skeleton_raw','PLctx full160','grp','160ep/200640updates',dict(R1=100*r['Rec1'],R5=100*r['Rec5'],DCG_x100=100*r['DCG']),s)
s='PLctx_continuation_test.json';r=source('method/phonojepa/runtime/plctx160_continuation/final_test/metrics.json',s)['metrics']
add('skeleton_raw','PLctx continuation','grp','selected210640;twoLRbranches searched',dict(R1=100*r['Rec@1'],R5=100*r['Rec@5'],R10=100*r['Rec@10'],MRR=r['MRR'],DCG_x100=100*r['DCG']),s)
phono={}
for epoch in (40,160):
    s=f'PhonoPath_epoch{epoch:03d}_test.json';p=f'artifacts/phonopath_jepa_results_20261019/results/epoch{epoch:03d}_test/results.json'
    result=source(p,s);assert result['status']=='complete' and result['query_n']==32941;phono[epoch]=result
    for key,r in result['readouts'].items():
        m=r['metrics'];add('skeleton_raw' if key=='grp' else 'skeleton_postprocessed',f'PhonoPath-JEPA A1 epoch{epoch}',key,
            f'{epoch}ep/{epoch*1254}updates',dict(R1=m['R@1'],R5=m['R@5'],R10=m['R@10'],MRR=m['MRR'],DCG_x100=100*m['DCG']),s,
            primary=(key=='grp+P'),PCA=0 if key=='grp' else 768,weights_archived=False)
s='Pose_JEPA_test.json';r=source('docs/experiments/pose_jepa_results_20261015/official_test_report.json',s)['results']
for key in ('raw_test','white512_test'):
    m=r[key];add('skeleton_raw' if key=='raw_test' else 'skeleton_postprocessed','Pose-JEPA Hybrid',key,'run125482;valselected115000',
        dict(R1=100*m['R@1'],R5=100*m['R@5'],R10=100*m['R@10'],MRR=m['MRR'],DCG_x100=100*m['DCG']),s,primary=(key=='raw_test'),usable_gallery=40126,query_denominator=32941)
s='ASM_JEPA_frozen_test.json';r=source('method/asm_jepa/runs/final_s0/pretrain/reports/frozen_test.json',s);m=r['signrep_style']
add('skeleton_raw','ASM-JEPA','state_part_concat','valselected5000',dict(R1=100*m['Rec@1'],R5=100*m['Rec@5'],DCG_x100=100*m['DCG']),s,usable_gallery=40132,failed_queries=6)
s='SignRep_readouts_registered.json';r=source('docs/proposal4/sign-ssl-research-agent/research/paper/tables.json',s)['aslc_test']
for key,name in [('R0','SignRep public weighted'),('R1','SignRep + bins2'),('R2','SignRep + chamfer20')]:
    m=r[key];add('RGB_SignRep_pipeline',name,key,'externalRGBpretrain;frozenencoder',dict(R1=m['R@1'],R5=m['R@5'],MRR=m['MRR']/100,DCG_x100=m['DCG']),s,registered_archived_measurement=True,set_matching=(key=='R2'))
s='P4_claim_ledger.md';text=source('docs/proposal4/sign-ssl-research-agent/research/CLAIM_LEDGER.md',s);assert '57.28/57.31/57.31' in text
add('RGB_SignRep_pipeline','H008 set-to-vector','single_vector1024','label-freehead;frozenSignRep',dict(R1=statistics.mean([57.28,57.31,57.31])),s,registered_rounded_seeds=[57.28,57.31,57.31])

for row in rows:
    assert row['split']=='test' and 0<=row['R1']<=100
with (OUT/'official_test_retrieval.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows([r|dict(notes=json.dumps(r['notes'])) for r in rows])
(OUT/'official_test_retrieval.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
recognition=[]
for name,run,regime in [('ASM-JEPA linear','linear','frozen linear probe'),('ASM-JEPA fine-tune','finetune','supervised fine-tune'),('ASM scratch','scratch','supervised scratch')]:
    key=f'ASM_{run}_recognition_test.json';m=source(f'method/asm_jepa/runs/final_s0/{run}/reports/test.json',key)['metrics']
    recognition.append(dict(method=name,regime=regime,top1=100*m['top1_instance'],top5=100*m['top5_instance'],source_bundle=key))
m=source('docs/experiments/pose_jepa_results_20261015/official_test_report.json','Pose_JEPA_test.json')['results']['linear_probe']['test']
recognition.append(dict(method='Pose-JEPA linear',regime='frozen linear probe',top1=100*m['instance_top1'],top5=100*m['instance_top5'],source_bundle='Pose_JEPA_test.json'))
(OUT/'recognition_test.json').write_text(json.dumps(recognition,indent=2)+'\n')
(OUT/'sources.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False)+'\n')

fmt=lambda v:'—' if v is None else f'{v:.3f}'
lines=['# Các method sau train và official test ASL Citizen — 19/10/2026','',
       'Tổng hợp số đo đã lưu, không chạy model hay chấm lại test. Galleryofficialtrain40.154 / testqueries32.941 /2.731gloss; R@K và DCG×100 là%. MRR giữ0–1. Không trộn pilot/val/recognition vào bảng retrieval.','',
       '**PLctx full160 giữ kết quả skeleton tốt nhất: R@1=16,678%. PhonoPath A1 full160 không vượt baseline: readout chính grp+P6,928%; rawgrp11,511%. Trong các pipelineRGB đã có kết quả đăng ký, SignRep+Chamfer58,42% cao nhất; H008singlevector khoảng57,30%.**','']
lines+=['## Readout chính của từng run skeleton','','| Method | Readout | Budget | Test R@1 (%) | R@5 (%) | DCG×100 |','|---|---|---|---:|---:|---:|']
for row in sorted([r for r in rows if r['group'].startswith('skeleton') and r['primary']],key=lambda r:r['R1'],reverse=True):
    lines.append(f"| {row['method']} | {row['readout']} | {row['budget']} | {fmt(row['R1'])} | {fmt(row['R5'])} | {fmt(row['DCG_x100'])} |")
for group,title in [('skeleton_raw','Cùng raw readout, không PCA/whitening'),('skeleton_postprocessed','Readout có PCA/whitening — báo riêng'),('RGB_SignRep_pipeline','RGB / SignRep — khác modality và external pretrain')]:
    lines+=['','## '+title,'','| Method | Readout | Test R@1 (%) | R@5 (%) | R@10 (%) | DCG×100 |','|---|---|---:|---:|---:|---:|']
    for row in sorted([r for r in rows if r['group']==group],key=lambda r:r['R1'],reverse=True):
        lines.append(f"| {row['method']} | {row['readout']} | {fmt(row['R1'])} | {fmt(row['R5'])} | {fmt(row['R10'])} | {fmt(row['DCG_x100'])} |")
lines+=['','## PhonoPath: epoch40 →160','','| Readout | Epoch40 R@1 (%) | Epoch160 R@1 (%) | Delta(points) |','|---|---:|---:|---:|']
for key in ('grp','grp+P','mw','ord2'):
    a=phono[40]['readouts'][key]['metrics']['R@1'];b=phono[160]['readouts'][key]['metrics']['R@1'];lines.append(f'| {key} | {a:.3f} | {b:.3f} | {b-a:+.3f} |')
lines+=['','## Recognition official test — tác vụ khác retrieval','','| Method | Regime | Top1 (%) | Top5 (%) |','|---|---|---:|---:|']
for row in sorted(recognition,key=lambda r:(r['regime'], -r['top1'])):lines.append(f"| {row['method']} | {row['regime']} | {row['top1']:.3f} | {row['top5']:.3f} |")
lines+=['','Không xếp chung fine-tune có nhãn với frozen linear probe. PLctx68,87% LP là ASLC300, không phải official2731gloss test; PhonoPath chưa có recognition official test nên không điền điểm smoke.','',
        '## Phạm vi và kết luận','',
        '- PLctx160 so với continuation16,633%: chênh0,046điểm R@1; một seed, chưa có kiểm định để gọi khác biệt có ý nghĩa.',
        '- PhonoPath grp full160 thấp hơn PLctx grp khoảng5,167điểm R@1. Cùng tổngbudget160ep, nhưng preprocessing/masking/kiến trúc khác; không quy riêng cho signature nếu chưa có A0/A2 cùng code.',
        '- PhonoPath grp+P giảm1,639điểm từ40→160; grp chỉ tăng0,206điểm. Modeltrainloss giảm nhưng retrieval không tăng tương ứng. Readout chính được giữ grp+P, không đổi winner theo test.',
        '- Mọi điểm cùng officialsplit nhưng không phải mọi preprocessing/failurepolicy giống hệt. Pose có40126usablegallery/4failedquery; ASM40132usablegallery/6failedquery, querydenominator giữ32941. PLctx giữzero vectors. PCA/whitening được ghi riêng.',
        '- SignRepRGB dùng pretrain bên ngoài; Chamfer là setmatching khác singlevector. P4/H008 là số đã đăng ký còn trong docs, không tuyên bố đã rerun weights/cache đã xóa.',
        '- S1/S2, PLint/PLboth, Graph-JEPA, SignRel-JEPA, SignCoord và SignPathLab có pilot/protocol khác hoặc chưa có fullofficialtest được xác minh. Không gán số pilot thành test. PLctx–Pose fusion đã dừng, không có kết quả fusionfulltest để xếp hạng.',
        '- SignRep tái tạo trên ASLC chỉ có proxy5462gallery/2731val, không phải officialfulltest.','',
        '## Lưu trữ','',
        'Kết quả/ranks/logs/config của PhonoPath đã được archive tại `artifacts/phonopath_jepa_results_20261019`. Runtime/checkpoint/cache nặng được dọn theo yêu cầu; code giữ lại. Bảng này có sourcebundles riêng và SHA256, nên vẫn tạo lại được sau cleanup.','']
(OUT/'REPORT_VI.md').write_text('\n'.join(lines)+'\n')
print('Official-test retrieval records:',len(rows),'recognition records:',len(recognition));print(OUT/'REPORT_VI.md')
