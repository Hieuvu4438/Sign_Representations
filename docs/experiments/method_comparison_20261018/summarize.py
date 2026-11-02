"""Aggregate existing measurements only. No new training/evaluation or test access."""
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
rows=[];sources={}


def load(path):
    p=ROOT/path;sources[path]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())


def add(group,method,readout,budget,m,source,**notes):
    rows.append(dict(group=group,method=method,readout=readout,budget=budget,dataset='ASL Citizen',split='test',
                     gallery=40154,queries=32941,R1=m.get('R1'),R5=m.get('R5'),R10=m.get('R10'),MRR=m.get('MRR'),
                     DCG_x100=m.get('DCG_x100'),source=source,notes=notes))

s='method/phonojepa/runtime/eval/PJ_PLctx_FULL_160ep_dictionary/aslc/results.json'
r=load(s)['readouts']['_grp']
add('skeleton_raw','PLctx full160','grp','160 epochs / 200640 updates',dict(R1=r['Rec1']*100,R5=r['Rec5']*100,DCG_x100=r['DCG']*100),s)
s='method/phonojepa/runtime/plctx160_continuation/final_test/metrics.json';r=load(s)['metrics']
add('skeleton_raw','PLctx continuation','grp','selected global210640; branches searched separately',
    dict(R1=r['Rec@1']*100,R5=r['Rec@5']*100,R10=r['Rec@10']*100,MRR=r['MRR'],DCG_x100=r['DCG']*100),s)
s='method/phonopath_jepa/runtime/retrieval_e040_test/results.json';results=load(s)
assert results['status']=='complete' and results['query_n']==32941
for key,r in results['readouts'].items():
    m=r['metrics'];add('skeleton_raw' if key=='grp' else 'skeleton_postprocessed','PhonoPath-JEPA A1 epoch40',key,'40 epochs / 50160 updates',
                       dict(R1=m['R@1'],R5=m['R@5'],R10=m['R@10'],MRR=m['MRR'],DCG_x100=m['DCG']*100),s,
                       primary=(key=='grp+P'),pca=0 if key=='grp' else 768,encoder_always_uses_S_P_G=True)
s='docs/experiments/pose_jepa_results_20261015/official_test_report.json';r=load(s)['results']
for key in ('raw_test','white512_test'):
    m=r[key];add('skeleton_raw' if key=='raw_test' else 'skeleton_postprocessed','Pose-JEPA Hybrid',key,
       'run125482 updates; selected115000',dict(R1=m['R@1']*100,R5=m['R@5']*100,R10=m['R@10']*100,MRR=m['MRR'],DCG_x100=m['DCG']*100),s,
       archived_weights_deleted=True,valid_gallery=40126,valid_query=32937,query_denominator=32941)
s='method/asm_jepa/runs/final_s0/pretrain/reports/frozen_test.json';r=load(s);m=r['signrep_style']
add('skeleton_raw','ASM-JEPA','state_part_concat','val-selected checkpoint5000',
    dict(R1=m['Rec@1']*100,R5=m['Rec@5']*100,DCG_x100=m['DCG']*100),s,
    usable_gallery=40132,query_denominator=32941,failed_queries=6)
s='docs/proposal4/sign-ssl-research-agent/research/paper/tables.json';r=load(s)['aslc_test']
for key,name in [('R0','SignRep public weighted'),('R1','SignRep + bins2'),('R2','SignRep + chamfer20')]:
    m=r[key];add('rgb_signrep_pipeline',name,key,'external RGB-pretrained author encoder; no extra encoder training',
                dict(R1=m['R@1'],R5=m['R@5'],MRR=m['MRR']/100,DCG_x100=m['DCG']),s,
                archived_registered_measurement=True,set_matching=(key=='R2'))
s='docs/proposal4/sign-ssl-research-agent/research/CLAIM_LEDGER.md'
p=ROOT/s;sources[s]=hashlib.sha256(p.read_bytes()).hexdigest()
assert '57.28/57.31/57.31' in p.read_text()
add('rgb_signrep_pipeline','H008 set-to-vector distillation','single vector1024','label-free head; frozen external SignRep teacher',
    dict(R1=(57.28+57.31+57.31)/3),s,seeds_R1=[57.28,57.31,57.31],rounded_registered_seed_results=True)

with (OUT/'measurements.csv').open('w',newline='') as f:
    fields=list(rows[0]);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for row in rows:writer.writerow(row|{'notes':json.dumps(row['notes'],ensure_ascii=False)})
(OUT/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
(OUT/'measurements.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
lines=['# So sánh kết quả các method — 18/10/2026','',
       'Chỉ tổng hợp số đã đo trong repo, không chạy lại model. Metric chính: R@1 frozen dictionary retrieval trên official ASL Citizen test. Không trộn ASLC300/LOSO/val hoặc recognition accuracy vào bảng test.','',
       '**Skeleton tốt nhất đã đo: PLctx full160, R@1=16,6783%. Pipeline RGB/SignRep tốt nhất trong các kết quả đã đăng ký: chamfer20, R@1=58,42%. H008 single-vector khoảng57,30% trên3seed.**','',
       'Đây là xếp hạng theo task; input, pretrain, budget, feature extraction và readout khác nhau. Không suy ra nhân quả hoặc SOTA ngoài repo.','']
for group,title in [('skeleton_raw','Skeleton — raw readout, không PCA/whitening'),('skeleton_postprocessed','Skeleton — các readout có PCA/whitening'),('rgb_signrep_pipeline','RGB — các pipeline dựa trên SignRep')]:
    lines+=['## '+title,'','| Method | Readout | Budget | R@1 (%) | R@5 (%) | DCG×100 |','|---|---|---|---:|---:|---:|']
    for row in sorted((r for r in rows if r['group']==group),key=lambda r:r['R1'],reverse=True):
        def num(x):return '—' if x is None else f'{x:.3f}'
        lines.append(f"| {row['method']} | {row['readout']} | {row['budget']} | {num(row['R1'])} | {num(row['R5'])} | {num(row['DCG_x100'])} |")
    lines+=['']
lines+=['## Diễn giải','',
        '- PLctx continuation16,6328% không cải thiện test so với baseline16,6783%; chênh lệch0,0455điểm, một seed và chưa có kiểm định để gọi là khác biệt có ý nghĩa.',
        '- PhonoPath A1 chỉ có test epoch40. Readout chính grp+P8,5668%; grp11,3051% là readout phụ cố định đã báo cùng lần. Chưa có A0 ở40epoch nên không kết luận riêng tác dụng của P hoặc hiệu quả compute-matched.',
        '- PhonoPath full160 đang train; không dùng checkpoint đang train như một kết quả đã đo.',
        '- Pose-JEPA whitening5129,3167% là postprocessing khác raw5,7436%, không gộp thành kết quả raw. Archive giữ metric nhưng đã xóa weights/runtime.',
        '- ASM đã loại22gallery không dùng được;6query lỗi vẫn nằm trong mẫu số. Pose có28gallery/4query không dùng được; mẫu số test vẫn giữ32941. PLctx giữ zero vectors theo policy gốc. Có khác biệt failure/preprocessing giữa method.',
        '- SignRep+chamfer là set-to-set/window matching; H008 là learned single-vector head không dùng gloss. Đây là cải tiến pipeline/readout trên cùng encoderRGB ngoại dữ liệu, không phải skeleton encoder mới.',
        '- P4RGB sử dụng bảng kết quả đã đăng ký/claim ledger lưu trong docs; không tuyên bố rerun hoặc kiểm tra lại binary đã xóa.',
        '- PhonoJEPA PLint/PLboth, S1/S2, Graph-JEPA, SignRel-JEPA/SignRel-Temporal và SignCoord có các pilot/protocol riêng; không điền điểm pilot thành full-test. PLctx–Pose fusion đã dừng và xóa runtime17/10, chưa có accuracy fusion được xác minh để xếp hạng.',
        '- SignRep tái tạo trên ASLC (40ksteps) chỉ có proxy5462gallery/2731val, bestproxyR@1=1,8674%; không phải official fulltest.','',
        '## Nguồn','']
for s,h in sources.items():lines.append(f'- `{s}` — SHA256 `{h}`')
(OUT/'REPORT_VI.md').write_text('\n'.join(lines)+'\n')
print('Wrote',OUT,'rows',len(rows))
