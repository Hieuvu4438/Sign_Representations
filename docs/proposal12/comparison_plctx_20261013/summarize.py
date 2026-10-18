"""Summarize saved local measurements; no training or external-paper scores."""
from pathlib import Path
import csv,hashlib,json,math

REPO=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
source={}

def read(path):
    path=REPO/path;source[str(path.relative_to(REPO))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

models={};locations={}
for directory in ['method/obs_jepa/runtime/eval/models','method/phonojepa/runtime/eval/models',
                  'method/phonojepa_s1/runtime/eval/models','method/phonojepa_s2/runtime/eval/models']:
    for path in sorted((REPO/directory).glob('*.json')):
        if path.stem.endswith('_cls'):continue
        j=read(str(path.relative_to(REPO)))
        if not all(k in j for k in ('loso','aslc300')):continue
        if path.stem in models:
            assert models[path.stem]==j,f'Conflicting cached measurements {path.stem}'
            continue
        models[path.stem]=j;locations[path.stem]=str(path.relative_to(REPO))

rows=[]
for name,j in models.items():
    if 'retrieval_white512' not in j['aslc300']:continue
    budget={'pretraining_clips':'','optimizer_steps':'','batch':''}
    base=name.removesuffix('_grp')
    paths=[f'method/phonojepa/runtime/runs/{base}/manifest.json',
           f'method/phonojepa_s1/runtime/runs/{base}/manifest.json',
           f'method/phonojepa_s2/runtime/runs/{base}/manifest.json']
    for p in paths:
        if not (REPO/p).exists():continue
        m=read(p);c=m['config']
        budget={'pretraining_clips':m.get('n_clips',m.get('provenance',{}).get('n_clips','')),
                'optimizer_steps':m.get('steps_done',''),'batch':c.get('batch','')};break
    row={'model':name,**budget,'source_json':locations[name]}
    for prefix,metrics in [('ASLC300',j['aslc300']['retrieval_white512']),('LOSO',j['loso']['white512'])]:
        assert metrics['n']==(1500 if prefix=='ASLC300' else 10304)
        for k,label in [('DCG','DCG_x100'),('R@1','Rec1_pct'),('R@5','Rec5_pct')]:
            assert math.isfinite(metrics[k]);row[prefix+'_'+label]=metrics[k]*100
    row['ASLC300_LP_macro_pct']=100*j['aslc300']['linear_probe']['macro_top1']
    rows.append(row)
rows.sort(key=lambda r:r['ASLC300_Rec1_pct'],reverse=True)
with (OUT/'all_methods_white512.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

dictionary=REPO/'method/phonojepa/runtime/eval/PJ_PLctx_FULL_160ep_dictionary/comparison.csv'
source[str(dictionary.relative_to(REPO))]=hashlib.sha256(dictionary.read_bytes()).hexdigest()
with dictionary.open(encoding='utf-8-sig',newline='') as f:
    measured=[r for r in csv.DictReader(f) if r['source']!='published reference' and
              (r['method']=='PLctx_FULL_160ep_grp' or r['source']=='existing local measurement')]
with (OUT/'full_dictionary_measured.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(measured[0]));w.writeheader();w.writerows(measured)
for ds,query in [('aslc',32941),('wlasl',2879)]:
    prefix=f'method/phonojepa/runtime/eval/PJ_PLctx_FULL_160ep_dictionary/{ds}'
    protocol=read(prefix+'/protocol.json');result=read(prefix+'/results.json');ver=read(prefix+'/verification.json')
    assert ver['rank_to_metric_check']=='PASS' and ver['query_count']==query
    assert protocol['query']['n']==query and not protocol['pca_whitening']
    assert result['readouts']['_grp']['queries']==query
    r=next(r for r in measured if r['method']=='PLctx_FULL_160ep_grp' and r['dataset']==('ASL Citizen' if ds=='aslc' else 'WLASL2000'))
    for metric,column in [('DCG','DCG'),('Rec1','Rec1'),('Rec5','Rec5')]:
        assert math.isclose(100*result['readouts']['_grp'][metric],float(r[column]),abs_tol=1e-10)

full=read('method/phonojepa/runtime/eval/PJ_PLctx_FULL_160ep.json')
pair=full['paired']['PJ_PLctx_FULL_160ep_grp-PJ_PLctx_16k_grp']
selected=['JH3D','PJ_PLctx_16k_grp','PJ_PLctx_FULL_160ep_grp','PJ_PLctx_grp',
          'PJ_PLint_16k_grp','PJ_PLboth_grp','PJ_BMAE_grp','PJ_BMAMP_grp','S1_FULL_4k_grp',
          'JV3D','B13','S2_FULL_4k_grp','SR3D']
byname={r['model']:r for r in rows}
labels={'PJ_PLctx_grp':'PLctx 4k','PJ_PLctx_16k_grp':'PLctx 16k',
        'PJ_PLctx_FULL_160ep_grp':'PLctx full160epoch','PJ_PLint_16k_grp':'PLint 16k',
        'PJ_PLboth_grp':'PLboth 4k','PJ_BMAE_grp':'B-MAE 4k','PJ_BMAMP_grp':'B-MAMP 4k',
        'S1_FULL_4k_grp':'S1 4k','S2_FULL_4k_grp':'S2 4k'}
lines=['# PLctx16k / full160epoch và các method đã chạy','',
       'Ngày tổng hợp:2026-10-13. Chỉ đọc và tổng hợp các kết quả đã có; không chạy training/evaluation mới.',
       'DCG trong mọi bảng dùng **DCG×100**; Rec@K và LP dùng %. Không gộp protocol có/không whitening hoặc validation/test.',
       '', '## Ngân sách training','',
       '| Run | Video train | Steps | Batch | Lượt clip trung bình/video |','|---|---:|---:|---:|---:|',
       '| PLctx16k | 3.000 | 16.000 |32|170,67|',
       '| PLctxfull160epoch |40.154|200.640|32|159,90|','',
       'Full160 train scratch,seed0; không warm-start pilot. Video train tăng13,38×, số optimizer update tăng12,54×. Không thể quy khác biệt riêng cho số epoch; lượt tiếp xúc trung bình mỗi clip tương đương, và corpus mở rộng từ300 lên2731 gloss.',
       '', '## ASLC300: cùng evaluator white512','',
       'Gallery3.000 clip train/query1.500 clip validation,300 gloss. Encoder frozen; whitening fit gallery. Đây là protocol đã dùng trong các câu trả lời trước.',
       '', '| Method | DCG×100 | Rec@1 | Rec@5 | LP macro |','|---|---:|---:|---:|---:|']
for name in selected:
    r=byname[name]
    lines.append(f"| {labels.get(name,name)} | {r['ASLC300_DCG_x100']:.2f} | {r['ASLC300_Rec1_pct']:.2f} | {r['ASLC300_Rec5_pct']:.2f} | {r['ASLC300_LP_macro_pct']:.2f} |")
lines.extend(['','Bảng đầy đủ mọi primary/group readout đã có cùng protocol: [all_methods_white512.csv](all_methods_white512.csv). Không chọn giữa grp/CLS theo test; các biến thể CLS không đưa vào bảng này.',
    '', '## LOSO validation: cùng white512','',
    '10.304 clip validation,6 signer; gallery nằm ở các signer khác trong validation. PCA whitening không nhãn fit trên validation nên thiết lập này có tính transductive.',
    '', '| Method | DCG×100 | Rec@1 | Rec@5 |','|---|---:|---:|---:|'])
for name in ['PJ_PLctx_grp','PJ_PLctx_16k_grp','PJ_PLctx_FULL_160ep_grp','JH3D','JV3D','S1_FULL_4k_grp','S2_FULL_4k_grp']:
    r=byname[name];lines.append(f"| {labels.get(name,name)} | {r['LOSO_DCG_x100']:.2f} | {r['LOSO_Rec1_pct']:.2f} | {r['LOSO_Rec5_pct']:.2f} |")
lines.extend(['', '## Full160 −16k: paired signer-cluster bootstrap','',
             '| Metric | Chênh lệch (điểm%) | CI95 |','|---|---:|---|'])
for key,value in pair.items():
    lo,hi=value['ci95'];lines.append(f"| {key} | {100*value['diff']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
lines.extend(['','Full160 tăng LOSO Rec@1 và LP; ASLC300 retrieval giảm nhẹ với CI95 chứa0. CI đo biến thiên theo signer của một cặp model, không đo biến thiên qua seed. Chỉ một seed và nhiều phép so sánh chưa điều chỉnh.',
              '', '## Frozen dictionary: official test, không whitening','',
              'Raw window mean → video L2 → cosine → max gallery score theo gloss. PLctx dùng skeleton64frame,stride2,avg; RGB references dùng16frame,stride2; weighted dùng activity head. Input, pretrained weights và data budget khác nhau.',
              '', '| Dataset | Method/pooling | DCG×100 | Rec@1 | Rec@5 |','|---|---|---:|---:|---:|'])
for r in measured:
    lines.append(f"| {r['dataset']} | {r['method']} / {r['pooling']} | {float(r['DCG']):.2f} | {float(r['Rec1']):.2f} | {float(r['Rec5']):.2f} |")
lines.extend(['','ASL Citizen test:40.154 gallery/32.941 query,2731 gloss. WLASL2000:14.296 gallery/2.879 query,2000 gloss. Chưa có PLctx16k trên hai full-test protocol trong các artifact đã kiểm tra; không dùng điểm pilot validation thay thế.',
    '', 'Full160 thấp hơn SignRep public trong cả hai tập. Trên WLASL, full160 hơn JH3D về Rec@1 nhưng thấp hơn DCG/Rec@5, nên không vượt toàn diện.',
    '', '## Kết luận từ kết quả đã đo','',
    '- Trong các skeleton primary/group run đã thống kê, PLctx16k có ASLC300 Rec@1 cao nhất; full160 không cải thiện metric này.',
    '- Full160 có representation tốt hơn trên LOSO và linear probe; LP68,87% cao hơn JH3D61,73% trong bảng cùng evaluator, nhưng retrieval thấp hơn JH3D.',
    '- Raw dictionary full-test còn yếu. Các điểm SignRep public dùng pretrainRGB bên ngoài và pooling khác phải được ghi như reference khác điều kiện.',
    '- Geometric probes full160:handshapeR²=.3246,relations=.1711,body=.0320,movement=−.0295,nonmanual=−.2599. Signer accuracy93,73%; chuyển động/nonmanual chưa được probe tuyến tính giải mã tốt.',
    '', '## Nguồn và kiểm tra','',
    '- Các cache JSON trùng tên trong nhiều runtime được đối chiếu nội dung, không cộng thành nhiều run.',
    '- `sources.json` ghim SHA256 của các kết quả/manifest đã đọc; script `summarize.py` tạo lại bảng.',
    '- Kết quả full dictionary đối chiếu results/protocol/verification với CSV; không đưa paperTable4 chưa tái chạy vào bảng measured.',
    '- ASLC300 raw avg trong bảng proposal11 dùng protocol/window pooling khác white512; SignRep_public84,56/72,20/90,80 ở bảng đó không được chèn vào bảng white512.',
    '- Không chạy model mới và không chọn hyperparameter/checkpoint theo test.', ''])
(OUT/'REPORT_VI.md').write_text('\n'.join(lines))
(OUT/'sources.json').write_text(json.dumps({'status':'VERIFIED_SAVED_RESULTS','date':'2026-10-13',
     'metric_scale':'DCGx100, recall percentages','n_primary_models':len(rows),'sha256':source},indent=2)+'\n')
print(json.dumps({'models':len(rows),'report':str(OUT/'REPORT_VI.md'),'status':'PASS'},indent=2))
