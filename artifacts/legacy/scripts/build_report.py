"""Build an evidence-linked status report; incomplete tasks cannot imply completion."""
import argparse
import csv
import json
import time
from pathlib import Path

from _common import ROOT
from signrepr.io import atomic_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--gates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.registry.open() as stream:
        runs = list(csv.DictReader(stream))
    gates = json.loads(args.gates.read_text())
    tasks = json.loads((ROOT / 'state/tasks.json').read_text())['tasks']
    inventory = json.loads((ROOT / 'evidence/bootstrap_inventory.json').read_text())
    coverage = json.loads((ROOT / 'reports/manifest_coverage.json').read_text())
    cost = json.loads((ROOT / 'reports/signrep_pilot_cost.json').read_text())
    checks = json.loads((ROOT / 'reports/integrity_checks.json').read_text())
    lines = ['# Trạng thái thực thi nghiên cứu Sign Representations', '',
             '## Material Passport', '',
             'Artifact: `reports/material_passport.json`. Mode: run. Raw datasets: read-only. Runtime: Codex, một agent.', '',
             '**IN_PROGRESS — chưa có method novelty được kiểm chứng; goal vẫn active.**', '',
             'Báo cáo này phản ánh bằng chứng hiện có, không phải final synthesis hoàn tất kế hoạch.', '',
             'Cập nhật UTC: ' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), '',
             'G0: **' + gates['G0']['decision'] + '** trong phạm vi nhãn đã kiểm tra. Baseline lexical và diagnostic nội bộ đã có kết quả; xem phạm vi bên dưới.', '',
             '## Dữ liệu và leakage', '',
             '| Dataset | Annotation rows | File hiện có | Train/val/test |', '|---|---:|---:|---|']
    for name, entry in coverage.items():
        lines.append(f"| {name} | {entry['annotation_rows']} | {entry['video_or_frames_present']} | {entry['split_counts']} |")
    lines += ['', 'Nguồn: `reports/manifest_coverage.json`, `reports/manifest_validation.json`, `reports/split_audit*.json`.', '',
              'ASL Citizen có 52 participant IDs trong CSV chính thức và không có signer overlap giữa các split.',
              'MS-ASL có 5 nguồn YouTube xuất hiện ở nhiều split. PHOENIX có 494 recording-name group candidates xuất hiện ở nhiều split; đây là grouping theo tên nguồn, chưa phải xác nhận video trùng.',
              'WLASL có 21.095 source recording groups chưa resolve; How2Sign test có 2.349 groups chưa resolve. Các điều này giới hạn transfer claims.',
              'Decode coverage toàn bộ corpus và pretraining overlap chưa được chứng minh.', '',
              '## Backbone và pilot', '',
              'SignRep upstream `06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2`, W01 SHA256 `f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b`.',
              'Strict load: không có missing/unexpected keys; pretrained projection head hiện diện; 55.644.329 tổng tham số.',
              '8 mẫu train smoke đều PASS, feature dimension 768, RGB/transform/timestamps/masks hợp lệ, sai khác eval lặp tối đa 0.',
              'Nguồn: `features/signrep_pilot/extraction_report.json`, `features/signrep_pilot/load_audit.json`, `logs/signrep_smoke_attempt4.log`.',
              f"Pilot single pass: {cost['measured_single_pass_seconds']:.3f} giây cho {cost['pilot_windows']} windows; peak allocation {cost['peak_gpu_memory_bytes']} bytes.",
              f"Cohort forecast {cost['cohort_hours_extrapolation']:.2f} giờ; full ASL Citizen forecast {cost['full_asl_citizen_hours_extrapolation']:.2f} giờ. Đây là extrapolation từ 8 train clips, chưa phải đo full-duration distribution.",
              'SHuBERT dùng project venv Python 3.10; strict load encoder và hai DINO checkpoints đã PASS trên CPU. Raw-video pilot xem trạng thái bên dưới. Public translator đã có CPU numerical smoke; xem report bên dưới, không gọi là paper checkpoint.', '',
              '## Protocol và runs', '',
              'Protocol common_v1 đã khóa trước test: 200 lớp chọn từ train, 2.400 train / 726 val / 1.600 test; primary metric macro recall.',
              'Baseline này phục vụ kiểm tra pipeline/lexical retention; không phải full benchmark reproduction hoặc kết quả grammar.',
              'B02 linear và B03 shallow temporal head chỉ học readout. Các head báo tham số thực tế; khác biệt head không được diễn giải thành novelty của encoder.', '',
              '| Run | Status | Metrics | Failure reason |', '|---|---|---|---|']
    for run in runs:
        lines.append(f"| {run['run_id']} | {run['status']} | {run['metrics_path'] or 'NA — chưa có'} | {run['failure_reason'] or ''} |")
    common_status = ROOT / 'features/shubert_native_common_v1/status.json'
    if common_status.exists():
        common = json.loads(common_status.read_text())
        prep = json.loads((ROOT / 'features/shubert_native_common_v1_preprocessing/status.json').read_text())
        control = json.loads((ROOT / 'runs/fair_common_v1/status.json').read_text())
        lines += ['',
                  f"Native lexical preprocessing: {prep['status']}, {prep['passed']}/{prep['samples']}; GPU0 features: {common['status']}, {common['success']}/{common['samples']} ({100*common['success']/common['samples']:.1f}%).",
                  f"Equal-grid lexical suite: {control['status']}. Số clip xử lý chưa phải số clip đủ điều kiện pooling; coverage cuối và class support còn chờ audit.",
                  'Tiến độ là snapshot từ status files; không suy diễn metric native từ cache một phần.']
    native_report = ROOT / 'features/shubert_native_pilot_gpu_attempt4/report.json'
    native = json.loads(native_report.read_text()) if native_report.exists() else None
    lines += ['', 'SHuBERT raw-video GPU pilot: ' + (native['status'] if native else 'PENDING — xem state/tasks.json và logs/shubert_native_pilot_gpu_attempt4.log'),
              'Native author output là nhánh FFN cuối (`layer_results[-1][-1]`), khác encoder output sau residual. Raw-video scripts giữ FPS đầu vào; paper pretraining bỏ mỗi frame thứ hai và downstream recipe còn TODO, nên không claim exact paper reproduction.',
              'Cả encoder và DINO sử dụng weights tác giả đã khớp SHA256, không có missing/unexpected keys. Cache cuối lưu [12,T,768], last-layer và layer-average; mask loại frame thiếu stream khỏi pooling/loss.',
              '7/8 native pilot clips có frame đủ bốn streams để pooling; clip PRETTY có zero valid pooling frames và phải exclude ở future probes. 8/8 extraction PASS không có nghĩa 8/8 usable readouts. Nguồn: `reports/shubert_native_missingness_audit.json`.',
              'Hai GPU smoke attempts SIGSEGV được giữ lại. Import-order audit tái hiện lỗi decord-before-CUDA; CUDA-first workaround và full raw-video forward đã PASS. Nguồn: `reports/shubert_cuda_import_audit.json`.', '',
              '## Local lexical results', '',
              '| Baseline | Macro recall mean | Recording bootstrap 95% CI | Training seed SD | Seeds |',
              '|---|---:|---|---:|---|']
    for baseline in ['B00', 'B01', 'B02', 'B03']:
        stats_path = ROOT / f'reports/common_v1_{baseline}_bootstrap.json'
        if stats_path.exists():
            summary = json.loads(stats_path.read_text())
            metric = summary['metrics']['macro_recall']
            sd = metric['training_seed_sd']
            lines.append(f"| {baseline} | {metric['point']:.4f} | [{metric['ci95'][0]:.4f}, {metric['ci95'][1]:.4f}] | {sd if sd is not None else 'NA — deterministic'} | {summary['ordered_seeds']} |")
        else:
            completed = []
            for seed in ([0] if baseline in ['B00', 'B01'] else [0, 1, 2]):
                measured = ROOT / f'runs/common_v1_{baseline}_signrep_seed{seed}/metrics.json'
                if measured.exists():
                    completed.append((seed, json.loads(measured.read_text())['macro_recall']))
            lines.append(f"| {baseline} | NA — pending complete statistics | NA | NA | available point estimates: {completed} |")
    primary = ROOT / 'reports/common_v1_primary_bootstrap.json'
    if primary.exists():
        metric = json.loads(primary.read_text())['metrics']['macro_recall']
        lines += ['', f"Primary delta B03−B02: {metric['delta']:.4f}, paired recording CI [{metric['delta_ci95'][0]:.4f}, {metric['delta_ci95'][1]:.4f}]."]
    lines += ['', 'CIs conditional trên heads đã train và cohort 200 lớp; không bao gồm training randomness hoặc independent signer-population uncertainty. Không chọn best seed làm headline.',
              'B03 có 6 learning-rate/width candidates; B02 có 3 learning-rate candidates trong protocol đã khóa. Head parameters được match trong 0.1%, nhưng tuning search khác nhau là confound; delta chỉ là diagnostic, không chứng minh superiority của encoder/method.',
              'Active GPU compute hours chưa đo trực tiếp. Registry lưu wall_hours và một-GPU allocation wall-hours bao gồm decode/loading/I/O; không gọi chúng là GPU compute-hours.',
              'Error audit dùng 30 IDs cố định từ union lỗi B03, kiểm tra kỹ thuật tự động. Review lexical correctness/ambiguity và video identity bởi ASL expert còn PENDING.', '']
    diagnostic_path = ROOT / 'runs/ncslgr_diagnostic_v1/summary.json'
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        lines += ['', '## NCSLGR recorded-function diagnostic', '',
                  'Historical public XML mirrors có functional gold tiers; 222 utterances / 444 paired views. Train107, val27, test88; 2/1/1 source signers tách biệt.',
                  'Target: loại functional NEG/WH/YN được ghi dương trong nguồn. Không suy diễn affirmative/absence, không dùng gold core intervals làm input pooling.',
                  'Mỗi head linear768→3 có2.307 tham số, cùng100epochs và3LR candidates; val chọn head, test không chọn cấu hình. Trung bình seeds0/1/2.', '',
                  '| Readout | Macro recall mean | Conditional sequence bootstrap95%CI | Training seed SD |',
                  '|---|---:|---|---:|']
        for name, item in diagnostic['results'].items():
            if 'bootstrap' in item:
                m = item['bootstrap']['metrics']['macro_recall']
                lines.append(f"| {name} | {m['point']:.4f} | {m['ci95']} | {m['training_seed_sd']:.4f} |")
        delta = diagnostic['paired_vs_body']['metrics']['macro_recall']
        lines += ['', f"Predefined paired-minus-body delta: {delta['delta']:.4f}, CI {delta['delta_ci95']}; chưa thấy improvement được hỗ trợ. Prior macro recall0.3333.",
                  'Val chỉ2YN/4NEG, test chỉ một signer; camera color/resolution khác nhau. Không diễn giải view comparison thành causal facial-cue isolation hoặc encoder mất ngữ pháp.',
                  'Source sequence và exact decoded-video overlap bằng0; bốn XML aggregate groups vẫn cross-split. Historical release chưa xác nhận byte-identical với bản2011.',
                  'Nguồn: `reports/ncslgr_annotation_audit/report.json`, `reports/ncslgr_media_audit.json`, `reports/ncslgr_input_preparation.json`, `runs/ncslgr_diagnostic_v1/summary.json`.']
        sensitivity = ROOT / 'reports/ncslgr_diagnostic_audit.json'
        if sensitivity.exists():
            audit = json.loads(sensitivity.read_text())
            subset = audit['analyses']['exclude_shared_xml_test_subset']
            values = {k: v['metrics']['macro_recall']['point'] for k,v in subset['readouts'].items()}
            lines += [f"Post-hoc sensitivity bỏ test XML groups có mặt ngoài test: {subset['samples']} utterances, class counts {subset['class_counts']}, macro recall {values}. Giữ nguyên trained heads; không retune hoặc thay primary test.",
                      'Audit prediction coverage/gold/metadata đã PASS; 30 fixed error IDs chưa có expert linguistic review. Nguồn: `reports/ncslgr_diagnostic_audit.json`.']
    interval_path = ROOT / 'runs/ncslgr_intervals_v1/summary.json'
    if interval_path.exists():
        interval = json.loads(interval_path.read_text())
        delta = interval['body_temporal_vs_global']
        lines += ['', '## Recorded functional-core interval diagnostic', '',
                  'Protocol mới khóa trước interval metrics, sau khi đã xem classification test cùng cohort: exploratory, chưa phải xác nhận độc lập.',
                  '222 utterances dùng classification; 179 single-event utterances dùng endpoint loss/evaluation (89train/25val/65test). 43multi-event utterances không bị coi là absent.',
                  'Bốn readouts body/face × global/temporal, mỗi head6.915activeparameters; cùng3LR ×100epochs ×3seeds. Functional gold là supervision, không vào forward inputs.', '',
                  '| Readout | Macro class-aware interval IoU | Conditional95%CI |', '|---|---:|---|']
        for name, result in interval['results'].items():
            metric = result['primary_bootstrap']
            lines.append(f"| {name} | {metric['point']:.4f} | {metric['ci95']} |")
        lines += ['', f"Primary body temporal−global: {delta['delta']:.4f}, CI {delta['delta_ci95']}; chưa có bằng chứng improvement được hỗ trợ.",
                  'Boundary position distributions lưu theo timestamp nguồn; không phải xác suất presence/absence và chưa calibrated. Source functional-core khác complete grammatical scope.',
                  'Nguồn: `configs/protocol_ncslgr_intervals_v1.json`, `runs/ncslgr_intervals_v1/summary.json`.']
        audit_path = ROOT / 'reports/ncslgr_interval_audit.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            m = audit['posthoc_same_classifier_duration_prior']['readouts']['body_temporal']
            lines += [f"Machine audit: {audit['boundary_distribution_files_verified']}boundaryfiles và12predictiontables khớp originalclock/targets; zeroASLexpertreviews.",
                      f"Post-hoc same-classifier duration-prior sensitivity: priorIoU {m['reference_point']:.4f}, temporalIoU {m['point']:.4f}, delta {m['delta']:.4f}, CI {m['delta_ci95']}. Prior endpoints chỉ fit từ train; không thay primary hoặc chọn head bằng sensitivity này.",
                      'Head hiện tại chưa vượt prior đơn giản trong sensitivity; không suy diễn encoder thiếu ngữ pháp hoặc thêm attention là novelty.']
    translator_path = ROOT / 'features/shubert_translation_smoke_attempt1/report.json'
    if translator_path.exists():
        translator = json.loads(translator_path.read_text())
        lines += ['', '## Public-demo translator local smoke', '',
                  'SHuBERT-author-demo-11625: ' + translator['status'] + '; CPU, strictload không missing/unexpectedkeys.',
                  f"Nguồn caption: {translator['source_caption']}; meanNLL {translator['mean_nll']:.6f}, sumNLL {translator['sum_nll']:.6f}, {translator['token_count']}ByT5tokens cóEOS.",
                  f"Padding=-100 giữ logits token hợp lệ trong maxdelta {translator['padding_agreement_max_logit_difference']:.3g}. Generation49token budget: {translator['generation_text']}",
                  'Generation khác caption nguồn. Đây là numerical smoke một clip train-signer, không phải semantic pair accuracy, external test hoặc grammar comprehension.',
                  'Embedded demo encoder khác213/226W02tensors; không dùng kết quả decoder để kết luận W02 mất thông tin. Runtime torch2.1.1/transformers4.30.2 là tested compatibility, không bitwise author reproduction.',
                  'Nguồn: `features/shubert_translation_smoke_attempt1/report.json`, `reports/shubert_demo_source_audit.json`, `provenance/W08.json`.']
    calibration_path = ROOT / 'runs/ncslgr_calibration_v1/summary.json'
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        primary = calibration['primary']
        lines += ['', '## Categorical probability sensitivity', '',
                  'Temperature chọn bằng validation NLL trên sáu giá trị khóa trước; dùng lại chín head và normalization train đã lưu. Test predictions/probabilities gốc được tái tạo và đối chiếu.',
                  f"Body mean NLL: {primary['reference_point']:.6f} → {primary['point']:.6f}; delta {primary['delta']:.6f}, conditional95%CI {primary['delta_ci95']}.",
                  'Cả chín head chọn temperature8, ở biên trên grid; đây là sensitivity trong grid hữu hạn, chưa chứng minh xác suất đã calibrated. Class argmax không thay đổi.',
                  'ECE dùng năm equal-width reliability bins với counts lưu đầy đủ. Validation chỉ27utterances/một signer và đã dùng chọn LR trước đó; test đã được quan sát. Không calibration boundary/presence hoặc method novelty.',
                  'Nguồn: `configs/protocol_ncslgr_calibration_v1.json`, `runs/ncslgr_calibration_v1/summary.json`.']
    native_ncslgr = ROOT / 'features/shubert_native_ncslgr_v1/status.json'
    if native_ncslgr.exists():
        native = json.loads(native_ncslgr.read_text())
        prep = json.loads((ROOT / 'features/shubert_native_ncslgr_v1_preprocessing/status.json').read_text())
        control = json.loads((ROOT / 'runs/fair_ncslgr_v1/status.json').read_text())
        lines += ['', '## Native NCSLGR controls', '',
                  f"CPU preprocessing: {prep['status']}, {prep['passed']}/{prep['samples']}; native CPU features: {native['status']}, {native['success']}/{native['samples']}.",
                  f"Equal-budget function suite: {control['status']}. SignRep, native last FFN và fixed12-layer average; 18heads cùng6915parameters,3seeds,3LR,100epochs.",
                  'Đối chứng duration prior chỉ học interval fractions từ train. Chung eligible IDs; ghi mọi exclusions và dừng nếu lớp mất support. Không đánh giá partial native test hoặc thêm GPU worker.',
                  'Protocol: `configs/protocol_fair_ncslgr_v1.json`; pilot/cache provenance và timestamp erratum: `reports/native_ncslgr_cpu_pilot_audit.json`.']
        terminal = ROOT / 'runs/fair_ncslgr_v1/summary.json'
        if terminal.exists():
            measured = json.loads(terminal.read_text())
            coverage = json.loads((terminal.parent / 'coverage.json').read_text())
            primary = measured['primary']
            lines += ['',
                      f"Strict common population: {len(coverage['eligible_ids'])}/222utterances; {len(coverage['excluded'])}exclusions. Train65/val27/test83; interval test61. Giữ mọi lớp nhưng population khác diagnostic222gốc.",
                      '| Readout | Interval IoU | Train-duration prior IoU | Classification macro recall |',
                      '|---|---:|---:|---:|']
            for kind, value in measured['results'].items():
                lines.append(f"| {kind} | {value['interval_bootstrap']['point']:.4f} | {value['train_duration_prior_bootstrap']['point']:.4f} | {value['classification_bootstrap']['metrics']['macro_recall']['point']:.4f} |")
            lines += [f"Primary SHuBERT last temporal−global: {primary['delta']:.4f}, conditional95%CI {primary['delta_ci95']}. Temporal readout không vượt global; cả sáu endpoint readouts thua prior học từ train.",
                      'Native cues/context khác SignRep; một test signer và selection validation nhỏ. Kết quả không chứng minh encoder thiếu grammar hoặc một adaptation mới.']
    author_pool = ROOT / 'runs/ncslgr_author_pool_v1/summary.json'
    if author_pool.exists():
        measured = json.loads(author_pool.read_text())
        primary = measured['primary']['metrics']['macro_recall']
        lines += ['', '## Author-imputed pooling sensitivity', '',
                  'Protocol riêng giữ toàn bộ native contextual tokens trong utterance support, kể cả carry-forward/black crop inputs. Mask quan sát không đổi; không gọi imputed là observed. Full222utterances107train/27val/88test,9heads2307parameters.',
                  'Đã khóa sau khi strict native result files tồn tại; lựa chọn dựa trên train missingness. Không claim blind/independent evaluation.',
                  '| Frozen backbone | Macro recall | Conditional95%CI |', '|---|---:|---|']
        for kind, value in measured['results'].items():
            stats = value['bootstrap']['metrics']['macro_recall']
            lines.append(f"| {kind} | {stats['point']:.4f} | {stats['ci95']} |")
        lines += [f"Primary last−SignRep: {primary['delta']:.4f}, CI {primary['delta_ci95']}. Fixed12-layer average là secondary; không chọn best-test-layer làm primary.",
                  'Audit `reports/native_ncslgr_controls_audit.json`:222source clocks/caches,27savedheads và747boundaryfiles đã đối chiếu; zeroexpertreviews.']
    disk_stop = ROOT / 'evidence/native_common_disk_stop_v1/manifest.json'
    if disk_stop.exists():
        state = json.loads((ROOT / 'features/shubert_native_common_v1/status.json').read_text())
        if state['status'] == 'FAIL':
            lines += ['', 'Native lexical extraction terminal FAIL do reserved disk limit tại1235/4726features,1237preparedclips. Equal-grid lexical waiter cũng terminal FAIL trước fitting; không có native lexical metrics.',
                      'Lần dừng và logs được giữ ở `evidence/native_common_disk_stop_v1/`. Reviewed recovery cần khoảng60GiBfree; giữ21GiBguard và caches, không giảm reserve hoặc xóa dữ liệu khác.']
    recovery_status = ROOT / 'state/native_common_disk_recovery_v1/status.json'
    if recovery_status.exists():
        state = json.loads(recovery_status.read_text())
        lines += ['', f"Reviewed recovery hiện tại: {state['status']}. CPU preprocessing phải hoàn tất trước soleGPU0feature consumer, rồi same locked equal-grid suite trong `runs/fair_common_v1_recovery1`.",
                  'Các terminal failures ở trên thuộc attempt trước; recovery giữ nguyên attempts/logs, cache identities và disk guard.']
    semantic_readiness = ROOT / 'reports/semantic_pairs_readiness_v1/status.json'
    if semantic_readiness.exists():
        state = json.loads(semantic_readiness.read_text())
        lines += ['', f"Semantic-pair CLI readiness: {state['status']}. `scripts/run_semantic_pairs.py` đã triển khai; candidate ordering/ties, recording donors và utterance aggregation đã unit-test.",
                  'Dry-run dừng trước model loading vì thiếu reviewed canonical gold/media/cue caches. Không có external semantic accuracy hoặc full-model pair integration đã được kiểm chứng.']
    semantic_backend = ROOT / 'features/semantic_pair_backend_smoke_attempt1/report.json'
    if semantic_backend.exists():
        measured = json.loads(semantic_backend.read_text())
        lines += ['', f"Semantic backend numerical integration: {measured['status']}; {measured['sampled_frames']} frame, {measured['elapsed_seconds']:.2f} giây CPU.",
                  'Tái tạo NLL caption nguồn 1,474203467/token và tie credit 0,5 khi hai candidate giống nhau. Blank streams chạy hợp lệ. Một clip từ signer train; chưa có distinct semantic-pair gold hoặc external accuracy.']
    grounding_readiness = ROOT / 'reports/grounding_readiness_v1/status.json'
    if grounding_readiness.exists():
        state = json.loads(grounding_readiness.read_text())
        lines += ['', f"Q02 CLI readiness: {state['status']}. Subsequence-DTW, timestamp gốc, NMS, khớp interval một-một, macro/pooled AP, false positives/minute, boundary error và recording bootstrap đã có kiểm thử tổng hợp.",
                  'Cấu hình thật dừng trước search vì chưa có mapping/occurrence/absence/grouping xác minh. Fixture tổng hợp không cung cấp gold hoặc metric grounding thực tế. Xem `reports/track_requirements_audit_v1.md`.']
    branch=ROOT/'features/ncslgr_branch_ablation_v3/status.json'
    if branch.exists():
        measured=json.loads(branch.read_text())
        lines += ['',f"A03 cue-intervention extraction: {measured['status']}, {measured['success']}/{measured['samples']} utterance trong tập chung đã khóa; CPU-only, tám điều kiện.",
                  'Giữ nguyên video/crop/context/observation mask và head budget. Zeroing là distribution-shift diagnostic, không causal branch importance. V2 dừng ở absolute cross-cache guard; v3 dùng baseline all tính lại trong cùng lượt và kiểm tra repeat với ngưỡng1e-5. Không chứng nhận tương đương số học với cache cũ.',
                  'Các lỗi/import, smoke và numerical diagnosis được lưu trong evidence và registry. Chưa có metric ablation khi extraction chưa đầy đủ.']
    lines += ['', '## Grammar, grounding và novelty', '',
              'Người dùng xác nhận chưa có annotation gold bổ sung ngoài datasets đã liệt kê.',
              'ASL-MTP CSV có 1.275 rows, 48 raw video IDs và 938 URLs; raw video IDs không phải canonical utterance IDs.',
              'Pilot 3 public DAI pages đã resolve metadata cho 5 pairs, nhưng video/linguistic review chưa thực hiện; chức năng download yêu cầu login.',
              'ASL-MTP vẫn là external test; không train/tune trên gold candidates hoặc published NLL/BLEURT tables.',
              'A-G1: ' + gates['A-G1']['status'] + '; A-G2: ' + gates['A-G2']['status'] + '.',
              'Adaptation: ' + gates['adaptation']['status'] + '. External A-G0 và grounding B-G0/B-G1/B-G2 vẫn gated.',
              'Nguồn: `reports/gate_decisions.json`, `reports/asl_mtp_metadata_audit.json`, `reports/asl_mtp_resolution_pilot.json`, `reports/annotation_request.csv`.',
              'P35/P36/P37 đã nhận dạng chức năng ngữ pháp, temporal relations, core/onset/offset và timing rules. P26 dùng frozen motion encoder cùng readout cho non-manual lexical distinctions. Prior art cũng đã bao gồm non-manual cue analysis, phonological subspaces/minimal pairs, headshake detection, query-by-example spotting, DTW, retrieval và QA. Không claim first từ các task/component này.',
              'Câu hỏi còn kiểm chứng: encoder information-access so với decoder utilization, và functional scope-sensitive adaptation có lexical retention. Local recorded-function diagnostic đã có; chưa có method gain hoặc external semantic confirmation.',
              'Nguồn: `evidence/literature_matrix.csv`, `reports/novelty_matrix.csv`.', '',
              '## Kiểm tra và giới hạn', '',
              f"{checks['test_count']} integrity tests: {checks['status']}. Coverage gồm grouping/view leakage, path escape, timestamps, padding/masks, matched-capacity heads, cache invalidation/corruption, ambiguous DAI mapping và incomplete PDF detection.",
              'P04 explicit-version PDF ban đầu chỉ trả 4 MiB; đã quarantine. Canonical PDF có đúng v3 trong tài liệu nhưng checksum khác ETag của explicit-version HEAD; giữ FAIL, không đổi expected hash để bỏ lỗi. Claims P04 dùng HTML v3 chính thức, không dùng page anchors của PDF lỗi.',
              '37 nguồn đã có mức đọc khai báo: 17 core, 17 expansion trong kế hoạch và ba prior functional non-manual cũ. Mức đọc gồm phần liên quan của bài hoặc trang tài nguyên chính thức; chưa phải toàn văn mọi paper hoặc acquisition/PDF integrity hoàn tất.',
              'Môi trường SignRep dùng project venv kế thừa các package đã có; đã khóa phiên bản quan sát và không cài vào conda base. Không claim bitwise reproduction.', '',
              '## Task states', '', '| Task | Status | Lý do/phần còn lại |', '|---|---|---|']
    for task in tasks:
        lines.append(f"| {task['task_id']} | {task['status']} | {task['reason'] or ''} |")
    review_manifest = ROOT / 'reports/ncslgr_expert_review_v1/manifest.json'
    if review_manifest.exists():
        review = json.loads(review_manifest.read_text())
        lines += ['', '## Expert review và quyết định method', '',
                  f"Đã chuẩn bị hồ sơ {review['case_count']} trường hợp; expert reviews hoàn thành: {review['expert_reviews_completed']}. Form và hướng dẫn đề xuất: `reports/ncslgr_expert_review_v1/`.",
                  'Quyết định interim: `reports/method_decision_v1.md`. Chưa có representation adaptation hoặc novelty được kiểm chứng.']
    lines += ['', '## Tiếp tục và tái lập', '',
              'Xem `REPRODUCE.md`, `state/tasks.json`, `state/common_suite.json` và `runs/experiment_registry.csv`.',
              'Revalidate PID/job handles trước khi tiếp tục; không restart extraction đang live.',
              'Việc tiếp theo: reviewed disk recovery cho native lexical cohort và equal-grid controls trong output riêng; native NCSLGR và function controls đã hoàn tất. Còn semantic pairs độc lập, adjudication cho đầy đủ grammatical scope/absence và kiểm chứng cơ chế trước method adaptation.',
              'Claim ledger: `reports/claim_ledger.csv`; chi phí native: `reports/shubert_native_pilot_cost.json`; resource plan: `reports/resource_plan.md`.',
              'Completion audit chưa đạt. Không task/claim blocked nào được chuyển thành PASS chỉ để báo hoàn thành.', '']
    atomic_text(args.output, '\n'.join(lines))
    print(json.dumps({'report': str(args.output), 'status': 'IN_PROGRESS', 'novel_method_verified': False}))


if __name__ == '__main__':
    main()
