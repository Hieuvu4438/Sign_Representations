"""Reconcile experiment states against reports and live process commands."""
import csv
import io
import json
from pathlib import Path

from _common import ROOT
from signrepr.io import atomic_text, read_jsonl, sha256

FIELDS = ['run_id', 'status', 'protocol_version', 'config_sha256', 'code_commit', 'checkpoint_sha256',
          'dataset_manifest_sha256', 'train_recording_count', 'validation_recording_count', 'test_recording_count',
          'trainable_parameters', 'gpu_hours', 'wall_hours', 'gpu_allocation_wall_hours', 'resource_measurement_note',
          'implementation_sha256', 'scientific_scope', 'coverage', 'metrics_path', 'failure_reason', 'pid', 'log_path']


def live(pid, expected):
    """Return None when this PID namespace cannot establish host liveness.

    A missing host PID in an isolated tool namespace is an observation limit,
    not a terminal experiment failure. Reports remain the terminal authority.
    """
    try:
        return expected.encode() in (Path('/proc') / str(pid) / 'cmdline').read_bytes()
    except (FileNotFoundError, PermissionError):
        return None


def main():
    rows = []
    for attempt, reason in [(1, 'Upstream constructor contains integer products; initial literal parser rejected them'),
                            (2, 'Inherited timm 0.4.5 lacks required Mlp; installed timm 1.0.20 in project venv'),
                            (3, 'Untracked Python bytecode falsely triggered source-change guard; tracked source remains unchanged')]:
        row = dict.fromkeys(FIELDS)
        row.update(run_id='signrep_smoke_attempt' + str(attempt), status='FAIL', failure_reason=reason,
                   log_path='logs/signrep_smoke_attempt' + str(attempt) + ('_FAIL.log' if attempt == 1 else '.log'))
        rows.append(row)
    for run_id, folder in [('signrep_smoke_attempt4', 'signrep_pilot'), ('common_v1_extraction', 'signrep_common_v1'),
                           ('ncslgr_signrep_smoke_v1', 'signrep_ncslgr_smoke_v1'),
                           ('ncslgr_signrep_extraction_v1', 'signrep_ncslgr_diagnostic_v1')]:
        path = ROOT / 'features' / folder
        report = path / 'extraction_report.json'
        row = dict.fromkeys(FIELDS)
        logs = {'signrep_pilot': 'logs/signrep_smoke_attempt4.log',
                'signrep_common_v1': 'logs/signrep_common_v1_extraction.log',
                'signrep_ncslgr_smoke_v1': 'logs/signrep_ncslgr_smoke_v1.log',
                'signrep_ncslgr_diagnostic_v1': 'logs/signrep_ncslgr_diagnostic_v1.log'}
        row.update(run_id=run_id, log_path=logs[folder])
        if report.exists():
            measured = json.loads(report.read_text())
            row.update(status=measured['status'], config_sha256=measured['config_sha256'],
                       dataset_manifest_sha256=measured['manifest_sha256'], coverage=measured['success'] / measured['samples'],
                       wall_hours=measured['elapsed_seconds'] / 3600,
                       gpu_allocation_wall_hours=measured['elapsed_seconds'] / 3600,
                       resource_measurement_note='One allocated GPU; wall time includes model loading, decoding and I/O. Active GPU compute hours unmeasured.',
                       implementation_sha256=measured['implementation_sha256'],
                       code_commit=measured['load_audit']['upstream_commit'], checkpoint_sha256=measured['load_audit']['checkpoint_sha256'],
                       trainable_parameters=0, metrics_path=str(report.relative_to(ROOT)))
        else:
            reports = list(path.glob('manifest_*.json'))
            state = json.loads(reports[0].read_text()) if reports else {}
            pid = state.get('pid')
            observed = live(pid, 'scripts/extract_features.py') if pid else False
            row.update(status='RUNNING' if observed is True else 'UNKNOWN' if observed is None else 'FAIL', pid=pid,
                       config_sha256=state.get('config_sha256'), dataset_manifest_sha256=state.get('manifest_sha256'),
                       failure_reason=None if observed is True else 'PID visibility unavailable; inspect report/log and worker lock' if observed is None else 'No matching extraction PID and no terminal report')
        rows.append(row)
    diagnostic = ROOT / 'runs/ncslgr_diagnostic_v1/summary.json'
    if diagnostic.exists():
        measured = json.loads(diagnostic.read_text())
        locked = ROOT / 'data/external/ncslgr/diagnostic_v1/protocol.lock.json'
        protocol_diag = json.loads(locked.read_text())
        row = dict.fromkeys(FIELDS)
        row.update(run_id='ncslgr_frozen_readout_suite_v1', status=measured['status'],
                   protocol_version='ncslgr_diagnostic_v1', config_sha256=sha256(locked),
                   dataset_manifest_sha256=protocol_diag['manifest_sha256'],
                   train_recording_count=measured['train'], validation_recording_count=measured['val'],
                   test_recording_count=measured['test'], trainable_parameters=2307,
                   gpu_hours=0, gpu_allocation_wall_hours=0, wall_hours=measured['elapsed_seconds']/3600,
                   resource_measurement_note='CPU-only suite timer covers fitting and statistics; feature loading precedes timer. Parameters per head; nine independently selected heads.',
                   implementation_sha256=measured['script_sha256'], coverage=1,
                   metrics_path=str(diagnostic.relative_to(ROOT)), log_path='logs/ncslgr_probes_v1.log',
                   scientific_scope='Historical positively recorded NEG/WH/YN type, one test signer, four shared XML archive groups; no verified absence, scope accuracy or method novelty.')
        rows.append(row)
    import yaml
    protocol_path = ROOT / 'configs/protocol_common_v1.yaml'
    protocol = yaml.safe_load(protocol_path.read_text())
    group_counts = {split: len({row['recording_group'] for _, row in read_jsonl(ROOT / protocol['manifest'])
                               if row['official_split'] == split and row['recording_group']})
                    for split in ['train', 'val', 'test']}
    for baseline, seed in [('B00', 0), ('B01', 0)] + [(b, s) for b in ['B02', 'B03'] for s in [0, 1, 2]]:
        run_id = 'common_v1_' + baseline + '_signrep_seed' + str(seed)
        path = ROOT / 'runs' / run_id
        row = dict.fromkeys(FIELDS)
        row.update(run_id=run_id, status='PENDING', protocol_version='common_v1', log_path='logs/common_v1_' + baseline + '_seed' + str(seed) + '.log')
        row.update(config_sha256=sha256(protocol_path), checkpoint_sha256=protocol['checkpoint_sha256'],
                   scientific_scope=protocol['scientific_claim_role'],
                   dataset_manifest_sha256=protocol['manifest_sha256'],
                   train_recording_count=group_counts['train'], validation_recording_count=group_counts['val'],
                   test_recording_count=group_counts['test'],
                   resource_measurement_note='Active GPU compute hours unmeasured. Recording counts refer to planned manifest; metric split_counts give surviving clips.')
        if (path / 'status.json').exists():
            state = json.loads((path / 'status.json').read_text())
            row.update(status=state['status'], pid=state.get('pid'), failure_reason=state.get('failure_reason'))
            if state['status'] == 'RUNNING' and live(state['pid'], 'scripts/run_probe.py') is False:
                row.update(status='FAIL', failure_reason='Probe PID missing without terminal status')
        if (path / 'metrics.json').exists():
            metrics = json.loads((path / 'metrics.json').read_text())
            row.update(metrics_path=str((path / 'metrics.json').relative_to(ROOT)), trainable_parameters=metrics['trainable_parameters'],
                       coverage=metrics['coverage'], wall_hours=metrics['elapsed_seconds'] / 3600,
                       gpu_hours=0 if baseline in ['B00', 'B01'] else None,
                       gpu_allocation_wall_hours=0 if baseline in ['B00', 'B01'] else metrics['elapsed_seconds'] / 3600,
                       implementation_sha256=metrics['metric_implementation_sha256'])
        rows.append(row)
    native_folders = [
        'shubert_encoder_smoke_attempt1', 'shubert_encoder_smoke_attempt2',
        'shubert_native_preprocessing_attempt1', 'shubert_native_preprocessing_attempt2',
        'shubert_native_pilot_attempt1',
        *['shubert_native_pilot_gpu_attempt' + str(attempt) for attempt in range(1, 5)]]
    for folder in native_folders:
        directory = ROOT / 'features' / folder
        report_path = directory / 'report.json'
        state_path = directory / 'status.json'
        if not report_path.exists() and not state_path.exists():
            continue
        row = dict.fromkeys(FIELDS)
        row.update(run_id=folder, log_path='logs/' + folder + '.log',
                   scientific_scope='Bounded native train-video smoke or synthetic encoder interface; no benchmark/grammar accuracy.')
        measured = json.loads((report_path if report_path.exists() else state_path).read_text())
        status = measured['status']
        row.update(status='PASS' if status.startswith('PASS') else status, pid=measured.get('pid'),
                   failure_reason=measured.get('failure_reason'), config_sha256=measured.get('config_sha256'),
                   implementation_sha256=measured.get('implementation_sha256'),
                   dataset_manifest_sha256=measured.get('manifest_sha256'))
        if status == 'RUNNING' and live(measured['pid'], 'shubert') is False:
            row.update(status='FAIL', failure_reason='Native process missing without terminal report; inspect process exit log')
        if report_path.exists():
            row['metrics_path'] = str(report_path.relative_to(ROOT))
            row['wall_hours'] = measured['elapsed_seconds'] / 3600
            gpu = measured.get('device', 'cpu').startswith('cuda')
            row.update(gpu_hours=None if gpu else 0,
                       gpu_allocation_wall_hours=measured['elapsed_seconds'] / 3600 if gpu else 0,
                       resource_measurement_note='GPU allocation wall-time proxy, not active compute hours.' if gpu else 'CPU-only job; no GPU allocated.',
                       trainable_parameters=0,
                       coverage=(measured.get('passed', measured.get('samples', 1)) / measured.get('samples', 1)) if status.startswith('PASS') else None)
            audit = measured.get('encoder_load_audit', {})
            row.update(code_commit=audit.get('upstream_commit', measured.get('upstream_commit')),
                       checkpoint_sha256=audit.get('checkpoint_sha256'))
        rows.append(row)
    for folder, terminal_name in [('shubert_native_common_v1_preprocessing', 'report.json'),
                                  ('shubert_native_common_v1', 'extraction_report.json'),
                                  ('shubert_native_ncslgr_v1_preprocessing', 'report.json'),
                                  ('shubert_native_ncslgr_v1', 'extraction_report.json')]:
        directory = ROOT / 'features' / folder
        state_path = directory / 'status.json'
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        terminal = directory / terminal_name
        # A bounded pilot report can survive a later failed attempt. The latest
        # producer state wins unless its terminal report agrees with that state.
        measured, measured_path = state, state_path
        if terminal.exists() and state['status'] != 'RUNNING':
            candidate = json.loads(terminal.read_text())
            if candidate['status'] == state['status'] and candidate.get('success', candidate.get('passed')) == state.get('success', state.get('passed')):
                measured, measured_path = candidate, terminal
        row = dict.fromkeys(FIELDS)
        row.update(run_id=folder, status=measured['status'], pid=measured.get('pid'),
                   protocol_version='native_ncslgr_v1' if 'ncslgr' in folder else 'native_common_v1', config_sha256=measured.get('config_sha256'),
                   implementation_sha256=measured.get('implementation_sha256'),
                   dataset_manifest_sha256=measured.get('manifest_sha256'),
                   trainable_parameters=0, failure_reason=measured.get('failure_reason'),
                   log_path='logs/' + folder + '.log',
                   metrics_path=str(measured_path.relative_to(ROOT)),
                   scientific_scope='Full locked native public-pipeline extraction; missing streams and source PTS preserved. Usable readout coverage and native accuracy pending.')
        # New CPU jobs have PID3 in a different per-tool namespace. The same
        # number can name this registry process; it cannot establish liveness.
        observed = live(measured['pid'], 'shubert_cohort') if measured['status'] == 'RUNNING' and measured['pid'] != 3 and 'ncslgr' not in folder else None
        if measured['status'] == 'RUNNING' and observed is False:
            row.update(status='FAIL', failure_reason='Native cohort process missing without terminal status; inspect existing session before recovery')
        elapsed = measured.get('elapsed_seconds')
        gpu = measured.get('device', 'cpu').startswith('cuda')
        row.update(wall_hours=elapsed/3600 if elapsed is not None else None,
                   gpu_hours=None if gpu else 0,
                   gpu_allocation_wall_hours=elapsed/3600 if gpu and elapsed is not None else 0,
                   coverage=measured.get('success', measured.get('passed', 0))/measured['samples'],
                   resource_measurement_note='Allocation wall time includes CPU-producer waits; not active CUDA compute time.' if gpu else 'CPU-only preparation; attempt wall-time reported separately from feature worker.')
        if folder == 'shubert_native_common_v1':
            row['log_path'] = 'logs/shubert_native_common_v1_extraction.log'
        if folder == 'shubert_native_ncslgr_v1':
            row['log_path'] = 'logs/shubert_native_ncslgr_v1_cpu_extraction.log'
        if measured['status'] == 'RUNNING' and observed is None:
            row['resource_measurement_note'] += ' Host PID not visible in current tool namespace; RUNNING is producer-reported, not independently inferred from /proc. Inspect progress and exclusive worker lock before any recovery.'
        rows.append(row)
    fair_ncslgr = ROOT / 'runs/fair_ncslgr_v1/status.json'
    if fair_ncslgr.exists():
        measured = json.loads(fair_ncslgr.read_text())
        config = ROOT / 'configs/protocol_fair_ncslgr_v1.json'
        terminal = fair_ncslgr.parent / 'summary.json'
        row = dict.fromkeys(FIELDS)
        row.update(run_id='fair_ncslgr_v1_suite', status=measured['status'], pid=measured.get('pid'),
                   protocol_version='fair_ncslgr_v1', config_sha256=sha256(config),
                   implementation_sha256=sha256(ROOT / 'scripts/run_fair_ncslgr.py'),
                   gpu_hours=0, gpu_allocation_wall_hours=0, trainable_parameters=6915,
                   wall_hours=measured.get('elapsed_seconds', 0)/3600,
                   resource_measurement_note='CPU-only18equal-budget heads after full native PASS. Waiting wall time separate from fitting time; no extra GPU.',
                   metrics_path=str((terminal if terminal.exists() else fair_ncslgr).relative_to(ROOT)),
                   failure_reason=measured.get('failure_reason'), log_path='logs/fair_ncslgr_v1.log',
                   scientific_scope='Exploratory recorded positive functional intervals on common eligible IDs; predeclared train-duration prior. No absence, adjudicated scope, encoder adaptation or novelty claim.')
        rows.append(row)
    translation = ROOT / 'features/shubert_translation_smoke_attempt1/report.json'
    author_pool = ROOT / 'runs/ncslgr_author_pool_v1/summary.json'
    if author_pool.exists():
        measured = json.loads(author_pool.read_text())
        row = dict.fromkeys(FIELDS)
        row.update(run_id='ncslgr_author_pool_v1_suite', status=measured['status'],
                   protocol_version='ncslgr_author_pool_v1', config_sha256=measured['config_sha256'],
                   implementation_sha256=sha256(ROOT / 'scripts/run_ncslgr_author_pool.py'),
                   gpu_hours=0, gpu_allocation_wall_hours=0, trainable_parameters=2307,
                   wall_hours=measured['elapsed_seconds']/3600, coverage=1,
                   metrics_path=str(author_pool.relative_to(ROOT)), log_path='logs/ncslgr_author_pool_v1.log',
                   resource_measurement_note='CPU-only9heads, same3LR/100epochs/3seeds; no additional GPU.',
                   scientific_scope='Exploratory author-imputed contextual-token pooling on222positive-function utterances. Observation counts retained separately. Native result files existed at lock; no blind, independent, causal or novelty claim.')
        rows.append(row)
    if translation.exists():
        measured = json.loads(translation.read_text())
        row = dict.fromkeys(FIELDS)
        row.update(run_id='shubert_translation_smoke_attempt1', status=measured['status'],
                   checkpoint_sha256=measured['checkpoint_sha256'], trainable_parameters=0,
                   gpu_hours=0, gpu_allocation_wall_hours=0, wall_hours=measured['elapsed_seconds']/3600,
                   implementation_sha256=measured['implementation_sha256'], coverage=1,
                   metrics_path=str(translation.relative_to(ROOT)), log_path='logs/shubert_translation_smoke_attempt1.log',
                   resource_measurement_note='CPU-only local numerical smoke; no GPU allocated.',
                   scientific_scope='Author-demo11625 strict load/token NLL/padding/generation on one source-captioned clip; not semantic pair accuracy or ASL-MTP reproduction.')
        rows.append(row)
    intervals = ROOT / 'runs/ncslgr_intervals_v1/summary.json'
    if intervals.exists():
        measured = json.loads(intervals.read_text())
        config = ROOT / 'configs/protocol_ncslgr_intervals_v1.json'
        p = json.loads(config.read_text())
        row = dict.fromkeys(FIELDS)
        row.update(run_id='ncslgr_recorded_interval_suite_v1', status=measured['status'],
                   protocol_version='ncslgr_intervals_v1', config_sha256=sha256(config),
                   implementation_sha256=measured['script_sha256'], trainable_parameters=measured['head_parameters'],
                   dataset_manifest_sha256=p['identity_sha256']['data/external/ncslgr/diagnostic_v1/manifest.jsonl'],
                   train_recording_count=107, validation_recording_count=27, test_recording_count=88,
                   gpu_hours=0, gpu_allocation_wall_hours=0, wall_hours=measured['elapsed_seconds']/3600,
                   resource_measurement_note='CPU-only12heads;6915active parameters each. Classification222utterances; recorded-single-interval loss/metric179utterances (89/25/65).',
                   coverage=1, metrics_path=str(intervals.relative_to(ROOT)), log_path='logs/ncslgr_intervals_v1.log',
                   scientific_scope='Exploratory recorded functional-core interval readout, not adjudicated grammatical scope, absent detection, representation adaptation or novelty.')
        rows.append(row)
    calibration = ROOT / 'runs/ncslgr_calibration_v1/summary.json'
    if calibration.exists():
        measured = json.loads(calibration.read_text())
        row = dict.fromkeys(FIELDS)
        row.update(run_id='ncslgr_categorical_calibration_v1', status=measured['status'],
                   protocol_version='ncslgr_calibration_v1', config_sha256=measured['config_sha256'],
                   implementation_sha256=sha256(ROOT / 'scripts/run_ncslgr_calibration.py'),
                   trainable_parameters=0, validation_recording_count=27, test_recording_count=88,
                   gpu_hours=0, gpu_allocation_wall_hours=0, wall_hours=measured['elapsed_seconds']/3600,
                   metrics_path=str(calibration.relative_to(ROOT)), log_path='logs/ncslgr_calibration_v1.log',
                   resource_measurement_note='CPU-only validation grid selection on9existing2307parameter heads; no retraining. Zero gradient-trained new parameters; one selected scalar temperature per head.',
                   scientific_scope='Exploratory categorical probability sensitivity; original predictions preserved. Not boundary/presence calibration, independent confirmation or novelty.')
        rows.append(row)
    fair_status = ROOT / 'runs/fair_common_v1/status.json'
    if fair_status.exists():
        measured = json.loads(fair_status.read_text())
        row = dict.fromkeys(FIELDS)
        terminal = fair_status.parent / 'summary.json'
        row.update(run_id='fair_common_v1_suite', status=measured['status'], pid=measured.get('pid'),
                   protocol_version='fair_common_v1', config_sha256=measured['config_sha256'],
                   gpu_hours=0, gpu_allocation_wall_hours=0,
                   wall_hours=measured.get('elapsed_seconds', 0)/3600,
                   resource_measurement_note='CPU-only waiter and future equal-grid controls; native GPU worker remains separate. Waiting time is not fitting/compute time.',
                   metrics_path=str((terminal if terminal.exists() else fair_status).relative_to(ROOT)),
                   log_path='logs/fair_common_v1.log', failure_reason=measured.get('failure_reason'),
                   scientific_scope='Locked three-backbone lexical common intersection and same tuning grid. No partial native test inference; no matched-context causal or novelty claim.')
        rows.append(row)
    stream = io.StringIO()
    semantic_readiness = ROOT / 'reports/semantic_pairs_readiness_v1/status.json'
    if semantic_readiness.exists():
        measured = json.loads(semantic_readiness.read_text())
        row = dict.fromkeys(FIELDS)
        row.update(run_id='semantic_pairs_readiness_v1', status=measured['status'], config_sha256=measured['config_sha256'],
                   implementation_sha256=measured['script_sha256'], gpu_hours=0, gpu_allocation_wall_hours=0,
                   metrics_path=str(semantic_readiness.relative_to(ROOT)), log_path='logs/semantic_pairs_readiness_v1.log',
                   failure_reason=measured['failure_reason'], scientific_scope='Readiness dry-run only; blocked before model loading/scoring by missing reviewed external gold/media. No semantic accuracy.')
        rows.append(row)
    semantic_backend = ROOT / 'features/semantic_pair_backend_smoke_attempt1/report.json'
    if semantic_backend.exists():
        measured = json.loads(semantic_backend.read_text())
        row = dict.fromkeys(FIELDS)
        row.update(run_id='semantic_pair_backend_smoke_attempt1', status=measured['status'],
                   implementation_sha256=measured['source_sha256']['scripts/smoke_semantic_pair_backend.py'],
                   trainable_parameters=0, gpu_hours=0, gpu_allocation_wall_hours=0,
                   wall_hours=measured['elapsed_seconds']/3600, metrics_path=str(semantic_backend.relative_to(ROOT)),
                   log_path='logs/semantic_pair_backend_smoke_attempt1.log',
                   resource_measurement_note='CPU numerical integration on one 104-frame training-signer clip, two identical caption candidates and blank streams.',
                   scientific_scope='Strict offline model load and source-caption NLL reproduction; identical-text tie credit 0.5. No external semantic-pair accuracy or causal visual-reliance claim.')
        rows.append(row)
    grounding_readiness = ROOT / 'reports/grounding_readiness_v1/status.json'
    if grounding_readiness.exists():
        measured = json.loads(grounding_readiness.read_text())
        row = dict.fromkeys(FIELDS)
        row.update(run_id='grounding_readiness_v1', status=measured['status'], config_sha256=measured['config_sha256'],
                   implementation_sha256=measured['script_sha256'], gpu_hours=0, gpu_allocation_wall_hours=0,
                   metrics_path=str(grounding_readiness.relative_to(ROOT)), log_path='logs/grounding_readiness_v1.log',
                   failure_reason=measured['failure_reason'],
                   scientific_scope='Q02 readiness dry-run stops before search/metrics because verified ASL mapping, exhaustive occurrence/absence and grouping are missing. Synthetic integration tests are not linguistic evidence.')
        rows.append(row)
    recovery = ROOT / 'state/native_common_disk_recovery_v1/status.json'
    if recovery.exists():
        measured = json.loads(recovery.read_text())
        row = dict.fromkeys(FIELDS)
        row.update(run_id='native_common_disk_recovery_v1', status=measured['status'], pid=measured['pid'],
                   implementation_sha256=measured['script_sha256'], gpu_hours=0 if measured['status'] == 'WAITING_FOR_CAPACITY' else None,
                   metrics_path=str(recovery.relative_to(ROOT)), log_path='logs/native_common_disk_recovery_v1.log',
                   wall_hours=measured.get('elapsed_seconds', 0)/3600, failure_reason=measured.get('failure_reason'),
                   resource_measurement_note='Capacity waiter; then CPU preprocessing, oneGPU0consumer, CPUcontrols serially. Phase wall-time is not CUDA compute time.',
                   scientific_scope='Reviewed disk-guard recovery of same locked4726cohort; priorfailedattempts retained, no partial evaluation.')
        rows.append(row)
    recovered_fair = ROOT / 'runs/fair_common_v1_recovery1/status.json'
    if recovered_fair.exists():
        measured = json.loads(recovered_fair.read_text())
        terminal = recovered_fair.parent / 'summary.json'
        row = dict.fromkeys(FIELDS)
        row.update(run_id='fair_common_v1_recovery1_suite', status=measured['status'], pid=measured.get('pid'),
                   config_sha256=measured['config_sha256'], protocol_version='fair_common_v1', gpu_hours=0,
                   metrics_path=str((terminal if terminal.exists() else recovered_fair).relative_to(ROOT)),
                   log_path='logs/fair_common_v1_recovery1.log', scientific_scope='Same locked equal-grid lexical suite in separate recovery output; original prefittingFAIL retained.')
        rows.append(row)
    branch_import_failure = ROOT / 'evidence/ncslgr_branch_ablation_import_failure_v1/manifest.json'
    if branch_import_failure.exists():
        measured=json.loads(branch_import_failure.read_text())
        row=dict.fromkeys(FIELDS)
        row.update(run_id='ncslgr_branch_ablation_import_failure_v1',status=measured['status'],
                   metrics_path=str(branch_import_failure.relative_to(ROOT)),log_path='logs/ncslgr_branch_ablation_smoke_v1.log',
                   failure_reason=measured['failure_reason'],gpu_hours=0,gpu_allocation_wall_hours=0,
                   scientific_scope='Failed before model loading due eager XML dependency import; preserved original protocol and sources. No scientific metrics.')
        rows.append(row)
    for version in ['v2','v3']:
        branch_config=ROOT/f'configs/protocol_ncslgr_branch_ablation_{version}.json'
        if not branch_config.exists():
            continue
        policy=json.loads(branch_config.read_text())
        smoke=ROOT/f'reports/ncslgr_branch_ablation_smoke_{version}_audit.json'
        if smoke.exists():
            measured=json.loads(smoke.read_text());row=dict.fromkeys(FIELDS)
            row.update(run_id=f'ncslgr_branch_ablation_smoke_{version}',status=measured['status'],config_sha256=measured['config_sha256'],
                       metrics_path=str(smoke.relative_to(ROOT)),log_path=f'logs/ncslgr_branch_ablation_smoke_{version}.log',gpu_hours=0,gpu_allocation_wall_hours=0,
                       scientific_scope='Training-only numerical cue-intervention smoke; source clocks/masks unchanged, no classification, adaptation or novelty.')
            rows.append(row)
        for phase,relative,log in [('features',f'features/ncslgr_branch_ablation_{version}/status.json',f'logs/ncslgr_branch_ablation_extraction_{version}.log'),
                                   ('readouts',f'runs/ncslgr_branch_ablation_{version}/status.json',f'logs/ncslgr_branch_ablation_readouts_{version}.log'),
                                   ('pipeline',f'state/ncslgr_branch_ablation_{version}/status.json',f'logs/ncslgr_branch_ablation_pipeline_{version}.log')]:
            path=ROOT/relative
            if not path.exists():
                continue
            measured=json.loads(path.read_text());terminal=path.parent/('summary.json' if phase=='readouts' else 'extraction_report.json')
            row=dict.fromkeys(FIELDS)
            row.update(run_id=f'ncslgr_branch_ablation_{version}_{phase}',status=measured['status'],config_sha256=sha256(branch_config),pid=measured.get('pid'),
                       implementation_sha256=policy['identity_sha256']['scripts/extract_ncslgr_branch_ablation.py' if phase=='features' else 'scripts/run_ncslgr_branch_ablation.py'] if phase!='pipeline' else measured['script_sha256'],
                       trainable_parameters=policy['head_parameters'] if phase=='readouts' else 0,
                       metrics_path=str((terminal if phase=='readouts' and terminal.exists() else path).relative_to(ROOT)),log_path=log,
                       wall_hours=measured.get('elapsed_seconds',0)/3600,gpu_hours=0,gpu_allocation_wall_hours=0,
                       failure_reason=measured.get('failure_reason'),
                       resource_measurement_note='CPU-only fixed eight cue interventions; exact175cohort,24equal-budget heads only after complete feature extraction.',
                       scientific_scope='Exploratory zeroing distribution-shift diagnostic with matched fresh all-stream baseline. Original-cache numerical equivalence unestablished forv3; failures preserved. No representation adaptation/novelty.')
            rows.append(row)
    numerical=ROOT/'reports/ncslgr_branch_reproduction_diagnostic_v1.json'
    if numerical.exists():
        measured=json.loads(numerical.read_text());row=dict.fromkeys(FIELDS)
        row.update(run_id='ncslgr_branch_reproduction_diagnostic_v1',status=measured['status'],metrics_path=str(numerical.relative_to(ROOT)),
                   implementation_sha256=measured['source_sha256_map']['scripts/diagnose_ncslgr_branch_reproduction.py'],
                   wall_hours=measured['elapsed_seconds']/3600,gpu_hours=0,gpu_allocation_wall_hours=0,log_path='logs/ncslgr_branch_reproduction_diagnostic_v1.log',
                   scientific_scope='One failed training clip numerical repeat/thread/cache comparison; not scientific classification or cache-equivalence certification.')
        rows.append(row)
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(ROOT / 'runs/experiment_registry.csv', stream.getvalue())
    print(json.dumps({'registry': 'runs/experiment_registry.csv', 'runs': len(rows), 'states': {row['run_id']: row['status'] for row in rows}}))


if __name__ == '__main__':
    main()
