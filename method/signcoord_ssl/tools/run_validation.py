import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from signcoord.provenance import atomic_json,execution_stamp,object_hash,sha256_file


def main():
    evidence=PROJECT/'docs/evidence'; evidence.mkdir(parents=True,exist_ok=True)
    stamp=execution_stamp()
    def test_hash():
        return object_hash({str(p.relative_to(PROJECT)):sha256_file(p) for p in sorted((PROJECT/'tests').rglob('*.py'))})
    tests_before=test_hash()
    (PROJECT/'runs/validation').mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix='run_',dir=PROJECT/'runs/validation'))
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result=subprocess.run([sys.executable,'-m','pytest','-q',str(PROJECT/'tests'),
                           '--junitxml='+str(run/'tests_junit.xml')],env=env,text=True,capture_output=True)
    (run/'tests_output.txt').write_text(result.stdout+result.stderr)
    cases=ET.parse(run/'tests_junit.xml').getroot().findall('.//testcase')
    failed=sum(case.find('failure') is not None or case.find('error') is not None for case in cases)
    drift=execution_stamp()['implementation_hash']!=stamp['implementation_hash'] or test_hash()!=tests_before
    code=3 if drift else result.returncode
    report={**stamp,'status':'source_changed_during_validation' if drift else ('passed' if code==0 else 'failed'),'returncode':code,
            'test_source_hash':tests_before,'artifact_directory':str(run),'source_unchanged_during_execution':not drift,
            'test_count':len(cases),'failed':failed,'case_ids':[case.attrib['classname']+'::'+case.attrib['name'] for case in cases],
            'scope':'current implemented contracts; unfinished spec items remain in implementation_plan',
            'torch_runtime':'see local_inventory.json','source_trace':'data_card.md'}
    atomic_json(run/'tests_report.json',report)
    if not drift:
        for name in ('tests_junit.xml','tests_output.txt'):
            (evidence/name).write_bytes((run/name).read_bytes())
        atomic_json(evidence/'tests_report.json',report)
    print(json.dumps(report)); return code


if __name__=='__main__': raise SystemExit(main())
