"""Run research integrity tests and save authoritative exit/count evidence."""
import json
import re
import subprocess
import sys

from _common import ROOT
from signrepr.io import atomic_text, sha256, write_json


def main():
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    output = result.stdout + result.stderr
    atomic_text(ROOT / 'logs/integrity_checks.log', output)
    count = re.search(r'Ran (\d+) tests', output)
    report = {'status': 'PASS' if result.returncode == 0 and count else 'FAIL',
              'exit_code': result.returncode, 'test_count': int(count[1]) if count else None,
              'command': command, 'python_executable': sys.executable,
              'test_source_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in (ROOT / 'tests').glob('test_*.py')},
              'log': 'logs/integrity_checks.log'}
    write_json(ROOT / 'reports/integrity_checks.json', report)
    print(json.dumps(report))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
