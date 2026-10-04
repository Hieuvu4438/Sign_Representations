"""Bounded primary-PDF retrieval and local extraction, with explicit read advisory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import file_sha
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        import fitz
        records = []
        outputs = []
        for source in config['sources']:
            name = source['name']
            if Path(name).name != name or not source['url'].startswith('https://openaccess.thecvf.com/'):
                raise ValueError('Expected explicit primary CVF source')
            request = urllib.request.Request(source['url'], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=20) as response:
                data = response.read(config['max_bytes_per_source'] + 1)
                metadata = {k: v for k, v in response.headers.items()
                            if k.lower() in ('etag', 'content-length', 'content-type', 'last-modified')}
            if len(data) > config['max_bytes_per_source'] or not data.startswith(b'%PDF-'):
                raise ValueError('Oversize or non-PDF primary source')
            pdf = run.output / (name + '.pdf')
            pdf.write_bytes(data)
            sidecar = run.output / (name + '_preflight.json')
            subprocess.run([sys.executable, config['preflight_script'], str(pdf), '--output', str(sidecar)], check=True)
            advisory = json.loads(sidecar.read_text())
            # UNAVAILABLE permits section/table reading with an advisory, never trusted page anchors.
            if advisory['verdict'] == 'FAIL':
                raise ValueError('Primary PDF read-integrity failure')
            with fitz.open(pdf) as document:
                text = '\n\n'.join(page.get_text() for page in document)
                rendered_pages = len(document)
            if not text.strip():
                raise ValueError('No extracted primary text')
            text_path = run.output / (name + '.txt')
            text_path.write_text(text)
            records.append(dict(name=name, url=source['url'], bytes=len(data), pdf_sha256=file_sha(pdf),
                text_sha256=file_sha(text_path), preflight_sha256=file_sha(sidecar),
                preflight_verdict=advisory['verdict'], trusted_page_anchors=advisory['verdict'] == 'PASS',
                extraction_engine='PyMuPDF', observed_pages=rendered_pages, headers=metadata,
                scope='retrieval/extraction only; reading scope recorded separately in source card'))
            outputs.extend([pdf.name, sidecar.name, text_path.name])
        run.save('retrieval.json', dict(sources=records, fitz_version=fitz.VersionBind,
            preflight_script_sha256=file_sha(Path(config['preflight_script'])),
            no_install=True, GPU_used=False))
        outputs.append('retrieval.json')
        run.status['output_hashes'] = {name: file_sha(run.output / name) for name in outputs}


if __name__ == '__main__':
    main()
