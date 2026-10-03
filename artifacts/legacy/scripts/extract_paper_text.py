"""Extract PDF text with parser warnings preserved; never infer read coverage."""
import argparse
from pathlib import Path

import fitz

from _common import ROOT
from signrepr.io import atomic_text, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'evidence/papers')
    args = parser.parse_args()
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
    for path in sorted(args.directory.glob('*.pdf')):
        fitz.TOOLS.reset_mupdf_warnings()
        with fitz.open(path) as document:
            text = '\n'.join(page.get_text() for page in document)
            report = {'parser': 'PyMuPDF', 'parser_version': fitz.VersionBind,
                      'pdf_sha256': sha256(path), 'pages': len(document),
                      'extracted_characters': len(text), 'parser_warnings': fitz.TOOLS.mupdf_warnings(),
                      'repaired': document.is_repaired, 'encrypted': document.is_encrypted,
                      'read_level': 'NOT_ESTABLISHED_BY_EXTRACTION',
                      'page_anchor_trust': 'Requires separate ARS preflight; text locators use sections.'}
        atomic_text(path.with_suffix('.txt'), text)
        write_json(path.with_suffix('.extraction.json'), report)
        print(path.name, len(text), 'WARNINGS' if report['parser_warnings'] else 'NO_PARSER_WARNINGS')


if __name__ == '__main__':
    main()
