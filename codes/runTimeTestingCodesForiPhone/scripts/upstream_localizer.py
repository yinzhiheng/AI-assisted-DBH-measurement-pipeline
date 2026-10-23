#!/usr/bin/env python3
"""Reproducible iOS adaptation of the pinned GitHub MobileSAM localizer.

Keep the upstream MobileSAM methods verbatim except for the existing iOS
PixelLoc_cpp name. Omit the unrelated SAM3 methods/dependency. Params lives
outside the class for Apple Clang's default-argument compatibility.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'Reference/GitHub'


def ported_files():
    manifest = json.loads((REFERENCE / 'provenance.json').read_text())
    for entry in manifest['files']:
        data = (ROOT / entry['copy']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']
        assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == entry['git_blob_sha1']

    source = (REFERENCE / 'BreastHeightLocalizer.cpp').read_text(encoding='latin-1')
    source = source.split('\nnamespace\n', 1)[0]
    source = source.replace('#include "../SegmentationModel/TreeMask.h"\n', '')
    source = source.replace('PixelLoc', 'PixelLoc_cpp')

    header = (REFERENCE / 'BreastHeightLocalizer.h').read_text()
    params_start = header.index('    struct Params\n')
    params_end = header.index('        // SAM3 uses', params_start)
    params = header[params_start:params_end]
    params = '\n'.join(line[4:] if line.startswith('    ') else line for line in params.splitlines()).rstrip() + '\n};\n\n'
    class_start = header.index('class BreastHeightLocalizer\n')
    constructor = header.index('    explicit BreastHeightLocalizer(')
    sam3_start = header.index('    // Port of fusedMethod.ipynb')
    private_start = header.index('private:\n')
    header = (header[:class_start] + params + 'class BreastHeightLocalizer\n{\npublic:\n' +
              header[constructor:sam3_start] + header[private_start:])
    header = header.replace('PixelLoc', 'PixelLoc_cpp')
    return {'BreastHeightLocalizer.cpp': source, 'BreastHeightLocalizer.h': header}


def verify_port():
    for name, expected in ported_files().items():
        actual = ROOT / 'Vendor/iForester/BreastHeightLocalizer' / name
        assert actual.read_text() == expected, f'GitHub localizer port differs: {actual}'


if __name__ == '__main__':
    verify_port()
    print('PASS: pinned GitHub MobileSAM localizer and iOS adaptation')
