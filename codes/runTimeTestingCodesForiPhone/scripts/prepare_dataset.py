#!/usr/bin/env python3
"""Download the pinned, complete RGB/depth dataset; verify every Git blob.

Run again to resume. Only verified files enter the manifest. No Python package
dependencies. This preparation runs on the Mac; all inference runs on the phone.
"""
import concurrent.futures
import csv
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / 'Resources'
DATA = RESOURCES / 'Dataset'
TREE = json.loads((RESOURCES / 'repository-tree.json').read_text())
COMMIT = TREE['sha']
BASE = f'https://raw.githubusercontent.com/yinzhiheng/AI-assisted-DBH-measurement-pipeline/{COMMIT}/'


def git_hash(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def download(entry):
    relative = entry['path'].removeprefix('data/')
    dest = DATA / relative
    if dest.exists() and git_hash(dest.read_bytes()) == entry['sha']:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            request = urllib.request.Request(BASE + entry['path'], headers={'User-Agent': 'iForester-runtime-test'})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if len(payload) != entry['size'] or git_hash(payload) != entry['sha']:
                raise ValueError(f'Checksum mismatch: {relative}')
            temporary = dest.with_suffix(dest.suffix + '.partial')
            temporary.write_bytes(payload)
            temporary.replace(dest)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    entries = [e for e in TREE['tree'] if e['type'] == 'blob' and e['path'].startswith('data/')]
    print(f'Downloading/verifying {len(entries)} files ({sum(e["size"] for e in entries) / 1e6:.1f} MB)', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download, entry) for entry in entries]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if count % 50 == 0 or count == len(entries):
                print(f'{count}/{len(entries)} verified', flush=True)

    by_path = {e['path']: e for e in entries}
    records = []
    duplicate_annotations = []
    for field in ('natural', 'plotted'):
        with (DATA / f'basePoints_{field}.csv').open(encoding='utf-8-sig', newline='') as source:
            rows = list(csv.DictReader(source))
        bases = {row['uniqueName']: row for row in rows}
        # Last row wins, explicitly recorded below: the published natural CSV
        # contains one image with two distinct taps. Keep the original CSV too.
        for row in rows:
            if row != bases[row['uniqueName']]:
                duplicate_annotations.append(dict(field=field, id=row['uniqueName'],
                    earlier=row, selected=bases[row['uniqueName']], policy='last_csv_row'))
        images = sorted(e['path'] for e in entries if e['path'].startswith(f'data/rawdata/rgb_{field}Field/'))
        for rgb in images:
            image_id = Path(rgb).stem.removeprefix('rgb-')
            depth = f'data/rawdata/depth_{field}Field/combine-{image_id}.png'
            assert depth in by_path, f'Missing paired depth: {rgb}'
            assert image_id in bases, f'Missing base annotation: {rgb}'
            base = bases[image_id]
            records.append(dict(id=image_id, field=field,
                                rgb=rgb.removeprefix('data/'), depth=depth.removeprefix('data/'),
                                baseX=int(base['baseCol']), baseY=int(base['baseRow']),
                                rgbSHA1=by_path[rgb]['sha'], depthSHA1=by_path[depth]['sha']))
    assert len(records) == 487
    manifest = dict(commit=COMMIT, records=records, totalBytes=sum(e['size'] for e in entries),
                    duplicateAnnotations=duplicate_annotations)
    temporary = DATA / 'manifest.json.partial'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(DATA / 'manifest.json')
    print('Ready: 285 natural + 202 plotted images, each with depth and base point.', flush=True)


if __name__ == '__main__':
    main()
