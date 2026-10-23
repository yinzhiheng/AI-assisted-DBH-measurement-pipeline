#!/usr/bin/env python3
"""Offline integrity/metadata validation. Optionally also verify a built .app."""
import argparse
import collections
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path
from upstream_localizer import verify_port

ROOT = Path(__file__).resolve().parents[1]


def jpeg_size(path):
    data = path.read_bytes()
    assert data[:2] == b'\xff\xd8', path
    i = 2
    while i < len(data):
        assert data[i] == 255, path
        while data[i] == 255: i += 1
        marker = data[i]; i += 1
        if marker in (0xD8, 0xD9): continue
        length = int.from_bytes(data[i:i+2], 'big')
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack('>HH', data[i+3:i+7])
            return w, h
        i += length
    raise ValueError(f'JPEG size not found: {path}')


def png_metadata(path):
    data = path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    w, h, bits, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', data[16:29])
    assert (bits, color, compression, filtering, interlace) == (16, 0, 0, 0, 0)
    stream = bytearray(); i = 8
    while i < len(data):
        count = int.from_bytes(data[i:i+4], 'big')
        kind = data[i+4:i+8]; payload = data[i+8:i+8+count]
        assert zlib.crc32(kind + payload) == int.from_bytes(data[i+8+count:i+12+count], 'big')
        if kind == b'IDAT': stream.extend(payload)
        i += count + 12
    pixels = zlib.decompress(stream)
    assert len(pixels) == h * (w * 2 + 1)
    previous = bytearray(w * 2); meta = bytearray()
    for row_index in range(4):
        offset = row_index * (w * 2 + 1)
        filter_type = pixels[offset]
        row = bytearray(pixels[offset+1:offset+1+w*2])
        for x in range(w * 2):
            a = row[x-2] if x >= 2 else 0
            b = previous[x]
            c = previous[x-2] if x >= 2 else 0
            if filter_type == 0: predictor = 0
            elif filter_type == 1: predictor = a
            elif filter_type == 2: predictor = b
            elif filter_type == 3: predictor = (a+b)//2
            elif filter_type == 4:
                p = a+b-c
                pa,pb,pc = abs(p-a),abs(p-b),abs(p-c)
                predictor = a if pa <= pb and pa <= pc else b if pb <= pc else c
            else: raise ValueError('Unsupported PNG filter')
            row[x] = (row[x] + predictor) & 255
        meta.extend(row); previous = row
    allowed = set(b'|,.-+0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    def tokens(blob):
        return [t.strip() for t in ''.join(chr(b) if b in allowed else ' ' for b in blob).split('|') if t.strip()]
    be = tokens(meta)
    le = tokens(b''.join(meta[i:i+2][::-1] for i in range(0, len(meta), 2)))
    chosen = be if len(be) > len(le) else le  # Exactly the original Load/load.cpp rule.
    assert len(chosen) >= 21, (path, chosen)
    k = [float(v) for v in chosen[19].replace(' ', '').split(',')]
    assert len(k) == 9 and all(math.isfinite(v) for v in k) and k[0] > 0 and k[4] > 0, (path, k)
    return w, h - 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', type=Path)
    args = parser.parse_args()
    (ROOT/'validation').mkdir(exist_ok=True)
    opencv = ROOT/'Vendor/opencv2.framework/Versions/A/opencv2'
    if opencv.stat().st_size < 1024:
        raise SystemExit('OpenCV is a Git LFS pointer. Run git lfs install and git lfs pull before building.')
    dataset = ROOT / 'Resources/Dataset'
    manifest = json.loads((dataset / 'manifest.json').read_text())
    counts = collections.Counter(r['field'] for r in manifest['records'])
    assert counts == {'natural': 285, 'plotted': 202}
    assert len(set(r['id'] for r in manifest['records'])) == 487
    tree = json.loads((ROOT/'Resources/repository-tree.json').read_text())
    files = [e for e in tree['tree'] if e['type'] == 'blob' and e['path'].startswith('data/')]
    for e in files:
        relative = e['path'].removeprefix('data/')
        data = (dataset/relative).read_bytes()
        assert len(data) == e['size']
        assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() == e['sha'], relative
        if args.app: assert (args.app/'Dataset'/relative).read_bytes() == data
    shapes = collections.Counter()
    for r in manifest['records']:
        w,h = jpeg_size(dataset/r['rgb']); dw,dh = png_metadata(dataset/r['depth'])
        assert w*dh == h*dw and 0 <= r['baseX'] < w and 0 <= r['baseY'] < h, r['id']
        shapes[f'{w}x{h} / {dw}x{dh}'] += 1
    copies = json.loads((ROOT/'Resources/source-provenance.json').read_text())
    verify_port()
    for entry in copies:
        original = ROOT.parent/entry['source']
        if original.exists(): assert hashlib.sha256(original.read_bytes()).hexdigest() == entry['sha256'], 'Original modified: '+str(original)
        # These two files now use the pinned GitHub recovery algorithm. Its
        # exact source and limited iOS adaptations are checked by verify_port().
        if entry['copy'].startswith('Vendor/iForester/BreastHeightLocalizer/'):
            continue
        data = (ROOT/entry['copy']).read_bytes()
        if entry['copy'].endswith('DBHMeasurer.h'):
            data = data.replace(b'struct RuntimeSegmentationResult;\r\n\r\nclass DBHMeasurer', b'class DBHMeasurer')
            data = data.replace(b'public:\r\n    RuntimeSegmentationResult segmentForRuntime(const Frame&, const CameraModel&, Sam&, const cv::Point2i&);', b'public:')
        if entry['copy'] == 'Vendor/SegmentAnything/sam.hpp':
            for addition in (
                b'#include "sam_inference_timing.hpp"\n',
                b'  SamInferenceTiming inferenceTiming;\n',
                b'    void resetInferenceTiming() { inferenceTiming = {}; }\n',
                b'    const SamInferenceTiming& getInferenceTiming() const { return inferenceTiming; }\n',
            ):
                assert data.count(addition) == 1, 'Missing/duplicate timing instrumentation'
                data = data.replace(addition, b'', 1)
        if entry['copy'] == 'Vendor/SegmentAnything/sam.cpp':
            encoder_run = b'sessionPre->Run(runOptionsPre, inputNames, &inputTensor, 1, outputNames, &outputTensor, 1);'
            decoder_run = b'sessionSam->Run(runOptionsSam, inputNamesSam, inputTensorsSam.data(), inputTensorsSam.size(), outputNamesSam, 3);'
            timed_encoder = (b'        timeSamInference(inferenceTiming.encoderMs, [&] {\n'
                             b'            ' + encoder_run + b'\n        });')
            timed_decoder = (b'    auto outputTensorsSam = timeSamInference(inferenceTiming.decoderMs, [&] {\n'
                             b'        return ' + decoder_run + b'\n    });')
            assert data.count(timed_encoder) == 1 and data.count(timed_decoder) == 1
            data = data.replace(timed_encoder, b'        ' + encoder_run, 1)
            data = data.replace(timed_decoder, b'    auto outputTensorsSam = ' + decoder_run, 1)
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['copy']
    models = json.loads((ROOT/'Vendor/SegmentAnything/mobile_sam/provenance.json').read_text())['sha256']
    if args.app:
        for name, expected in models.items():
            assert hashlib.sha256((args.app/name).read_bytes()).hexdigest() == expected
        assert json.loads((args.app/'Dataset/manifest.json').read_text()) == manifest
    report = dict(status='passed', verified_files=len(files), image_counts=dict(counts),
                  shapes=dict(shapes), metadata_intrinsics_valid=487, base_points_in_bounds=487,
                  source_files_verified=len(copies), dataset_commit=manifest['commit'],
                  upstream_localizer_commit=json.loads((ROOT/'Reference/GitHub/provenance.json').read_text())['commit'],
                  duplicate_annotations=manifest['duplicateAnnotations'],
                  built_app_verified=bool(args.app), model_sha256=models,
                  note='Static data/build validation, not an iPhone timing run.')
    (ROOT/'validation/integrity.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
