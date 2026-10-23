#!/usr/bin/env python3
"""Run production C++ prompt generation on every dataset pair (no inference).

Requires macOS Python with OpenCV 4.8 and Xcode command-line tools. OpenCV's
exported C++ symbols are loaded from that Python wheel, so the iOS libraries
are not run on the Mac. This verifies prompts, never iPhone performance.
"""
import ctypes
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-model', action='store_true',
                        help='Run both named regression images through actual C++ SAM on desktop CPU')
    parser.add_argument('--macos-runtime', type=Path,
                        help='Optional desktop regression dependency: path to ONNX Runtime 1.21.0 macOS framework')
    args = parser.parse_args()
    if 'DEVELOPER_DIR' not in os.environ:
        os.environ['DEVELOPER_DIR'] = subprocess.check_output(['xcode-select', '-p'], text=True).strip()
    import cv2
    assert cv2.__version__.startswith('4.8.'), 'Use host OpenCV 4.8 to match the app'
    package = Path(cv2.__file__).parent
    binary = next(package.glob('cv2*.so'))
    ctypes.CDLL(str(binary), mode=ctypes.RTLD_GLOBAL)
    output = ROOT / ('validation/segmentation-regression' if args.real_model else 'validation/prompt-regression')
    output.mkdir(parents=True, exist_ok=True)
    library = output / 'prompt-regression.dylib'
    includes = ([] if args.real_model else [ROOT / 'tests/prompt_stub'])
    includes += [ROOT / 'Sources', ROOT / 'Vendor/SegmentAnything', ROOT / 'Vendor/onnxruntime.framework/Headers']
    includes += sorted((ROOT / 'Vendor/iForester').iterdir())
    sources = [ROOT / 'tests/PromptDatasetRegression.cpp', ROOT / 'Sources/RuntimePipeline.cpp']
    sources += sorted((ROOT / 'Vendor/iForester').glob('*/*.cpp'))
    command = ['xcrun', '--sdk', 'macosx', 'clang++', '-std=c++17', '-O2', '-dynamiclib',
               '-undefined', 'dynamic_lookup', '-F', str(ROOT / 'Vendor')]
    command += ['-I' + str(p) for p in includes if p.is_dir()]
    if args.real_model:
        # Optional host-only dependency; the iPhone app uses the bundled framework.
        if not args.macos_runtime:
            parser.error('--real-model requires --macos-runtime /path/to/onnxruntime.framework (macOS 1.21.0)')
        runtime = args.macos_runtime.resolve()
        assert (runtime / 'Versions/A/onnxruntime').exists(), 'macOS ONNX Runtime framework missing'
        sources.append(ROOT / 'Vendor/SegmentAnything/sam.cpp')
        command += ['-DRUNTIME_REAL_SAM_REGRESSION',
                    '-DRUNTIME_MODELS_DIRECTORY="' + str(ROOT / 'Vendor/SegmentAnything/mobile_sam') + '"',
                    str(runtime / 'Versions/A/onnxruntime'),
                    '-framework', 'Foundation', '-framework', 'CoreML', '-framework', 'Accelerate']
    command += [str(p) for p in sources] + ['-o', str(library)]
    subprocess.run(command, check=True)
    runner = ctypes.CDLL(str(library)).inspectPrompt
    runner.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    runner.restype = ctypes.c_int
    report = output / 'prompts.txt'
    report.write_text('')
    dataset = ROOT / 'Resources/Dataset'
    records = json.loads((dataset / 'manifest.json').read_text())['records']
    if args.real_model:
        records = [r for r in records if r['id'].startswith(('03BB002E', '50AAA4FD'))]
        assert len(records) == 2
    failures = []
    for item in records:
        if args.real_model:
            report = output / (item['id'] + '.txt')
            report.write_text('')
        reached = runner(os.fsencode(dataset / item['rgb']), os.fsencode(dataset / item['depth']),
                         item['baseX'], item['baseY'], os.fsencode(report))
        if reached != 1: failures.append(item['id'])
    summary = dict(images=len(records), passed=len(records)-len(failures),
                   failures=failures, host_opencv=cv2.__version__,
                   note=('Production C++ pipeline and actual ONNX models on desktop CPU; nonempty masks required; no iPhone timing.'
                         if args.real_model else
                         'Production C++ prompt pipeline with a test encoder boundary; no model inference or device timing.'))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    assert not failures, 'Some regression images failed; see the report'


if __name__ == '__main__':
    main()
