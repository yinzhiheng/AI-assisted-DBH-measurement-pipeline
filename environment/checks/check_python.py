"""Runtime checks used by the independently runnable installation stages."""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_python(args):
    expected = (3, 10) if args.environment == 'export' else (3, 13)
    require(sys.version_info[:2] == expected, f'Expected Python {expected}, found {sys.version}')
    require(Path(sys.prefix).resolve() == args.venv.resolve(), f'Wrong environment: {sys.prefix}')
    require(sys.prefix != sys.base_prefix, 'An isolated virtual environment is required')
    require(struct.calcsize('P') == 8, '64-bit Python is required')
    require('include-system-site-packages = false' in (args.venv / 'pyvenv.cfg').read_text().lower(),
            'System-site packages must be disabled')
    print(f'PASS Python: {sys.version.split()[0]} at {sys.executable}', flush=True)
    return {'python': sys.version, 'executable': sys.executable, 'isolated': True}


def check_torch(args):
    report = check_python(args)
    import numpy as np
    import torch
    import torchvision
    from torchvision.ops import nms
    pair = ('2.5.1+cpu', '0.20.1+cpu') if args.environment == 'export' else (
        f'2.7.0+{args.device}', f'0.22.0+{args.device}')
    require((torch.__version__, torchvision.__version__) == pair,
            f'Wrong PyTorch/torchvision build: {torch.__version__}, {torchvision.__version__}; expected {pair}')
    torch.set_num_threads(2)
    array = np.arange(6, dtype=np.float32)
    require(np.array_equal(torch.from_numpy(array).numpy(), array), 'NumPy/PyTorch bridge failed')
    devices = ['cpu', 'cuda:0'] if args.device == 'cu126' else ['cpu']
    if args.device == 'cu126':
        require(torch.version.cuda == '12.6', f'Expected CUDA 12.6 wheel, found {torch.version.cuda}')
        require(torch.cuda.is_available(), 'CUDA unavailable: check the NVIDIA driver, or install the CPU environment')
        report['gpu'] = torch.cuda.get_device_name(0)
        report['gpu_memory_bytes'] = torch.cuda.get_device_properties(0).total_memory
    for device in devices:
        matrix = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        require(torch.allclose(matrix @ torch.eye(4, device=device), matrix), f'{device} matrix calculation failed')
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
        require(nms(boxes, torch.tensor([.9, .8], device=device), .5).tolist() == [0], f'{device} torchvision NMS failed')
        print(f'PASS PyTorch calculation and torchvision NMS: {device}', flush=True)
    report.update(torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda)
    return report


def load_exporter():
    path = ROOT / 'codes/segmentation/MobileSAMv2_export/export_dbh_onnx.py'
    spec = importlib.util.spec_from_file_location('dbh_export', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_packages(args):
    from importlib.metadata import version
    report = check_torch(args)
    recipes = ['mobilesamv2_export'] if args.environment == 'export' else ['sam31_cpu', 'visualization']
    packages = {}
    for recipe in recipes:
        for line in (ROOT / f'environment/python/{recipe}/requirements.txt').read_text().splitlines():
            if line.strip() and not line.startswith('#') and '==' in line:
                name, expected = line.strip().split('==')
                actual = version(name)
                require(actual.split('+')[0] == expected, f'{name}: expected {expected}, found {actual}')
                packages[name] = actual
    if args.environment == 'export':
        for module in ('cv2', 'onnx', 'onnxruntime', 'torchpack', 'onnxsim'):
            importlib.import_module(module)
        load_exporter()
    else:
        sys.path.insert(0, str(ROOT / 'codes/segmentation'))
        import batchProcess_SAM31_CPU as pipeline
        sys.path.insert(0, str(pipeline.locateSam31SourceDirectory()))
        from sam3 import model_builder  # noqa: F401
        for module in ('pandas', 'scipy', 'pycocotools.mask', 'jupyterlab', 'ipykernel', 'nbconvert'):
            importlib.import_module(module)
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(args.output_dir / 'plot-package-check.png')
        plt.close(fig)
    report['packages'] = packages
    print('PASS package versions, native imports and bundled model imports', flush=True)
    return report


def check_notebook(args):
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager
    output = args.output_dir / 'visualization'
    output.mkdir(parents=True, exist_ok=True)
    runtime = args.output_dir / 'jupyter-runtime'
    runtime.mkdir(exist_ok=True)
    os.environ.update(DBH_RESULTS_DIR=str(args.results_dir.resolve()), DBH_VISUALIZATION_DIR=str(output.resolve()),
                      JUPYTER_RUNTIME_DIR=str(runtime), IPYTHONDIR=str(runtime / 'ipython'))
    kernel = KernelManager(kernel_name='python3').format_kernel_cmd()[0]
    require(Path(kernel).resolve() == Path(sys.executable).resolve(), f'Notebook selected the wrong Python: {kernel}')
    notebook = nbformat.read(ROOT / 'visualization/fusedSAM_visualization_clean.ipynb', as_version=4)
    client = NotebookClient(notebook, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
    client.execute()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code' and cell.source.strip()]
    require(all(cell.execution_count is not None for cell in code_cells), 'Some notebook cells were not executed')
    nbformat.write(notebook, output / 'fusedSAM_visualization_executed.ipynb')
    html, _ = HTMLExporter().from_notebook_node(notebook)
    (output / 'index.html').write_text(html, encoding='utf-8')
    counts = {suffix: len(list(output.rglob(f'*.{suffix}'))) for suffix in ('png', 'pdf', 'csv')}
    require(counts['png'] >= 10 and counts['pdf'] >= 10 and counts['csv'] >= 13, f'Missing notebook outputs: {counts}')
    print(f'PASS all {len(code_cells)} notebook code cells and HTML export: {counts}', flush=True)
    return {'code_cells': len(code_cells), 'outputs': counts, 'kernel': kernel}


def check_export(args):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import cv2
    exporter = load_exporter()
    model, _ = exporter.build_model()
    torch.set_num_threads(2)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    sessions = []
    for name in ('mobile_sam_preprocess.onnx', 'mobile_sam.onnx'):
        path = args.model_dir / name
        onnx.checker.check_model(str(path))
        sessions.append(ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider']))
    image_path = sorted((ROOT / 'data/rawdata/rgb_naturalField').glob('*.jpeg'))[0]
    image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(cv2.resize(image, (1024, 720)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].copy()
    embedding = sessions[0].run(None, {sessions[0].get_inputs()[0].name: image})[0]
    with torch.inference_mode():
        expected_embedding = exporter.ImageEncoderExport(model)(torch.from_numpy(image)).numpy()
    np.testing.assert_allclose(embedding, expected_embedding, rtol=3e-3, atol=1e-2)
    inputs = {'image_embeddings': embedding, 'point_coords': np.array([[[512., 360.]]], np.float32),
              'point_labels': np.array([[1.]], np.float32), 'mask_input': np.zeros((1, 1, 256, 256), np.float32),
              'has_mask_input': np.zeros(1, np.float32), 'orig_im_size': np.array([720., 1024.], np.float32)}
    actual = sessions[1].run(None, inputs)
    with torch.inference_mode():
        expected = exporter.MultimaskExport(model, return_single_mask=False).eval()(
            **{k: torch.from_numpy(v) for k, v in inputs.items()})
    differences = []
    for a, e in zip(actual, expected):
        e = e.numpy()
        require(np.isfinite(a).all(), 'Non-finite ONNX predictions')
        np.testing.assert_allclose(a, e, rtol=3e-3, atol=1e-2)
        differences.append(float(np.max(np.abs(a - e))))
    print('PASS ONNX graph checks, encoder/decoder inference and agreement with PyTorch', flush=True)
    return {'encoder_max_abs_error': float(np.max(np.abs(embedding - expected_embedding))),
            'decoder_max_abs_errors': differences, 'models': str(args.model_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['Python', 'Torch', 'Packages', 'Notebook', 'Export'], required=True)
    parser.add_argument('--environment', choices=['sam31', 'export'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cu126'], required=True)
    parser.add_argument('--venv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    functions = {'Python': check_python, 'Torch': check_torch, 'Packages': check_packages,
                 'Notebook': check_notebook, 'Export': check_export}
    report = functions[args.stage](args)
    (args.output_dir / f'{args.stage.lower()}-check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
