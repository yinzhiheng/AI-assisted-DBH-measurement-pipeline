"""SAM 3.1 single-image detector, using the actual 3.1 checkpoint on CPU.

Builds the detector exactly as build_sam3_multiplex_video_predictor does.
The temporal tracker and interactive click-refinement modules are not needed
for independent text-prompted images. No SAM 3 checkpoint is downloaded/used.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

SAM31_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_sam31(source: Path):
    # Set before importing torch/SAM: initialization caches must also stay on CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["USE_PERFLIB"] = "0"
    if not (source / "sam3" / "model_builder.py").is_file():
        raise FileNotFoundError(f"SAM source directory is missing: {source}")
    sys.path.insert(0, str(source.resolve()))
    import torch
    from sam3 import model_builder as builder
    actual = Path(builder.__file__).resolve()
    if actual.parent.parent != source.resolve():
        raise RuntimeError(f"Wrong SAM source imported: {actual}")
    if torch.cuda.is_available():
        raise RuntimeError("Run this CPU script in a fresh Python process; CUDA was already initialized.")
    if not hasattr(builder, "build_sam3_multiplex_video_predictor"):
        raise RuntimeError("This source checkout does not contain SAM 3.1.")
    return torch, builder


def complete_rotary_buffers(model, weights):
    """Derive real/imaginary positional caches from saved complex constants.

    The official checkpoint stores freqs_cis, whereas use_rope_real=True also
    registers these two deterministic buffers. They are not learned weights.
    Every learned parameter must still be present; strict loading follows.
    """
    derived = []
    for name, _ in model.named_buffers():
        if name in weights:
            continue
        if name.endswith(".freqs_cis_real"):
            base = name.removesuffix("_real")
            if base in weights and weights[base].is_complex():
                weights[name] = weights[base].real.contiguous()
                derived.append(name)
        elif name.endswith(".freqs_cis_imag"):
            base = name.removesuffix("_imag")
            if base in weights and weights[base].is_complex():
                weights[name] = weights[base].imag.contiguous()
                derived.append(name)
    return derived


def build_detector(source: Path, checkpoint: Path, threads: int):
    torch, b = import_sam31(source)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.vl_combiner import SAM3VLBackboneTri

    print("Checking the SAM 3.1 checkpoint fingerprint (outside image timing)...", flush=True)
    digest = file_sha256(checkpoint)
    if digest != SAM31_SHA256:
        raise ValueError(
            "Checkpoint does not match Meta's official sam3.1_multiplex.pt. "
            f"Expected {SAM31_SHA256}; got {digest}. "
            "The script will not silently fall back to SAM 3 or partially downloaded weights."
        )
    bpe = source / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe.is_file():
        raise FileNotFoundError(f"Tokenizer vocabulary is missing: {bpe}")

    # Same architecture/settings as the detector in the official SAM 3.1 builder.
    print("Building the SAM 3.1 detector on CPU...", flush=True)
    model = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=SAM3VLBackboneTri(
            scalp=0,
            visual=b._create_multiplex_tri_backbone(
                compile_mode=None, use_fa3=False, use_rope_real=True
            ),
            text=b._create_text_encoder(str(bpe)),
        ),
        transformer=b._create_sam3_transformer(use_fa3=False),
        segmentation_head=b._create_segmentation_head(use_fa3=False),
        semantic_segmentation_head=None,
        input_geometry_encoder=b._create_geometry_encoder(),
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=b._create_dot_product_scoring(),
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )
    # mmap avoids reading a second full copy of the 3.5 GB file into RAM.
    weights = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
    if "model" in weights and isinstance(weights["model"], dict):
        weights = weights["model"]
    detector_weights = {
        key.removeprefix("detector."): value
        for key, value in weights.items() if key.startswith("detector.")
    }
    if not detector_weights:
        raise ValueError("No detector weights found in the SAM 3.1 checkpoint.")
    checkpoint_tensor_count = len(detector_weights)
    derived_buffers = complete_rotary_buffers(model, detector_weights)
    # A missing/wrong tensor is an error, not an ignored warning.
    model.load_state_dict(detector_weights, strict=True)
    model.eval().requires_grad_(False)
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("The detector must be entirely CPU / FP32.")
    info = {
        "model": "SAM 3.1",
        "inference_path": "SAM 3.1 detector; independent text-prompted images",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "detector_checkpoint_tensors_loaded": checkpoint_tensor_count,
        "derived_rotary_buffer_count": len(derived_buffers),
        "derived_rotary_buffers": derived_buffers,
        "strict_weight_loading": True,
        "detector_parameters": sum(p.numel() for p in model.parameters()),
        "source": str(source.resolve()),
        "device": "cpu",
        "precision": "float32",
        "torch_version": str(torch.__version__),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }
    del detector_weights, weights
    return model, info


def forward_image(model, tensor, prompt: str):
    """Only neural-network work; no disk I/O, resizing, drawing, or saving.

    Recompute both image and text features per image so the reported time is
    standalone image inference, not cached-feature or prompt-only inference.
    """
    import torch
    from sam3.model.data_misc import FindStage
    with torch.inference_mode():
        backbone = model.backbone.forward_image(
            tensor, need_sam3_out=True,
            need_interactive_out=False, need_propagation_out=False,
        )
        backbone.update(model.backbone.forward_text([prompt], device="cpu"))
        stage = FindStage(
            img_ids=torch.tensor([0], dtype=torch.long),
            text_ids=torch.tensor([0], dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        out = model.forward_grounding(
            backbone_out=backbone, find_input=stage,
            find_target=None, geometric_prompt=model._get_dummy_prompt(),
        )
        return {key: out[key] for key in (
            "pred_masks", "pred_logits", "pred_boxes", "presence_logit_dec"
        )}
