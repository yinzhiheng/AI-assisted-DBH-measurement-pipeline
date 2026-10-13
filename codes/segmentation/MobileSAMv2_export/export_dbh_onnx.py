"""Export the supplied MobileSAMv2 TinyViT components for the unchanged DBH Sam class.

Reuses scripts/export_onnx_model.py and mobilesamv2.utils.onnx.SamOnnxModel.
The separate encoder export follows SAM_cpp/export_pre_model.py's uint8 interface.
No source files in codes/ or in the vendored model implementation are modified.
"""

import argparse
import hashlib
import importlib.util
import sys
import warnings
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

PROJECT = Path(__file__).resolve().parents[3]
MODEL = PROJECT / "model/MobileSAMv2"
REPO = MODEL / "source"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "MobileSAMv2"))

import onnx
import torch
from mobilesamv2 import sam_model_registry
from mobilesamv2.modeling.sam import Sam
from mobilesamv2.utils.onnx import SamOnnxModel


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_model():
    encoder_path = MODEL / "checkpoints/mobile_sam.pt"
    decoder_path = MODEL / "checkpoints/Prompt_guided_Mask_Decoder.pt"
    encoder = sam_model_registry["tiny_vit"](checkpoint=None)
    # This supplied checkpoint is a full SAM state dict, not {'model': encoder}.
    checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=True)
    prefix = "image_encoder."
    encoder_weights = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
    if not encoder_weights:
        raise ValueError("Expected the supplied full SAM checkpoint with image_encoder.* keys")
    encoder.load_state_dict(encoder_weights, strict=True)
    parts = sam_model_registry["PromptGuidedDecoder"](checkpoint=None)
    decoder = torch.load(decoder_path, map_location="cpu", weights_only=True)
    parts["PromtEncoder"].load_state_dict(decoder["PromtEncoder"], strict=True)
    parts["MaskDecoder"].load_state_dict(decoder["MaskDecoder"], strict=True)
    model = Sam(image_encoder=encoder, prompt_encoder=parts["PromtEncoder"],
                mask_decoder=parts["MaskDecoder"]).cpu().eval()
    return model, [encoder_path, decoder_path]


class ImageEncoderExport(torch.nn.Module):
    """C++ has already resized the image and converted BGR to planar RGB."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.image_encoder(self.model.preprocess(image.to(torch.float32)))


class MultimaskExport(SamOnnxModel):
    """Keep the upstream six-input/three-output contract and SamPredictor's 3 masks.

The referenced task uses multimask_output=True. Upstream ONNX export otherwise
returns all four tokens, including the separate single-mask token at index zero.
The unchanged C++ getMask() already selects the highest score for any mask count.
"""

    def forward(self, image_embeddings, point_coords, point_labels, mask_input,
                has_mask_input, orig_im_size):
        masks, scores, low_res = super().forward(
            image_embeddings, point_coords, point_labels, mask_input,
            has_mask_input, orig_im_size)
        return masks[:, 1:, :, :], scores[:, 1:], low_res[:, 1:, :, :]



def export_models(model, out):
    out.mkdir(parents=True, exist_ok=True)
    encoder_path = out / "mobile_sam_preprocess.onnx"
    decoder_path = out / "mobile_sam.onnx"
    print("Exporting the v2 TinyViT image encoder...", flush=True)
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        torch.onnx.export(ImageEncoderExport(model).eval(),
                          torch.zeros(1, 3, 720, 1024, dtype=torch.uint8),
                          str(encoder_path), input_names=["input"], output_names=["output"],
                          opset_version=16, do_constant_folding=True, export_params=True)
    # C++ allocates this output from model metadata before its first inference.
    graph = onnx.load(str(encoder_path))
    for dim, value in zip(graph.graph.output[0].type.tensor_type.shape.dim, [1, 256, 64, 64]):
        dim.ClearField("dim_param")
        dim.dim_value = value
    onnx.checker.check_model(graph)
    onnx.save(graph, str(encoder_path))

    # Execute the original official export routine with the v2 model and helper.
    # Rebinding this imported module affects only this export process, not its file.
    spec = importlib.util.spec_from_file_location("upstream_sam_export", REPO / "scripts/export_onnx_model.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.sam_model_registry = {"mobilesamv2_tiny_vit": lambda checkpoint: model}
    upstream.SamOnnxModel = MultimaskExport
    upstream.run_export("mobilesamv2_tiny_vit", "components-loaded-strictly", str(decoder_path),
                        opset=16, return_single_mask=False, gelu_approximate=False,
                        use_stability_score=False, return_extra_metrics=False)
    # The upstream wrapper starts in train mode; legacy torch.onnx.export restores
    # that mode recursively after exporting. References must run in eval mode.
    model.eval()
    for path in [encoder_path, decoder_path]:
        graph = onnx.load(str(path))
        onnx.helper.set_model_props(graph, {
            "deployment": "DBH C++ Sam interface; externally supplied prompts; no object finder",
            "implementation": "MobileSAMv2 TinyViT + separate PromptGuidedDecoder",
            "mask_selection": "SamPredictor multimask_output=True; C++ argmax(iou_predictions)",
            "encoder_checkpoint_sha256": sha256(MODEL / "checkpoints/mobile_sam.pt"),
            "decoder_checkpoint_sha256": sha256(MODEL / "checkpoints/Prompt_guided_Mask_Decoder.pt"),
        })
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))
    return encoder_path, decoder_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MODEL / "onnx")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, _ = build_model()
    paths = export_models(model, args.output_dir)
    print("Saved ONNX models:", *paths, sep="\n", flush=True)


if __name__ == "__main__":
    main()
