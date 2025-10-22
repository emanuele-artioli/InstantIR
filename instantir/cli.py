import argparse
from typing import Optional

from .api import load_runtime, restore_images_in_directory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InstantIR image restoration")
    parser.add_argument("--instantir-path", required=True, help="Path containing InstantIR weights (adapter.pt, aggregator.pt, etc.)")
    parser.add_argument("--sdxl-model-id", default="stabilityai/stable-diffusion-xl-base-1.0", help="Base SDXL model identifier")
    parser.add_argument("--image-encoder", default="facebook/dinov2-large", help="Image encoder identifier or local path")
    parser.add_argument("--adapter-path", default=None, help="Optional path to adapter weights")
    parser.add_argument("--previewer-lora-path", default=None, help="Optional path to previewer LoRA weights")
    parser.add_argument("--device", default=None, help="Device to run on (cuda, cuda:0, cpu)")
    parser.add_argument("--prompt", default=None, help="Prompt for creative restoration")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt for restoration")
    parser.add_argument("--num-steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=7.0, help="Classifier-free guidance scale")
    parser.add_argument("--preview-start", type=float, default=0.0, help="Proportion of timesteps to skip previewing")
    parser.add_argument("--creative-start", type=float, default=1.0, help="Late diffusion portion for creative restoration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed; set negative value to disable seeding")
    parser.add_argument("--width", type=int, default=None, help="Optional target width")
    parser.add_argument("--height", type=int, default=None, help="Optional target height")
    parser.add_argument("--input-dir", required=True, help="Directory containing low-quality images")
    parser.add_argument("--output-dir", required=True, help="Directory to save restored images")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    runtime = load_runtime(
        instantir_path=args.instantir_path,
        sdxl_model_id=args.sdxl_model_id,
        image_encoder_or_path=args.image_encoder,
        adapter_path=args.adapter_path,
        previewer_lora_path=args.previewer_lora_path,
        device=args.device,
    )

    prompt = args.prompt
    neg_prompt = args.negative_prompt
    seed = args.seed if args.seed >= 0 else None

    restore_images_in_directory(
        runtime,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt=prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=args.num_steps,
        cfg=args.cfg,
        preview_start=args.preview_start,
        creative_start=args.creative_start,
        seed=seed,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
