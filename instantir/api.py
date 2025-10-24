from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
from diffusers import DDPMScheduler
from huggingface_hub import hf_hub_download
from PIL import Image

from .module.ip_adapter.utils import load_adapter_to_pipe
from .pipelines.sdxl_instantir import InstantIRPipeline
from .schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler

Prompt = Union[str, Sequence[str]]
ImageInput = Union[str, Path, Image.Image]

_DEFAULT_PROMPT = (
    "Photorealistic, highly detailed, hyper detailed photo - realistic maximum detail, "
    "32k, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, "
    "perfect without deformations, taken using a Canon EOS R camera, Cinematic, High Contrast, "
    "Color Grading."
)

_DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, unclear, depth of field, over-smooth, sketch, oil painting, cartoon, "
    "CG Style, 3D render, unreal engine, dirty, messy, worst quality, low quality, frames, "
    "painting, illustration, drawing, art, watermark, signature, jpeg artifacts, deformed, lowres"
)


@dataclass
class InstantIRRuntime:
    """Container for the InstantIR runtime state."""

    pipe: InstantIRPipeline
    scheduler: LCMSingleStepScheduler
    device: torch.device
    torch_dtype: torch.dtype

    def __post_init__(self) -> None:
        self.pipe.to(device=self.device, dtype=self.torch_dtype)
        self.pipe.aggregator.to(device=self.device, dtype=self.torch_dtype)


def _ensure_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _download_weight_if_missing(
    target: Path,
    *,
    repo_id: str,
    filename: str,
    remote_filename: Optional[str] = None,
) -> Path:
    """Download a required InstantIR weight from Hugging Face if it's not already on disk."""

    target = Path(target)
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    remote_name = remote_filename or filename
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_name,
            local_dir=str(target.parent),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download '{filename}' from '{repo_id}'. Please verify network access and repository availability."
        ) from exc

    downloaded_path = Path(downloaded_path)
    if downloaded_path.resolve() != target.resolve():
        shutil.copy2(downloaded_path, target)
        return target

    return downloaded_path


def _make_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resize_image(
    input_image: Image.Image,
    *,
    max_side: int = 1024,
    min_side: int = 768,
    width: Optional[int] = None,
    height: Optional[int] = None,
    pad_to_max_side: bool = False,
    mode: int = Image.BILINEAR,
    base_pixel_number: int = 64,
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Resize image for Stable Diffusion pipeline compat."""

    w, h = input_image.size
    # Prepare output size
    if width is not None and height is not None:
        out_w, out_h = width, height
    elif width is not None:
        out_w = width
        out_h = round(h * width / w)
    elif height is not None:
        out_h = height
        out_w = round(w * height / h)
    else:
        out_w, out_h = w, h

    w, h = out_w, out_h
    if min(w, h) < min_side:
        ratio = min_side / min(w, h)
        w, h = round(ratio * w), round(ratio * h)
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        w, h = round(ratio * w), round(ratio * h)

    w_resize_new = max(base_pixel_number, (w // base_pixel_number) * base_pixel_number)
    h_resize_new = max(base_pixel_number, (h // base_pixel_number) * base_pixel_number)
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = Image.new("RGB", (max_side, max_side), (255, 255, 255))
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res.paste(input_image, (offset_x, offset_y))
        input_image = res

    return input_image, (out_w, out_h)


def load_runtime(
    *,
    instantir_path: Union[str, Path],
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    image_encoder_or_path: str = "facebook/dinov2-large",
    adapter_path: Optional[Union[str, Path]] = None,
    previewer_lora_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    torch_dtype: torch.dtype = torch.float16,
    use_clip_encoder: bool = False,
    map_location: Union[str, torch.device] = "cpu",
) -> InstantIRRuntime:
    """Load the InstantIR pipeline and keep it ready for inference."""

    device = _make_device(device)
    instantir_path = Path(instantir_path).expanduser()
    instantir_path.mkdir(parents=True, exist_ok=True)

    repo_id = "InstantX/InstantIR"
    remote_prefix = "models"
    adapter_filename = "adapter.pt"
    aggregator_filename = "aggregator.pt"
    previewer_filename = "previewer_lora_weights.bin"

    pipe = InstantIRPipeline.from_pretrained(
        sdxl_model_id,
        torch_dtype=torch_dtype,
    )

    adapter_user_provided = adapter_path is not None
    adapter_path = Path(adapter_path) if adapter_path is not None else instantir_path / adapter_filename
    if adapter_path.is_dir():
        adapter_path = adapter_path / adapter_filename
    if adapter_user_provided:
        adapter_path = _ensure_path(adapter_path)
    else:
        adapter_path = _download_weight_if_missing(
            adapter_path,
            repo_id=repo_id,
            filename=adapter_filename,
            remote_filename=f"{remote_prefix}/{adapter_filename}",
        )
    load_adapter_to_pipe(
        pipe,
        str(adapter_path),
        image_encoder_or_path,
        use_clip_encoder=use_clip_encoder,
    )

    previewer_user_provided = previewer_lora_path is not None
    previewer_resolved = Path(previewer_lora_path) if previewer_lora_path is not None else instantir_path
    if previewer_resolved.suffix and not previewer_resolved.is_dir():
        previewer_weights_path = previewer_resolved
        previewer_dir = previewer_weights_path.parent
    else:
        previewer_dir = previewer_resolved
        previewer_weights_path = previewer_dir / previewer_filename
    previewer_dir.mkdir(parents=True, exist_ok=True)
    if previewer_user_provided:
        if not previewer_weights_path.exists():
            raise FileNotFoundError(f"Missing previewer weights at {previewer_weights_path}")
    else:
        previewer_weights_path = _download_weight_if_missing(
            previewer_weights_path,
            repo_id=repo_id,
            filename=previewer_filename,
            remote_filename=f"{remote_prefix}/{previewer_filename}",
        )
    lora_alpha = pipe.prepare_previewers(str(previewer_dir))
    if lora_alpha is not None:
        print(f"Loaded previewer LoRA with alpha={lora_alpha}")

    pipe.scheduler = DDPMScheduler.from_pretrained(sdxl_model_id, subfolder="scheduler")
    scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)

    aggregator_path = instantir_path / aggregator_filename
    aggregator_path = _download_weight_if_missing(
        aggregator_path,
        repo_id=repo_id,
        filename=aggregator_filename,
        remote_filename=f"{remote_prefix}/{aggregator_filename}",
    )
    if not aggregator_path.exists():
        raise FileNotFoundError(f"Missing aggregator weights at {aggregator_path}")
    state_dict = torch.load(str(aggregator_path), map_location=map_location)
    pipe.aggregator.load_state_dict(state_dict)

    return InstantIRRuntime(pipe=pipe, scheduler=scheduler, device=device, torch_dtype=torch_dtype)


def restore_image(
    runtime: InstantIRRuntime,
    image: ImageInput,
    *,
    prompt: Prompt = _DEFAULT_PROMPT,
    negative_prompt: Prompt = _DEFAULT_NEGATIVE_PROMPT,
    num_inference_steps: int = 30,
    seed: Optional[int] = 42,
    cfg: float = 7.0,
    preview_start: float = 0.0,
    creative_start: float = 1.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Image.Image:
    """Restore a single image using a warm InstantIR runtime."""

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    resized_image, original_size = _resize_image(image, width=width, height=height)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=runtime.device).manual_seed(seed)

    if prompt is None:
        prompt = _DEFAULT_PROMPT
    if negative_prompt is None:
        negative_prompt = _DEFAULT_NEGATIVE_PROMPT

    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    if len(prompts) != 1:
        raise ValueError("`prompt` must be a string or a single-item sequence when restoring one image.")

    neg_prompts = [negative_prompt] if isinstance(negative_prompt, str) else list(negative_prompt)
    if len(neg_prompts) != 1:
        raise ValueError("`negative_prompt` must be a string or a single-item sequence when restoring one image.")

    prompt_input = prompts[0] if len(prompts) == 1 else prompts
    neg_prompt_input = neg_prompts[0] if len(neg_prompts) == 1 else neg_prompts

    result = runtime.pipe(
        prompt=prompt_input,
        image=[resized_image],
        num_inference_steps=num_inference_steps,
        generator=generator,
        negative_prompt=neg_prompt_input,
        guidance_scale=cfg,
        previewer_scheduler=runtime.scheduler,
        preview_start=preview_start,
        control_guidance_end=creative_start,
    ).images[0]

    return result.resize(original_size, Image.BILINEAR)


def restore_images_in_directory(
    runtime: InstantIRRuntime,
    *,
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    glob_patterns: Iterable[str] = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"),
    overwrite: bool = True,
    **restore_kwargs,
) -> List[Path]:
    """Restore all images found in ``input_dir`` and save them to ``output_dir``."""

    input_dir = _ensure_path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    restored_paths: List[Path] = []
    files: List[Path] = []
    for pattern in glob_patterns:
        files.extend(sorted(input_dir.glob(pattern)))

    if not files:
        print(f"No images found in {input_dir}")
        return restored_paths

    for image_path in files:
        output_path = output_dir / image_path.name
        if output_path.exists() and not overwrite:
            restored_paths.append(output_path)
            continue
        restored = restore_image(runtime, image_path, **restore_kwargs)
        restored.save(output_path)
        restored_paths.append(output_path)

    return restored_paths
