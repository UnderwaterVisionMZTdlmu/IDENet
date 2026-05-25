#!/usr/bin/env python3
# coding=utf-8
"""Batch inference for the local IDENet RGB-D saliency model."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from IDENet import IDENet


ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_resample_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def rgb_to_tensor(path, size):
    image = Image.open(path).convert("RGB")
    original_size = image.size
    image = image.resize((size, size), get_resample_filter())

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, original_size


def depth_to_tensor(path, size):
    depth = Image.open(path)
    depth = depth.resize((size, size), get_resample_filter())
    array = np.asarray(depth, dtype=np.float32)

    if array.ndim == 3:
        array = array[..., 0]

    min_value = float(array.min())
    max_value = float(array.max())
    if max_value > min_value:
        array = (array - min_value) / (max_value - min_value)
    else:
        array = np.zeros_like(array, dtype=np.float32)

    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).float()
    return tensor


def load_model(checkpoint_path, device):
    model = IDENet().to(device)
    try:
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=device)
    except RuntimeError as exc:
        message = str(exc)
        if "failed finding central directory" in message:
            raise RuntimeError(
                f"error loading checkpoint: {checkpoint_path}\n"
            ) from exc
        raise
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def saliency_to_image(saliency, original_size):
    saliency = torch.sigmoid(saliency)
    saliency = F.interpolate(
        saliency,
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=True,
    )

    array = saliency.squeeze().detach().cpu().numpy()
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value > min_value:
        array = (array - min_value) / (max_value - min_value)
    else:
        array = np.zeros_like(array, dtype=np.float32)

    return Image.fromarray((array * 255).astype(np.uint8))


def iter_rgb_images(rgb_dir):
    for path in sorted(rgb_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def run_inference(args):
    rgb_dir = Path(args.rgb_dir)
    depth_dir = Path(args.depth_dir)
    output_dir = Path(args.output_dir)
    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)

    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint, device)

    total = 0
    skipped = 0
    with torch.no_grad():
        for rgb_path in iter_rgb_images(rgb_dir):
            depth_path = depth_dir / rgb_path.name
            if not depth_path.exists():
                skipped += 1
                print(f"Skip {rgb_path.name}: matching depth image not found.")
                continue

            rgb_tensor, original_size = rgb_to_tensor(rgb_path, args.size)
            depth_tensor = depth_to_tensor(depth_path, args.size)
            rgb_tensor = rgb_tensor.to(device)
            depth_tensor = depth_tensor.to(device)

            s1, *_ = model(rgb_tensor, depth_tensor, shape=(args.size, args.size))
            result = saliency_to_image(s1, original_size)

            output_path = output_dir / f"{rgb_path.stem}.png"
            result.save(output_path)
            total += 1
            print(f"Saved {output_path}")

    print(f"Done. Processed: {total}, skipped: {skipped}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for IDENet.")
    parser.add_argument("--rgb_dir", default=str(ROOT / "rgb"), help="RGB image directory.")
    parser.add_argument("--depth_dir", default=str(ROOT / "depth"), help="Depth image directory.")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "IDENet_usod10k_epoch_best_224.pth"),
        help="IDENet state_dict checkpoint.",
    )
    parser.add_argument("--output_dir", default=str(ROOT / "outputs"), help="Output directory.")
    parser.add_argument("--size", type=int, default=224, help="Square inference size.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, such as cuda or cpu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
