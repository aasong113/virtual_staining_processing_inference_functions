"""
Crop all images in a folder to fixed pixel coordinates and save with a user-defined prefix.

Coordinates are interpreted as (x_start, y_start) inclusive and (x_end, y_end) exclusive
(i.e., Python slicing / PIL crop semantics).

Examples
--------
python crop_images_in_folder.py ^
  --input-dir "C:\\data\\BIT" ^
  --output-dir "C:\\data\\BIT_cropped" ^
  --x-start 100 --y-start 200 --x-end 1100 --y-end 1200 ^
  --prefix "cropped_"

To overwrite in-place (safer default is NOT in-place):
python crop_images_in_folder.py --input-dir "C:\\data\\BIT" --inplace --x-start 0 --y-start 0 --x-end 512 --y-end 512 --prefix "crop_"
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class CropBox:
    x_start: int
    y_start: int
    x_end: int
    y_end: int

    def validate(self) -> None:
        for name, value in (
            ("x_start", self.x_start),
            ("y_start", self.y_start),
            ("x_end", self.x_end),
            ("y_end", self.y_end),
        ):
            if not isinstance(value, int):
                raise ValueError(f"{name} must be an int, got {type(value)}")
        if self.x_end <= self.x_start:
            raise ValueError(f"x_end ({self.x_end}) must be > x_start ({self.x_start})")
        if self.y_end <= self.y_start:
            raise ValueError(f"y_end ({self.y_end}) must be > y_start ({self.y_start})")


def _iter_image_paths(input_dir: str, recursive: bool, exts: Tuple[str, ...]) -> Iterable[str]:
    if recursive:
        for root, _, files in os.walk(input_dir):
            for name in files:
                if name.lower().endswith(exts):
                    yield os.path.join(root, name)
        return
    for name in os.listdir(input_dir):
        if name.lower().endswith(exts):
            yield os.path.join(input_dir, name)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _crop_numpy_like(arr, crop: CropBox):
    crop.validate()
    x1, y1, x2, y2 = crop.x_start, crop.y_start, crop.x_end, crop.y_end

    # Clamp to bounds (consistent with existing helper in processing_functions.py).
    # We assume the last two dims are (Y, X) unless the array looks like (Y, X, C).
    if getattr(arr, "ndim", None) is None:
        raise TypeError("Expected a numpy-like array with .ndim/.shape attributes.")

    if arr.ndim == 2:
        height, width = arr.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        return arr[y1:y2, x1:x2]

    if arr.ndim == 3 and arr.shape[2] in (3, 4):  # (Y, X, C)
        height, width = arr.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        return arr[y1:y2, x1:x2, :]

    # General case: (..., Y, X)
    height, width = arr.shape[-2], arr.shape[-1]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    slicer = (slice(None),) * (arr.ndim - 2) + (slice(y1, y2), slice(x1, x2))
    return arr[slicer]


def _is_tiff(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".tif") or lower.endswith(".tiff")


def _crop_and_save_one(
    in_path: str,
    out_path: str,
    crop: CropBox,
) -> None:
    if _is_tiff(in_path):
        try:
            import tifffile  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency 'tifffile' for TIFF support.") from e

        arr = tifffile.imread(in_path)
        cropped = _crop_numpy_like(arr, crop)
        tifffile.imwrite(out_path, cropped)
        return

    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'Pillow' (PIL) for image support.") from e

    with Image.open(in_path) as im:
        im = ImageOps.exif_transpose(im)
        crop.validate()
        cropped = im.crop((crop.x_start, crop.y_start, crop.x_end, crop.y_end))
        cropped.save(out_path)


def _make_output_path(
    in_path: str,
    input_dir: str,
    output_dir: str,
    prefix: str,
    recursive: bool,
) -> str:
    if not recursive:
        return os.path.join(output_dir, f"{prefix}{os.path.basename(in_path)}")

    # Preserve relative subfolders under output_dir when walking recursively.
    rel = os.path.relpath(in_path, input_dir)
    rel_dir = os.path.dirname(rel)
    out_dir = os.path.join(output_dir, rel_dir)
    _ensure_dir(out_dir)
    return os.path.join(out_dir, f"{prefix}{os.path.basename(in_path)}")


def crop_images_in_folder(
    input_dir: str,
    *,
    x_start: int,
    y_start: int,
    x_end: int,
    y_end: int,
    prefix: str,
    output_dir: Optional[str] = None,
    inplace: bool = False,
    recursive: bool = False,
    dry_run: bool = False,
    extensions: Tuple[str, ...] = IMAGE_EXTS,
    stop_on_error: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Crop all images in a folder to fixed pixel coordinates and save with a user-defined prefix.

    Parameters
    ----------
    input_dir:
        Folder containing images to crop.
    x_start, y_start, x_end, y_end:
        Pixel coordinates. Interpreted as (x_start, y_start) inclusive and (x_end, y_end) exclusive.
    prefix:
        String to prepend to each output filename.
    output_dir:
        Where to write cropped images. If None and not inplace, defaults to "<input_dir>_cropped".
    inplace:
        Overwrite images in input_dir. (Ignored if output_dir is provided.)
    recursive:
        Recurse into subfolders. If True, preserves relative subfolders under output_dir.
    dry_run:
        If True, prints planned actions but does not write files.
    extensions:
        File extensions to include (case-insensitive). Default includes png/jpg/tif/tiff/bmp.
    stop_on_error:
        If True, re-raises on first error. If False, continues and counts failures.
    verbose:
        If True, prints progress and summary.

    Returns
    -------
    dict with keys:
        - output_dir (str)
        - num_ok (int)
        - num_fail (int)
        - total (int)
    """
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"input_dir is not a directory: {input_dir}")

    if inplace and output_dir is not None:
        raise ValueError("Use either inplace=True OR output_dir=..., not both.")

    resolved_output_dir = input_dir if inplace else os.path.abspath(output_dir or (input_dir + "_cropped"))
    if not inplace:
        _ensure_dir(resolved_output_dir)

    crop = CropBox(x_start, y_start, x_end, y_end)
    crop.validate()

    exts = tuple(e.lower() for e in extensions)
    paths = list(_iter_image_paths(input_dir, recursive=recursive, exts=exts))
    if not paths:
        if verbose:
            print(f"No images found in {input_dir} with extensions: {', '.join(exts)}")
        return {"output_dir": resolved_output_dir, "num_ok": 0, "num_fail": 0, "total": 0}

    num_ok = 0
    num_fail = 0
    for in_path in paths:
        try:
            out_path = _make_output_path(
                in_path=in_path,
                input_dir=input_dir,
                output_dir=resolved_output_dir,
                prefix=prefix,
                recursive=recursive,
            )
            if dry_run:
                if verbose:
                    print(f"[DRY RUN] {in_path} -> {out_path}")
                num_ok += 1
                continue

            out_parent = os.path.dirname(out_path)
            if out_parent and not os.path.isdir(out_parent):
                _ensure_dir(out_parent)

            if inplace:
                # Avoid clobbering on partial failures: write temp then replace.
                tmp_path = out_path + ".tmp"
                _crop_and_save_one(in_path, tmp_path, crop)
                os.replace(tmp_path, out_path)
            else:
                _crop_and_save_one(in_path, out_path, crop)

            num_ok += 1
        except Exception as e:
            num_fail += 1
            if verbose:
                print(f"[ERROR] {in_path}: {e}")
            if stop_on_error:
                raise

    if verbose:
        print(f"Done. Cropped {num_ok} file(s); {num_fail} failed.")
        if not inplace:
            print(f"Output folder: {resolved_output_dir}")
    return {"output_dir": resolved_output_dir, "num_ok": num_ok, "num_fail": num_fail, "total": len(paths)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Crop all images in a folder to fixed coordinates.")
    p.add_argument("--input-dir", required=True, help="Folder containing images to crop.")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write cropped images (default: <input-dir>_cropped).",
    )
    p.add_argument("--inplace", action="store_true", help="Overwrite images in input-dir (dangerous).")
    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    p.add_argument("--prefix", required=True, help="String to prepend to each output filename.")
    p.add_argument("--x-start", type=int, required=True)
    p.add_argument("--y-start", type=int, required=True)
    p.add_argument("--x-end", type=int, required=True)
    p.add_argument("--y-end", type=int, required=True)
    p.add_argument("--dry-run", action="store_true", help="Print planned actions but do not write files.")
    args = p.parse_args(argv)

    result = crop_images_in_folder(
        args.input_dir,
        x_start=args.x_start,
        y_start=args.y_start,
        x_end=args.x_end,
        y_end=args.y_end,
        prefix=args.prefix,
        output_dir=args.output_dir,
        inplace=args.inplace,
        recursive=args.recursive,
        dry_run=args.dry_run,
        stop_on_error=False,
        verbose=True,
    )
    return 0 if int(result["num_fail"]) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
