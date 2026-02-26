# flir_wrapper.py
"""
Stable wrapper API around ITVRoC/FlirImageExtractor.

Goal:
- Provide a single import/function that Colab (and other scripts) can rely on:
    temps_c = extract_temp_c(jpg_path)
- Keeps notebook stable even if upstream internals change.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

#  README usage example:
#   import flir_image_extractor
#   fir = flir_image_extractor.FlirImageExtractor()
import flir_image_extractor


def extract_temp_c(
    jpg_path: str | Path,
    exiftool_path: str = "exiftool",
) -> np.ndarray:
    """
    Extract per-pixel temperatures (Celsius) as a 2D numpy array from a FLIR JPEG.

    Parameters
    ----------
    jpg_path : str | Path
        Path to the FLIR .jpg captured by FLIR ONE (or other FLIR cameras).
    exiftool_path : str
        Path/name for exiftool binary (default: "exiftool").
        In Colab it will be installed system-wide, so "exiftool" works.

    Returns
    -------
    np.ndarray
        2D array (H, W) of temperature values in Celsius.
    """
    jpg_path = Path(jpg_path)
    if not jpg_path.exists():
        raise FileNotFoundError(f"Image not found: {jpg_path}")

    fir = flir_image_extractor.FlirImageExtractor()
    # Most forks allow specifying exiftool path; if your fork’s class uses a different attribute, update it here.
    if hasattr(fir, "exiftool_path"):
        fir.exiftool_path = exiftool_path

    fir.process_image(str(jpg_path))

    # Common method in repo get_thermal_np().
    if hasattr(fir, "get_thermal_np"):
        temps_c = fir.get_thermal_np()
    else:
        # Fallback: if upstream changes method names, fix it here.
        raise AttributeError(
            "Upstream FlirImageExtractor object does not have get_thermal_np(). "
            "Update wrapper to match upstream API."
        )

    temps_c = np.asarray(temps_c)
    if temps_c.ndim != 2:
        raise ValueError(f"Expected 2D thermal array, got shape {temps_c.shape}")

    return temps_c


if __name__ == "__main__":
    # Simple manual test:
    # python flir_wrapper.py path/to/image.jpg
    import sys
    if len(sys.argv) < 2:
        print("Usage: python flir_wrapper.py /path/to/flir.jpg")
        raise SystemExit(2)
    arr = extract_temp_c(sys.argv[1])
    print("OK:", arr.shape, "min/max:", float(arr.min()), float(arr.max()))
