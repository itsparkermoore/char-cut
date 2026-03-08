#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image


INPUT_PATH = Path("hair.png")
OUTPUT_NO_AA = Path("hair_no_aa.png")
OUTPUT_AA = Path("hair_aa.png")


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb.astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    mx = np.max(rgb, axis=2)
    mn = np.min(rgb, axis=2)
    delta = mx - mn

    hue = np.zeros_like(mx)
    nonzero = delta > 1e-6

    r_max = (mx == r) & nonzero
    g_max = (mx == g) & nonzero
    b_max = (mx == b) & nonzero

    hue[r_max] = ((g[r_max] - b[r_max]) / delta[r_max]) % 6.0
    hue[g_max] = ((b[g_max] - r[g_max]) / delta[g_max]) + 2.0
    hue[b_max] = ((r[b_max] - g[b_max]) / delta[b_max]) + 4.0
    hue *= 60.0

    sat = np.zeros_like(mx)
    value_nonzero = mx > 1e-6
    sat[value_nonzero] = delta[value_nonzero] / mx[value_nonzero]

    return hue, sat, mx


def circular_distance_deg(values: np.ndarray, center: float) -> np.ndarray:
    delta = np.abs(values - center)
    return np.minimum(delta, 360.0 - delta)


def circular_mean_deg(values: np.ndarray) -> float:
    radians = np.deg2rad(values)
    angle = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return float(np.rad2deg(angle) % 360.0)


def smooth_histogram(histogram: np.ndarray) -> np.ndarray:
    padded = np.pad(histogram, ((1, 1), (1, 1)), mode="edge")
    return (
        padded[:-2, :-2]
        + 2.0 * padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + 2.0 * padded[1:-1, :-2]
        + 4.0 * padded[1:-1, 1:-1]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 16.0


def smooth_histogram_1d(histogram: np.ndarray) -> np.ndarray:
    padded = np.pad(histogram, 1, mode="edge")
    return (
        padded[:-2]
        + 2.0 * padded[1:-1]
        + padded[2:]
    ) / 4.0


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="edge")
        neighbors = [
            padded[y : y + out.shape[0], x : x + out.shape[1]]
            for y in range(3)
            for x in range(3)
        ]
        out = np.logical_or.reduce(neighbors)
    return out


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="edge")
        neighbors = [
            padded[y : y + out.shape[0], x : x + out.shape[1]]
            for y in range(3)
            for x in range(3)
        ]
        out = np.logical_and.reduce(neighbors)
    return out


def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    opened = dilate(erode(mask, 1), 1)
    closed = erode(dilate(opened, 1), 1)
    return closed


def blur_mask(mask: np.ndarray, passes: int = 2) -> np.ndarray:
    out = mask.astype(np.float32)
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32)
    kernel /= kernel.sum()

    for _ in range(passes):
        padded_x = np.pad(out, ((0, 0), (2, 2)), mode="edge")
        out = sum(
            weight * padded_x[:, offset : offset + out.shape[1]]
            for offset, weight in enumerate(kernel)
        )

        padded_y = np.pad(out, ((2, 2), (0, 0)), mode="edge")
        out = sum(
            weight * padded_y[offset : offset + out.shape[0], :]
            for offset, weight in enumerate(kernel)
        )

    return np.clip(out, 0.0, 1.0)


def detect_saturated_pixels(
    hue: np.ndarray, sat: np.ndarray, value: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    valid = alpha > 0
    analyzable = valid & (value > 0.09)
    if not np.any(analyzable):
        raise RuntimeError("No visible pixels were found in hair.png.")

    weights = sat[analyzable] * (0.35 + value[analyzable])
    sat_histogram, sat_edges = np.histogram(
        sat[analyzable],
        bins=32,
        range=(0.0, 1.0),
        weights=weights,
    )
    sat_histogram = smooth_histogram_1d(sat_histogram)
    sat_floor_index = np.searchsorted(sat_edges[:-1], 0.25, side="left")
    sat_peak_index = sat_floor_index + int(np.argmax(sat_histogram[sat_floor_index:]))
    sat_peak = float((sat_edges[sat_peak_index] + sat_edges[sat_peak_index + 1]) * 0.5)

    local_saturation = analyzable & (np.abs(sat - sat_peak) <= 0.154)
    if np.count_nonzero(local_saturation) < 250:
        local_saturation = analyzable & (np.abs(sat - sat_peak) <= 0.242)

    strict_sat_min = float(np.clip(np.percentile(sat[local_saturation], 12) - 0.033, 0.225, 0.95))
    sat_min = float(max(0.0, strict_sat_min * 0.81))
    high_sat_mask = analyzable & (sat >= sat_min)
    return high_sat_mask, {"sat_peak": sat_peak, "sat_min": sat_min, "strict_sat_min": strict_sat_min}


def detect_green_cluster(
    hue: np.ndarray, sat: np.ndarray, value: np.ndarray, high_sat_mask: np.ndarray
) -> dict[str, float]:
    if not np.any(high_sat_mask):
        raise RuntimeError("No highly saturated pixels were found in hair.png.")

    weights = (sat[high_sat_mask] ** 2.0) * (0.35 + value[high_sat_mask])
    histogram, hue_edges, sat_edges = np.histogram2d(
        hue[high_sat_mask],
        sat[high_sat_mask],
        bins=(72, 32),
        range=((0.0, 360.0), (0.0, 1.0)),
        weights=weights,
    )

    histogram = smooth_histogram(histogram)
    hue_centers = (hue_edges[:-1] + hue_edges[1:]) * 0.5
    green_band = (hue_centers >= 60.0) & (hue_centers <= 170.0)
    histogram[~green_band, :] = 0.0

    if not np.any(histogram):
        raise RuntimeError("No green cluster was found inside the saturated pixels.")

    peak_hue_idx, peak_sat_idx = np.unravel_index(np.argmax(histogram), histogram.shape)
    peak_hue = float((hue_edges[peak_hue_idx] + hue_edges[peak_hue_idx + 1]) * 0.5)
    peak_sat = float((sat_edges[peak_sat_idx] + sat_edges[peak_sat_idx + 1]) * 0.5)

    local_cluster = high_sat_mask & (circular_distance_deg(hue, peak_hue) <= 15.4) & (
        np.abs(sat - peak_sat) <= 0.176
    )
    if np.count_nonzero(local_cluster) < 250:
        local_cluster = high_sat_mask & (circular_distance_deg(hue, peak_hue) <= 24.2) & (
            np.abs(sat - peak_sat) <= 0.264
        )

    hue_center = circular_mean_deg(hue[local_cluster])
    hue_width = float(
        np.clip(np.percentile(circular_distance_deg(hue[local_cluster], hue_center), 98) + 4.4, 9.0, 46.0)
    )
    sat_min = float(np.clip(np.percentile(sat[local_cluster], 8) - 0.044, 0.25, 0.95))
    green_margin = value[local_cluster]

    return {
        "peak_hue": peak_hue,
        "peak_sat": peak_sat,
        "hue_center": hue_center,
        "hue_width": hue_width,
        "sat_min": sat_min,
        "value_min": float(np.clip(np.percentile(green_margin, 2) - 0.088, 0.04, 0.95)),
    }


def build_masks(
    rgb: np.ndarray,
    hue: np.ndarray,
    sat: np.ndarray,
    value: np.ndarray,
    alpha: np.ndarray,
    high_sat_mask: np.ndarray,
    cluster: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    r = rgb[..., 0].astype(np.float32) / 255.0
    g = rgb[..., 1].astype(np.float32) / 255.0
    b = rgb[..., 2].astype(np.float32) / 255.0
    max_rb = np.maximum(r, b)
    hue_distance = circular_distance_deg(hue, cluster["hue_center"])
    valid = alpha > 0
    thresholds = {
        "hard_sat_min": float(max(0.0, cluster["sat_min"] * 0.81)),
        "hard_value_min": float(max(0.0, cluster["value_min"] * 0.81)),
        "hard_hue_width": float(min(46.2, cluster["hue_width"] * 1.21)),
        "soft_sat_min": float(max(0.05, cluster["sat_min"] * 0.738)),
        "soft_value_min": float(max(0.02, cluster["value_min"] * 0.756)),
        "soft_hue_width": float(min(57.2, cluster["hue_width"] * 1.375)),
    }

    hard_mask = (
        high_sat_mask
        &
        valid
        & (sat >= thresholds["hard_sat_min"])
        & (value >= thresholds["hard_value_min"])
        & (hue_distance <= thresholds["hard_hue_width"])
        & (g >= r * 1.027)
        & (g >= b * 1.018)
        & ((g - max_rb) >= 0.018)
    )

    soft_mask = (
        valid
        & (sat >= thresholds["soft_sat_min"])
        & (value >= thresholds["soft_value_min"])
        & (hue_distance <= thresholds["soft_hue_width"])
        & (g >= max_rb - 0.011)
    )

    return hard_mask, soft_mask, thresholds


def apply_binary_alpha(alpha: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = alpha.copy()
    output[mask] = 0
    return output


def apply_antialiased_alpha(alpha: np.ndarray, hard_mask: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    expanded = dilate(hard_mask, 2)
    feather_zone = expanded & soft_mask

    matte = hard_mask.astype(np.float32)
    matte[(feather_zone & ~hard_mask)] = 0.72
    matte = blur_mask(matte, passes=1)
    matte[hard_mask] = 1.0
    return np.clip(alpha.astype(np.float32) * (1.0 - matte), 0.0, 255.0).astype(np.uint8)


def despill_edges(rgb: np.ndarray, matte: np.ndarray) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    r = output[..., 0]
    g = output[..., 1]
    b = output[..., 2]

    edge_strength = np.clip(matte * (1.0 - matte) * 4.0, 0.0, 1.0)
    max_rb = np.maximum(r, b)
    green_excess = np.maximum(g - max_rb, 0.0)
    g -= green_excess * edge_strength * 0.95

    output[..., 1] = g
    return np.clip(output, 0.0, 255.0).astype(np.uint8)


def save_rgba(rgb: np.ndarray, alpha: np.ndarray, destination: Path) -> None:
    rgba = np.dstack([rgb, alpha])
    Image.fromarray(rgba, "RGBA").save(destination)


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"Missing input image: {INPUT_PATH}", file=sys.stderr)
        return 1

    image = Image.open(INPUT_PATH).convert("RGBA")
    rgba = np.array(image)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]

    hue, sat, value = rgb_to_hsv(rgb)
    high_sat_mask, saturation_info = detect_saturated_pixels(hue, sat, value, alpha)
    cluster = detect_green_cluster(hue, sat, value, high_sat_mask)
    hard_mask, soft_mask, thresholds = build_masks(rgb, hue, sat, value, alpha, high_sat_mask, cluster)

    no_aa_alpha = apply_binary_alpha(alpha, hard_mask)
    save_rgba(rgb, no_aa_alpha, OUTPUT_NO_AA)

    aa_alpha = apply_antialiased_alpha(alpha, hard_mask, soft_mask)
    matte = 1.0 - (aa_alpha.astype(np.float32) / np.clip(alpha.astype(np.float32), 1.0, 255.0))
    aa_rgb = despill_edges(rgb, matte)
    save_rgba(aa_rgb, aa_alpha, OUTPUT_AA)

    print(
        "Detected saturated region:",
        f"sat_peak={saturation_info['sat_peak']:.2f},",
        f"strict_sat>={saturation_info['strict_sat_min']:.2f},",
        f"adjusted_sat>={saturation_info['sat_min']:.2f}",
    )
    print(
        "Refined green cluster:",
        f"hue={cluster['hue_center']:.1f} +/- {cluster['hue_width']:.1f} deg,",
        f"sat>={cluster['sat_min']:.2f},",
        f"value>={cluster['value_min']:.2f}",
    )
    print(
        "Adjusted removal mask:",
        f"hue+/-{thresholds['hard_hue_width']:.1f} deg,",
        f"sat>={thresholds['hard_sat_min']:.2f},",
        f"value>={thresholds['hard_value_min']:.2f}",
    )
    print(f"Saved {OUTPUT_NO_AA}")
    print(f"Saved {OUTPUT_AA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
