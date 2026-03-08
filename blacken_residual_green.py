#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image


INPUT_PATH = Path("hair_aa.png")
OUTPUT_PATH = Path("hair_aa_blackened.png")


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


def detect_residual_green_cluster(
    rgb: np.ndarray, alpha: np.ndarray
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    hue, sat, value = rgb_to_hsv(rgb)
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    max_rb = np.maximum(r, b)
    green_excess = g - max_rb

    visible = alpha > 0
    candidate = visible & (value > 0.02) & (green_excess >= 1.0)
    if not np.any(candidate):
        raise RuntimeError("No residual green candidates were found in hair_aa.png.")

    histogram_mask = candidate & (hue >= 55.0) & (hue <= 165.0)
    if not np.any(histogram_mask):
        histogram_mask = candidate

    clipped_excess = np.clip(green_excess[histogram_mask], 0.0, 128.0)
    weights = ((clipped_excess + 1.0) ** 1.4) * (0.20 + sat[histogram_mask]) * (0.25 + value[histogram_mask])
    histogram, hue_edges, excess_edges = np.histogram2d(
        hue[histogram_mask],
        clipped_excess,
        bins=(72, 32),
        range=((0.0, 360.0), (0.0, 128.0)),
        weights=weights,
    )
    histogram = smooth_histogram(histogram)

    hue_centers = (hue_edges[:-1] + hue_edges[1:]) * 0.5
    green_band = (hue_centers >= 60.0) & (hue_centers <= 160.0)
    histogram[~green_band, :] = 0.0

    if not np.any(histogram):
        raise RuntimeError("No green-leaning residual cluster was found in hair_aa.png.")

    peak_hue_idx, peak_excess_idx = np.unravel_index(np.argmax(histogram), histogram.shape)
    peak_hue = float((hue_edges[peak_hue_idx] + hue_edges[peak_hue_idx + 1]) * 0.5)
    peak_excess = float((excess_edges[peak_excess_idx] + excess_edges[peak_excess_idx + 1]) * 0.5)

    local_cluster = candidate & (circular_distance_deg(hue, peak_hue) <= 18.0) & (
        np.abs(green_excess - peak_excess) <= 12.0
    )
    if np.count_nonzero(local_cluster) < 250:
        local_cluster = candidate & (circular_distance_deg(hue, peak_hue) <= 28.0) & (
            np.abs(green_excess - peak_excess) <= 18.0
        )

    if not np.any(local_cluster):
        raise RuntimeError("Residual green cluster detection did not produce a usable seed.")

    hue_center = circular_mean_deg(hue[local_cluster])
    hue_width = float(
        np.clip(
            np.percentile(circular_distance_deg(hue[local_cluster], hue_center), 97) + 10.0,
            16.0,
            54.0,
        )
    )
    excess_min = float(np.clip(np.percentile(green_excess[local_cluster], 8) - 4.0, 1.0, 64.0))
    sat_min = float(np.clip(np.percentile(sat[local_cluster], 5) - 0.10, 0.01, 0.60))
    value_min = float(np.clip(np.percentile(value[local_cluster], 2) - 0.08, 0.0, 0.60))

    cluster = {
        "peak_hue": peak_hue,
        "peak_excess": peak_excess,
        "hue_center": hue_center,
        "hue_width": hue_width,
        "excess_min": excess_min,
        "sat_min": sat_min,
        "value_min": value_min,
    }
    return cluster, green_excess, hue


def build_masks(
    rgb: np.ndarray,
    alpha: np.ndarray,
    cluster: dict[str, float],
    green_excess: np.ndarray,
    hue: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    _, sat, value = rgb_to_hsv(rgb)
    visible = alpha > 0
    hue_distance = circular_distance_deg(hue, cluster["hue_center"])
    max_rb = np.maximum(r, b)

    thresholds = {
        "hard_hue_width": float(min(66.0, cluster["hue_width"] * 1.35)),
        "hard_excess_min": float(max(0.5, cluster["excess_min"] * 0.70)),
        "hard_sat_min": float(max(0.0, cluster["sat_min"] * 0.55)),
        "hard_value_min": float(max(0.0, cluster["value_min"] * 0.60)),
        "soft_hue_width": float(min(78.0, cluster["hue_width"] * 1.65)),
        "soft_excess_min": float(max(-1.0, cluster["excess_min"] - 5.0)),
        "soft_sat_min": float(max(0.0, cluster["sat_min"] * 0.35)),
        "soft_value_min": float(max(0.0, cluster["value_min"] * 0.35)),
    }

    hard_mask = (
        visible
        & (value >= thresholds["hard_value_min"])
        & (sat >= thresholds["hard_sat_min"])
        & (hue_distance <= thresholds["hard_hue_width"])
        & (green_excess >= thresholds["hard_excess_min"])
        & (g >= max_rb - 1.0)
    )

    soft_mask = (
        visible
        & (value >= thresholds["soft_value_min"])
        & (sat >= thresholds["soft_sat_min"])
        & (hue_distance <= thresholds["soft_hue_width"])
        & (green_excess >= thresholds["soft_excess_min"])
        & (g >= max_rb - 3.0)
    )

    return hard_mask, soft_mask, thresholds


def apply_black_replacement(rgb: np.ndarray, alpha: np.ndarray, hard_mask: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    expanded = dilate(hard_mask, 3)
    feather_zone = expanded & soft_mask

    matte = hard_mask.astype(np.float32)
    matte[(feather_zone & ~hard_mask)] = 0.75
    matte = blur_mask(matte, passes=2)
    matte[hard_mask] = 1.0

    output = rgb.astype(np.float32) * (1.0 - matte[..., None])
    rgba = np.dstack([np.clip(output, 0.0, 255.0).astype(np.uint8), alpha])
    return rgba


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"Missing input image: {INPUT_PATH}", file=sys.stderr)
        return 1

    image = Image.open(INPUT_PATH).convert("RGBA")
    rgba = np.array(image)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]

    cluster, green_excess, hue = detect_residual_green_cluster(rgb, alpha)
    hard_mask, soft_mask, thresholds = build_masks(rgb, alpha, cluster, green_excess, hue)
    output = apply_black_replacement(rgb, alpha, hard_mask, soft_mask)
    Image.fromarray(output, "RGBA").save(OUTPUT_PATH)

    print(
        "Residual green cluster:",
        f"hue={cluster['hue_center']:.1f} +/- {cluster['hue_width']:.1f} deg,",
        f"green_excess>={cluster['excess_min']:.1f},",
        f"sat>={cluster['sat_min']:.2f},",
        f"value>={cluster['value_min']:.2f}",
    )
    print(
        "Black replacement mask:",
        f"hue+/-{thresholds['hard_hue_width']:.1f} deg,",
        f"green_excess>={thresholds['hard_excess_min']:.1f},",
        f"sat>={thresholds['hard_sat_min']:.2f},",
        f"value>={thresholds['hard_value_min']:.2f}",
    )
    print("Feathered 3 pixels outward before blurring to black.")
    print(f"Saved {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
