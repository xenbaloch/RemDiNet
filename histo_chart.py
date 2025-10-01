#!/usr/bin/env python3
"""
rgb_hist_grid.py — Plot smoothed per-channel RGB histograms for one or more images.

Features
- Accepts N images (OpenCV BGR read, robust to grayscale).
- Smoothed 256-bin histograms with Gaussian 1D conv (same as your code).
- Shared y-axis across all panels for fair comparison.
- Auto titles from filenames or custom via --titles.
- Save figure with --out and/or show interactively.
- Neat tick formatting and minimal styling.

Usage
------
# Basic: five images, show interactively
python rgb_hist_grid.py img1.jpg img2.jpg img3.jpg img4.jpg img5.jpg

# Provide custom titles and save a PNG (no GUI)
python rgb_hist_grid.py imgs/*.jpg \
  --titles "Input,Global,Semantic,SNR,Ours" \
  --out rgb_hist_grid.png --no-show

# Adjust smoothing and bins
python rgb_hist_grid.py a.jpg b.jpg c.jpg --smooth-sigma 2.5 --bins 256
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Plot smoothed RGB histograms for one or more images."
    )
    p.add_argument("images", nargs="+", help="Path(s) to input image files.")
    p.add_argument("--titles", type=str, default=None,
                   help="Comma-separated titles matching the images order.")
    p.add_argument("--bins", type=int, default=256, help="Number of histogram bins.")
    p.add_argument("--smooth-sigma", type=float, default=2.5,
                   help="Gaussian sigma for 1D smoothing of histograms.")
    p.add_argument("--share-y", action="store_true", default=True,
                   help="Share a global y-axis across all subplots.")
    p.add_argument("--no-share-y", dest="share_y", action="store_false",
                   help="Disable shared y-axis scaling.")
    p.add_argument("--legend", action="store_true", help="Show per-panel legend.")
    p.add_argument("--cols", type=int, default=5,
                   help="Max columns in the subplot grid.")
    p.add_argument("--figsize", type=float, nargs=2, default=None,
                   metavar=("W", "H"),
                   help="Figure size in inches (width height). If not set, auto-calculated.")
    p.add_argument("--out", type=str, default=None,
                   help="Output filepath (e.g., hist.png). Inferred by extension.")
    p.add_argument("--dpi", type=int, default=300, help="DPI for saving.")
    p.add_argument("--no-show", action="store_true",
                   help="Do not display the window (useful on servers/CI).")
    return p.parse_args()


# --- Helpers (smoothing + RGB hist) ---
def gaussian_kernel1d(sigma: float = 2.0, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        # No smoothing
        return np.array([1.0], dtype=np.float64)
    if radius is None:
        radius = int(3 * sigma)  # ~99% mass
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2 * sigma * sigma))
    k /= k.sum()
    return k


def smooth1d(h: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    k = gaussian_kernel1d(sigma=sigma)
    # same-mode convolution; histogram is 1D small vector -> fine
    return np.convolve(h, k, mode='same')


def rgb_hists_bgr(im_bgr: np.ndarray, bins: int = 256, smooth_sigma: float = 2.5):
    # OpenCV loads BGR; split and reorder to R,G,B for plotting
    if im_bgr.ndim == 2:  # grayscale -> treat as BGR
        im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_GRAY2BGR)
    b, g, r = cv2.split(im_bgr)

    # Using OpenCV for speed/consistency
    h_b = cv2.calcHist([b], [0], None, [bins], [0, 256]).flatten()
    h_g = cv2.calcHist([g], [0], None, [bins], [0, 256]).flatten()
    h_r = cv2.calcHist([r], [0], None, [bins], [0, 256]).flatten()

    # Normalize area to 1
    def norm(h):
        s = h.sum()
        return h / s if s > 0 else h

    h_b, h_g, h_r = norm(h_b), norm(h_g), norm(h_r)

    # Smooth for a clean look
    h_b = smooth1d(h_b, smooth_sigma)
    h_g = smooth1d(h_g, smooth_sigma)
    h_r = smooth1d(h_r, smooth_sigma)

    return h_r, h_g, h_b


def load_image_color(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return im


def main():
    args = parse_args()

    img_paths = [Path(p) for p in args.images]
    for p in img_paths:
        if not p.exists():
            sys.exit(f"[ERROR] File not found: {p}")

    # Titles
    if args.titles:
        titles = [t.strip() for t in args.titles.split(",")]
        if len(titles) != len(img_paths):
            sys.exit("[ERROR] --titles count must match number of images.")
    else:
        titles = [p.stem for p in img_paths]

    # Load images (BGR) and compute hists
    images = [load_image_color(p) for p in img_paths]
    x = np.arange(args.bins)

    all_hists = []
    ymax = 0.0
    for im in images:
        h_r, h_g, h_b = rgb_hists_bgr(im, bins=args.bins, smooth_sigma=args.smooth_sigma)
        all_hists.append((h_r, h_g, h_b))
        if args.share_y:
            ymax = max(ymax, float(h_r.max()), float(h_g.max()), float(h_b.max()))

    if args.share_y:
        # Round up ymax to nearest 0.01 so ticks look neat; set a sane floor
        ymax = float(np.ceil(ymax * 100.0) / 100.0)
        if ymax < 0.02:
            ymax = 0.02

    # Layout
    n = len(images)
    cols = max(1, args.cols)
    rows = int(np.ceil(n / cols))

    if args.figsize is not None:
        figsize = tuple(args.figsize)
    else:
        # Auto figure size: ~3.6h like your original, scale width with columns
        figsize = (3.5 * min(n, cols) + 0.1, 3.6 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for i, (ax, (h_r, h_g, h_b), title) in enumerate(zip(axes_flat, all_hists, titles)):
        # Draw in B, G, R order so fills stack nicely
        ax.fill_between(x, h_b, 0, alpha=0.35, color='blue',  linewidth=0)
        ax.plot(x, h_b, color='blue',  linewidth=1.2, label='Blue')

        ax.fill_between(x, h_g, 0, alpha=0.35, color='green', linewidth=0)
        ax.plot(x, h_g, color='green', linewidth=1.2, label='Green')

        ax.fill_between(x, h_r, 0, alpha=0.35, color='red',   linewidth=0)
        ax.plot(x, h_r, color='red',   linewidth=1.2, label='Red')

        ax.set_title(title)
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Normalized frequency')
        ax.set_xlim(0, args.bins - 1)
        if args.share_y:
            ax.set_ylim(0, ymax)

        # y-ticks every 0.01, formatted as 0.00, 0.01, ...
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        # x ticks (every 64 levels when bins=256; adapt for other bins)
        step = max(1, args.bins // 4)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
        ax.margins(x=0)

        if args.legend:
            ax.legend(loc='upper right', fontsize=8, frameon=False)

        # cosmetics
        ax.grid(False)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    # Hide any unused axes (if n < rows*cols)
    for j in range(len(all_hists), rows * cols):
        axes_flat[j].axis('off')

    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=args.dpi, bbox_inches='tight')
        print(f"[INFO] Saved figure to: {out_path.resolve()}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
