import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- 1) Load images in COLOR (BGR in OpenCV) ---
img_low_light    = cv2.imread(r"E:\materialp3\fig1_low.jpg",      cv2.IMREAD_COLOR)
img_global_light = cv2.imread(r"E:\materialp3\fig1_global.jpg",   cv2.IMREAD_COLOR)
img_semantic     = cv2.imread(r"E:\materialp3\fig1_semantic.jpg", cv2.IMREAD_COLOR)
img_snr          = cv2.imread(r"E:\materialp3\fig1_snr.jpg",      cv2.IMREAD_COLOR)
img_ours         = cv2.imread(r"E:\materialp3\fig1_ours.jpg",     cv2.IMREAD_COLOR)

images = [img_low_light, img_global_light, img_semantic, img_snr, img_ours]
titles  = ["Input", "Global", "Semantic", "SNR", "Ours"]

# --- 2) Safety + ensure 3 channels ---
for i, (name, im) in enumerate(zip(titles, images)):
    if im is None:
        raise FileNotFoundError(f"Image for '{name}' failed to load. Check the file path.")
    if len(im.shape) == 2 or im.shape[2] == 1:  # grayscale -> BGR
        images[i] = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

# --- 3) Helpers (smoothing + RGB hist) ---
def gaussian_kernel1d(sigma=2.0, radius=None):
    if radius is None:
        radius = int(3 * sigma)  # ~99% mass
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x**2) / (2 * sigma * sigma))
    k /= k.sum()
    return k

def smooth1d(h, sigma=2.0):
    k = gaussian_kernel1d(sigma=sigma)
    return np.convolve(h, k, mode='same')

def rgb_hists_bgr(im_bgr, bins=256, smooth_sigma=2.5):
    # OpenCV is BGR; split then reorder to R,G,B for plotting
    b, g, r = cv2.split(im_bgr)
    h_b = cv2.calcHist([b], [0], None, [bins], [0, 256]).flatten()
    h_g = cv2.calcHist([g], [0], None, [bins], [0, 256]).flatten()
    h_r = cv2.calcHist([r], [0], None, [bins], [0, 256]).flatten()

    # normalize to area = 1
    def norm(h):
        s = h.sum()
        return h / s if s > 0 else h

    h_b, h_g, h_r = norm(h_b), norm(h_g), norm(h_r)

    # smooth for a clean look
    h_b = smooth1d(h_b, smooth_sigma)
    h_g = smooth1d(h_g, smooth_sigma)
    h_r = smooth1d(h_r, smooth_sigma)

    return h_r, h_g, h_b

x = np.arange(256)

# --- 4) First pass: compute all hists + global ymax for shared y-axis ---
all_hists = []
ymax = 0.0
for im in images:
    h_r, h_g, h_b = rgb_hists_bgr(im, smooth_sigma=2.5)
    all_hists.append((h_r, h_g, h_b))
    ymax = max(ymax, float(h_r.max()), float(h_g.max()), float(h_b.max()))

# round up ymax to nearest 0.01 so ticks look neat
ymax = np.ceil(ymax * 100.0) / 100.0
if ymax < 0.02:  # avoid a too-flat axis if images are very uniform
    ymax = 0.02

# --- 5) Plot with smooth fills + consistent ticks ---
fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
for ax, (h_r, h_g, h_b), title in zip(axes, all_hists, titles):
    # draw in B, G, R order so fills stack nicely
    ax.fill_between(x, h_b, 0, alpha=0.35, color='blue',  linewidth=0)
    ax.plot(x, h_b, color='blue',  linewidth=1.2, label='Blue')

    ax.fill_between(x, h_g, 0, alpha=0.35, color='green', linewidth=0)
    ax.plot(x, h_g, color='green', linewidth=1.2, label='Green')

    ax.fill_between(x, h_r, 0, alpha=0.35, color='red',   linewidth=0)
    ax.plot(x, h_r, color='red',   linewidth=1.2, label='Red')

    ax.set_title(title)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Normalized frequency')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, ymax)

    # y-ticks every 0.01, formatted as 0.00, 0.01, ...
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # optional: tidy x ticks (every 64 levels)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(64))
    ax.margins(x=0)

    ax.legend(loc='upper right', fontsize=8, frameon=False)

# cosmetics
for ax in axes:
    ax.grid(False)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

plt.tight_layout()
# plt.savefig(r"E:\materialp3\rgb_hist_grid.png", dpi=300, bbox_inches='tight')  # optional save
plt.show()
