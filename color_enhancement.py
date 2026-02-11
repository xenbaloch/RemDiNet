import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedGlobalColorCorrection(nn.Module):
    """
    Enhanced color correction with vibrance control and proper gamma handling
    """

    def __init__(self):
        super().__init__()
        # More flexible color transform matrix (start at identity)
        self.weight = nn.Parameter(torch.eye(3).view(3, 3, 1, 1))
        self.bias = nn.Parameter(torch.zeros(3, 1, 1))

        # Learnable vibrance/saturation control (start at 1.2 instead of 1.0)
        self.vibrance_factor = nn.Parameter(torch.tensor(1.2))

        # Per-channel gamma correction (start at 1.0 = neutral)
        self.channel_gamma = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        # Register post-optimization hook to clamp weights
        self._register_weight_clamping()

    def _register_weight_clamping(self):
        """Register hooks to clamp weights after optimizer steps, not during forward pass"""

        def clamp_weights_hook(module, input):
            with torch.no_grad():
                # Tightened: near-identity clamps
                self.weight.data.clamp_(0.98, 1.02)
                self.bias.data.clamp_(-0.01, 0.01)
                self.vibrance_factor.data.clamp_(0.98, 1.05)
                # Tightened: gamma clamp to 0.95 - 1.05
                self.channel_gamma.data.clamp_(0.95, 1.05)

        self.register_forward_pre_hook(clamp_weights_hook)

    def forward(self, x):
        # x: [B,3,H,W] in [0,1]
        # NO weight clamping here - gradients are preserved!

        # Apply affine transform
        w = self.weight.expand(-1, -1, x.size(2), x.size(3))
        y = torch.einsum('bchw,ichw->bihw', x, w) + self.bias

        # 2. Apply vibrance enhancement (selective saturation boost)
        y = self.apply_vibrance(y, self.vibrance_factor)

        # 3. Apply per-channel gamma correction
        y = self.apply_channel_gamma(y, self.channel_gamma)

        return torch.clamp(y, 0., 1.)

    def apply_vibrance(self, x, factor):
        """Enhance color vibrance while preserving skin tones"""
        # Convert to LAB-like representation for better color manipulation
        luminance = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        luminance = luminance.unsqueeze(1)

        # Calculate saturation for each pixel
        max_rgb = torch.max(x, dim=1, keepdim=True)[0]
        min_rgb = torch.min(x, dim=1, keepdim=True)[0]
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)

        # Vibrance: boost less saturated colors more than already saturated ones
        vibrance_mask = 1.0 - saturation  # Less saturated = higher boost
        boost_factor = 1.0 + (factor - 1.0) * vibrance_mask

        # Apply vibrance
        color_diff = x - luminance
        enhanced = luminance + color_diff * boost_factor

        return enhanced

    def apply_channel_gamma(self, x, gamma):
        """Apply per-channel gamma correction"""
        # Protect against numerical issues near 0
        x_safe = torch.clamp(x, 1e-8, 1.0)

        # Apply per-channel gamma
        r = x_safe[:, 0:1].pow(gamma[0])
        g = x_safe[:, 1:2].pow(gamma[1])
        b = x_safe[:, 2:3].pow(gamma[2])

        return torch.cat([r, g, b], dim=1)


class AdaptiveColorEnhancement(nn.Module):
    """
    Adaptive color enhancement based on image statistics
    """

    def __init__(self):
        super().__init__()
        # Small network to predict enhancement parameters from image stats
        self.param_net = nn.Sequential(
            nn.Linear(9, 16),  # 9 stats: mean/std per channel + overall
            nn.ReLU(),
            nn.Linear(16, 6),  # 6 params: vibrance, 3 gammas, 2 curve params
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract image statistics
        stats = self.extract_stats(x)

        # Predict enhancement parameters
        params = self.param_net(stats)

        # Scale parameters to useful ranges
        vibrance = 0.8 + 1.2 * params[:, 0:1]  # 0.8 to 2.0
        gammas = 0.6 + 0.8 * params[:, 1:4]  # 0.6 to 1.4
        curve_strength = params[:, 4:5]  # 0 to 1
        curve_shift = params[:, 5:6] - 0.5  # -0.5 to 0.5

        # Apply adaptive enhancements
        x = self.apply_s_curve(x, curve_strength, curve_shift)
        x = self.apply_adaptive_vibrance(x, vibrance)
        x = self.apply_adaptive_gamma(x, gammas)

        return x

    def extract_stats(self, x):
        """Extract image statistics for adaptive enhancement"""
        b = x.size(0)
        # Channel-wise statistics
        r_mean = x[:, 0].mean(dim=[1, 2])
        g_mean = x[:, 1].mean(dim=[1, 2])
        b_mean = x[:, 2].mean(dim=[1, 2])
        r_std = x[:, 0].std(dim=[1, 2])
        g_std = x[:, 1].std(dim=[1, 2])
        b_std = x[:, 2].std(dim=[1, 2])

        # Overall statistics
        overall_mean = x.mean(dim=[1, 2, 3])
        overall_std = x.std(dim=[1, 2, 3])

        # Saturation statistic
        max_rgb = x.max(dim=1)[0]
        min_rgb = x.min(dim=1)[0]
        avg_saturation = (max_rgb - min_rgb).mean(dim=[1, 2])

        stats = torch.stack([
            r_mean, g_mean, b_mean,
            r_std, g_std, b_std,
            overall_mean, overall_std, avg_saturation
        ], dim=1)

        return stats

    def apply_s_curve(self, x, strength, shift):
        """Apply S-curve for contrast enhancement"""
        # S-curve formula: y = 1 / (1 + exp(-k*(x-0.5)))
        # Reshape parameters for broadcasting
        strength = strength.view(-1, 1, 1, 1) * 10  # Scale to useful range
        shift = shift.view(-1, 1, 1, 1)

        # Apply S-curve
        x_shifted = x - 0.5 - shift
        y = torch.sigmoid(strength * x_shifted)

        # Normalize to maintain range
        y_min = y.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        y_max = y.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        y_normalized = (y - y_min) / (y_max - y_min + 1e-8)

        return y_normalized

    def apply_adaptive_vibrance(self, x, vibrance):
        """Apply vibrance with batch-specific parameters"""
        vibrance = vibrance.view(-1, 1, 1, 1)
        luminance = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        color_diff = x - luminance
        return luminance + color_diff * vibrance

    def apply_adaptive_gamma(self, x, gammas):
        """Apply per-channel gamma with batch-specific parameters"""
        gammas = gammas.view(-1, 3, 1, 1)
        x_safe = torch.clamp(x, 1e-8, 1.0)

        # Apply per-channel gamma
        result = []
        for i in range(3):
            result.append(x_safe[:, i:i + 1].pow(gammas[:, i:i + 1]))

        return torch.cat(result, dim=1)


# Test time color enhancement function
def enhance_colors_post_process(image_tensor,
                                vibrance_boost=1.0,  # Neutral (was 1.2)
                                gamma=1.0,  # Neutral (was 0.9)
                                saturation_boost=1.0):  # Neutral (was 1.1)
    """
    Post-processing color enhancement for test time
    Can be applied after model inference for extra color pop
    NEUTRALIZED: Default values are now neutral (no enhancement)
    """
    # Ensure input is in [0, 1] range
    img = torch.clamp(image_tensor, 0, 1)

    # 1. Apply gamma correction first (linearize)
    img_gamma = img.pow(gamma)

    # 2. Boost saturation in HSV space
    img_hsv = rgb_to_hsv(img_gamma)
    img_hsv[:, 1] *= saturation_boost  # Boost saturation channel
    img_hsv[:, 1] = torch.clamp(img_hsv[:, 1], 0, 1)
    img_rgb = hsv_to_rgb(img_hsv)

    # 3. Apply vibrance (selective saturation)
    luminance = 0.299 * img_rgb[:, 0] + 0.587 * img_rgb[:, 1] + 0.114 * img_rgb[:, 2]
    luminance = luminance.unsqueeze(1)

    max_rgb = torch.max(img_rgb, dim=1, keepdim=True)[0]
    min_rgb = torch.min(img_rgb, dim=1, keepdim=True)[0]
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)

    vibrance_mask = 1.0 - saturation
    boost_factor = 1.0 + (vibrance_boost - 1.0) * vibrance_mask

    color_diff = img_rgb - luminance
    img_vibrant = luminance + color_diff * boost_factor

    return torch.clamp(img_vibrant, 0, 1)


def rgb_to_hsv(rgb):
    """Convert RGB to HSV color space"""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

    max_rgb = torch.max(rgb, dim=1, keepdim=True)[0]
    min_rgb = torch.min(rgb, dim=1, keepdim=True)[0]
    diff = max_rgb - min_rgb

    # Hue calculation
    hue = torch.zeros_like(max_rgb)

    mask_r = (max_rgb == r) & (diff > 0)
    mask_g = (max_rgb == g) & (diff > 0)
    mask_b = (max_rgb == b) & (diff > 0)

    hue[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6
    hue[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2
    hue[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4

    hue = hue / 6  # Normalize to [0, 1]

    # Saturation
    sat = torch.where(max_rgb > 0, diff / max_rgb, torch.zeros_like(max_rgb))

    # Value
    val = max_rgb

    return torch.cat([hue, sat, val], dim=1)


def hsv_to_rgb(hsv):
    """Convert HSV to RGB color space without 1D-cat errors."""
    # hsv: [B,3,H,W], h∈[0,1)→multiply by 6 for sector
    h, s, v = hsv[:, 0:1] * 6, hsv[:, 1:2], hsv[:, 2:3]

    c = v * s
    x = c * (1 - torch.abs((h % 2) - 1))
    m = v - c

    B, _, H, W = hsv.shape
    out = torch.zeros_like(hsv)

    # sector index [0..5] per-pixel
    h_idx = (h.long() % 6).squeeze(1)  # [B,H,W]
    c_img, x_img, m_img = c.squeeze(1), x.squeeze(1), m.squeeze(1)

    # for each hue sector, assign R/G/B + m
    for i in range(6):
        mask = (h_idx == i)  # bool [B,H,W]
        if not mask.any():
            continue

        # pick base (r,g,b) before adding m
        if i == 0:
            r_val, g_val, b_val = c_img[mask], x_img[mask], torch.zeros_like(c_img[mask])
        elif i == 1:
            r_val, g_val, b_val = x_img[mask], c_img[mask], torch.zeros_like(c_img[mask])
        elif i == 2:
            r_val, g_val, b_val = torch.zeros_like(c_img[mask]), c_img[mask], x_img[mask]
        elif i == 3:
            r_val, g_val, b_val = torch.zeros_like(c_img[mask]), x_img[mask], c_img[mask]
        elif i == 4:
            r_val, g_val, b_val = x_img[mask], torch.zeros_like(c_img[mask]), c_img[mask]
        else:  # i == 5
            r_val, g_val, b_val = c_img[mask], torch.zeros_like(c_img[mask]), x_img[mask]

        # write into each channel, adding the "m" offset
        out[:, 0][mask] = r_val + m_img[mask]
        out[:, 1][mask] = g_val + m_img[mask]
        out[:, 2][mask] = b_val + m_img[mask]

    return out