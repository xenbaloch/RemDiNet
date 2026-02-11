import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_weights import LOSS_WEIGHTS, COLOR_THRESHOLDS
from typing import Dict, Optional, Tuple
from torch import Tensor
import random

# Silent dependency checking
try:
    from pytorch_msssim import ms_ssim

    HAS_MS_SSIM = True
except ImportError:
    HAS_MS_SSIM = False

try:
    from torchvision.models import vgg19, VGG19_Weights

    HAS_VGG = True
except ImportError:
    HAS_VGG = False

try:
    import lpips

    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


def color_diversity_loss(enhanced_rgb):
    """Color diversity loss — always returns a differentiable value."""
    # Channel-wise statistics
    r, g, b = enhanced_rgb[:, 0], enhanced_rgb[:, 1], enhanced_rgb[:, 2]

    # Pairwise channel differences (want these to be non-zero)
    rg_diff = (r - g).abs().mean()
    rb_diff = (r - b).abs().mean()
    gb_diff = (g - b).abs().mean()
    avg_diversity = (rg_diff + rb_diff + gb_diff) / 3.0

    # Saturation measure
    max_rgb = enhanced_rgb.max(dim=1)[0]
    min_rgb = enhanced_rgb.min(dim=1)[0]
    saturation = ((max_rgb - min_rgb) / (max_rgb + 1e-8)).mean()

    # Target: we want diversity > 0.02 and saturation > 0.10
    # Loss is HIGH when diversity/saturation are LOW — always differentiable
    diversity_loss = torch.exp(-15.0 * avg_diversity)   # ≈1 when gray, ≈0 when colorful
    saturation_loss = torch.exp(-10.0 * saturation)     # ≈1 when gray, ≈0 when saturated

    return 0.5 * diversity_loss + 0.5 * saturation_loss


class SimplifiedDistractionAwareLoss(nn.Module):
    """Clean loss function with color preservation"""

    def __init__(self,
                 device: str = "cuda",
                 use_perceptual: bool = False,
                 w_reconstruction: float = LOSS_WEIGHTS["w_reconstruction"],
                 w_exposure: float = LOSS_WEIGHTS["w_exposure"],
                 w_edge: float = 0.0,
                 w_ssim: float = 0.0,
                 w_color_consistency: float = LOSS_WEIGHTS["w_color_consistency"],
                 w_perceptual: float = LOSS_WEIGHTS["w_perceptual"],
                 w_mask_mean: float = 0.6,
                 w_color_diversity: float = 0.0,
                 w_affine_decay: float = LOSS_WEIGHTS.get("w_affine_decay", 0.0),
                 w_semantic_preservation: float = 0.0,
                 eps: float = 1e-4
                 ):
        super().__init__()

        self.device = device
        self.use_perceptual = use_perceptual and HAS_VGG
        self.eps = eps

        # Simple loss components
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # Loss weights
        self.w_reconstruction = w_reconstruction
        self.w_exposure = w_exposure
        self.w_edge = w_edge
        self.w_ssim = w_ssim
        self.w_color_consistency = float(w_color_consistency)
        self.w_perceptual = w_perceptual if self.use_perceptual else 0.0
        self.w_mask_mean = w_mask_mean
        self.w_color_diversity   = float(w_color_diversity)
        self.w_semantic_preservation = w_semantic_preservation
        self.w_affine_decay = w_affine_decay

        # Perceptual loss (VGG-based) - Lazy initialization
        self.register_buffer("target_brightness", torch.tensor(0.60))
        self.vgg = None

        if self.use_perceptual and HAS_VGG:
            self._init_vgg_silent()

    def _init_vgg_silent(self):
        """Silent VGG initialization"""
        if self.vgg is None and self.use_perceptual and HAS_VGG:
            try:
                vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16]
                for param in vgg.parameters():
                    param.requires_grad = False
                self.vgg = vgg.to(self.device)
            except Exception:
                self.use_perceptual = False

    def set_weights(self, w_dict):
        """
        Update any w_* attributes in this loss module.
        Called once per epoch by the trainer when colour-ramp is active.
        """
        for k, v in w_dict.items():
            if hasattr(self, k):
                setattr(self, k, float(v))

    def enhanced_exposure_control_loss(self, enhanced, input_img, lower=0.45):
        """Penalize **only** if mean brightness < `lower`; never penalize brightening."""
        μ = enhanced.mean()
        return torch.relu(lower - μ) * 2.0

    def forward(self, results, target=None, input_img=None, epoch=0):
        """Clean loss computation with color preservation"""

        # Handle both dict and tensor inputs
        if isinstance(results, dict):
            enhanced = results.get('enhanced', results.get('output'))
            mask = results.get('mask', None)
            curves = results.get('curves', None)
            if enhanced is None:
                raise ValueError("No 'enhanced' key found in model output dict")
        else:
            enhanced = results
            mask = None
            curves = None

        if enhanced is None:
            raise ValueError("Enhanced image is required")
        if input_img is None:
            raise ValueError("Input image is required")

        # Clamp all inputs for stability
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        input_img = torch.clamp(input_img, 0.0, 1.0)
        if target is not None:
            target = torch.clamp(target, 0.0, 1.0)

        total_loss = torch.tensor(0.0, device=enhanced.device)

        loss_dict = {}

        # 1. L1 Reconstruction
        if target is None:
            recon_loss = torch.tensor(0.0, device=enhanced.device)
        else:
            recon_loss = self.l1_loss(enhanced, target)

        total_loss += self.w_reconstruction * recon_loss
        loss_dict['reconstruction'] = recon_loss

        if target is None:
            exp_loss = self.enhanced_exposure_control_loss(enhanced, input_img)
            total_loss += self.w_exposure * exp_loss
            loss_dict['exposure'] = exp_loss
        else:
            loss_dict['exposure'] = torch.tensor(0.0, device=enhanced.device)

        if curves is not None:
            curves_l1 = torch.mean(torch.abs(curves))
            curve_penalty = torch.relu(curves_l1 - 0.80)
            total_loss += 0.1 * curve_penalty
            loss_dict['curve_penalty'] = curve_penalty

        # 4. Edge Loss (When GT there)
        if self.w_edge > 0 and target is not None:
            edge_l = self.gradient_loss(enhanced, target)
            total_loss += self.w_edge * edge_l
            loss_dict['edge'] = edge_l
        else:
            loss_dict['edge'] = torch.tensor(0.0, device=enhanced.device)

        # 5. SSIM (MS-SSIM) structural term (enabled when GT is available)
        if self.w_ssim > 0 and target is not None:
            try:
                ssim_val = self.ssim_loss(enhanced, target)
            except Exception:
                ssim_val = torch.tensor(0.0, device=enhanced.device)
            ssim_term = torch.clamp(1.0 - ssim_val, 0.0, 1.0)
            total_loss += self.w_ssim * ssim_term
            loss_dict['ssim'] = ssim_term
        else:
            loss_dict['ssim'] = torch.tensor(0.0, device=enhanced.device)

        # 6. Enhanced color consistency loss
        ref_img = target if target is not None else input_img
        cc_loss = self.enhanced_color_consistency_loss(enhanced, ref_img)

        if target is None:                 # unsupervised batch → use 20 % weight
            cc_weight = 0.2 * self.w_color_consistency
        else:
            cc_weight = self.w_color_consistency
        total_loss += cc_weight * cc_loss

        loss_dict['colour'] = cc_loss

        # 7. Color diversity loss (prevents grayscale drift)
        if self.w_color_diversity > 0:
            cdiv = color_diversity_loss(enhanced)
            total_loss += self.w_color_diversity * cdiv
            loss_dict['color_diversity'] = cdiv
        else:
            loss_dict['color_diversity'] = torch.tensor(0.0, device=enhanced.device)

        # 8. Perceptual (VGG) Quality (only when GT available)
        if self.use_perceptual and target is not None:
            if self.vgg is None:
                self._init_vgg_silent()
            percep = self.simple_perceptual_loss(enhanced, target)
            total_loss += self.w_perceptual * percep
            loss_dict['perceptual'] = percep
        else:
            loss_dict['perceptual'] = torch.tensor(0.0, device=enhanced.device)

        # 9. Soft weight-decay on GlobalColourAffine
        if self.w_affine_decay > 0 and hasattr(self, 'model_affine_ref'):
            with torch.no_grad():
                aff_w = self.model_affine_ref.weight
            affine_l2 = (aff_w ** 2).mean()
            total_loss += self.w_affine_decay * affine_l2
            loss_dict['affine_decay'] = affine_l2

        # Mask mean-centering penalty only
        if mask is not None and mask.requires_grad:
            mean_reg = F.mse_loss(mask.mean(), torch.tensor(0.5, device=mask.device))
            total_loss += self.w_mask_mean * mean_reg
            loss_dict['mask_mean'] = mean_reg

        # Replace NaNs/Infs with zero before clamping
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=5.0, neginf=0.0)

        # Clean every sub-loss so logging stays sane
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                loss_dict[k] = torch.nan_to_num(v, nan=0.0, posinf=2.0, neginf=0.0)

        # Clamp after sanitising
        total_loss = torch.clamp(total_loss, 0.0, 5.0)
        loss_dict['total_loss'] = total_loss

        return loss_dict

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Enhanced edge-aware L1 between image gradients (Sobel)"""
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)

        def _grad(x, k):
            return F.conv2d(x, k.repeat(x.size(1), 1, 1, 1), padding=1, groups=x.size(1))

        gx_p, gy_p = _grad(pred, sobel_x), _grad(pred, sobel_y)
        gx_t, gy_t = _grad(target, sobel_x), _grad(target, sobel_y)

        # Enhanced gradient loss with magnitude weighting
        grad_mag_p = torch.sqrt(gx_p ** 2 + gy_p ** 2 + self.eps)
        grad_mag_t = torch.sqrt(gx_t ** 2 + gy_t ** 2 + self.eps)

        # L1 loss on gradients + magnitude preservation
        grad_l1 = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
        mag_loss = F.l1_loss(grad_mag_p, grad_mag_t)

        return grad_l1 + 0.5 * mag_loss

    def ssim_loss(self, pred: torch.Tensor, target: torch.Tensor, **_) -> torch.Tensor:
        """Use multi-scale SSIM for stronger structural guidance"""
        if HAS_MS_SSIM:
            return ms_ssim(pred, target, data_range=1.0, size_average=True)
        else:
            return self._simple_ssim(pred, target)

    def _simple_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simple SSIM implementation as fallback"""

        # Convert to grayscale
        def rgb_to_gray(x):
            return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        if pred.size(1) == 3:
            pred_gray = rgb_to_gray(pred)
            target_gray = rgb_to_gray(target)
        else:
            pred_gray = pred
            target_gray = target

        # Simple SSIM calculation
        mu1 = torch.mean(pred_gray)
        mu2 = torch.mean(target_gray)

        sigma1_sq = torch.var(pred_gray)
        sigma2_sq = torch.var(target_gray)
        sigma12 = torch.mean((pred_gray - mu1) * (target_gray - mu2))

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        denom = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / denom

        return ssim

    def enhanced_color_consistency_loss(self, enhanced: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Stronger color preservation that maintains saturation"""

        # 1. Preserve RGB ratios to maintain color relationships
        ref_sum = reference.sum(dim=1, keepdim=True) + 1e-8
        ref_ratios = reference / ref_sum

        enh_sum = enhanced.sum(dim=1, keepdim=True) + 1e-8
        enh_ratios = enhanced / enh_sum

        ratio_loss = F.mse_loss(enh_ratios, ref_ratios)

        # 2. Preserve saturation levels
        ref_max = reference.max(dim=1, keepdim=True)[0]
        ref_min = reference.min(dim=1, keepdim=True)[0]
        ref_sat = (ref_max - ref_min) / (ref_max + 1e-8)

        enh_max = enhanced.max(dim=1, keepdim=True)[0]
        enh_min = enhanced.min(dim=1, keepdim=True)[0]
        enh_sat = (enh_max - enh_min) / (enh_max + 1e-8)

        # Penalize saturation reduction
        sat_loss = F.relu(ref_sat - enh_sat).mean()

        # 3. Preserve hue
        hue_loss = self.color_consistency_loss(enhanced, reference)

        return ratio_loss + sat_loss + 0.5 * hue_loss

    def color_consistency_loss(self, enhanced: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Preserve color relationships while allowing saturation boost"""

        # Calculate hue preservation instead of strict color ratios
        def rgb_to_hue(img):
            max_c, _ = torch.max(img, dim=1, keepdim=True)
            min_c, _ = torch.min(img, dim=1, keepdim=True)
            diff = max_c - min_c + 1e-8

            r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
            hue = torch.where(max_c == r, ((g - b) / diff) % 6,
                              torch.where(max_c == g, (b - r) / diff + 2,
                                          (r - g) / diff + 4))

            return hue / 6.0  # Normalize to [0,1]

        # Only preserve hue, allow saturation/brightness changes
        enhanced_hue = rgb_to_hue(enhanced)
        reference_hue = rgb_to_hue(reference)
        return F.mse_loss(enhanced_hue, reference_hue)

    def simple_perceptual_loss(self, enhanced: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simple perceptual loss using early VGG features"""
        if self.vgg is None:
            return torch.tensor(0.0, device=enhanced.device)

        try:
            # Convert [0,1] → [-1,1] for VGG
            def _norm(x: torch.Tensor) -> torch.Tensor:
                return (x - 0.5) * 2.0

            enhanced_feat = self.vgg(_norm(enhanced))
            target_feat = self.vgg(_norm(target))

            # L1 over early-layer feature maps
            perceptual = F.l1_loss(enhanced_feat, target_feat)

        except Exception:
            perceptual = torch.tensor(0.0, device=enhanced.device)

        return perceptual


class SimpleReconstructionLoss(nn.Module):
    """Pure reconstruction loss with SSIM for better structure — FIX #10"""

    def __init__(self):
        super(SimpleReconstructionLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, results, target=None, input_img=None, epoch=0):
        if isinstance(results, dict):
            enhanced = results.get('enhanced', results.get('output'))
            if enhanced is None:
                raise ValueError("No 'enhanced' key found in model output dict")
        else:
            enhanced = results

        if enhanced is None:
            raise ValueError("Enhanced image is required")
        if target is None:
            return {'total_loss': torch.tensor(0.0,
                                               device=enhanced.device,
                                               dtype=enhanced.dtype)}

        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        # Combined L1 + MSE loss
        l1_loss = self.l1(enhanced, target)
        mse_loss = self.mse(enhanced, target)

        ssim_loss = torch.tensor(0.0, device=enhanced.device)
        if HAS_MS_SSIM:
            try:
                ssim_val = ms_ssim(enhanced, target, data_range=1.0, size_average=True)
                ssim_loss = 1.0 - ssim_val
            except Exception:
                pass

        # Weighted combination: L1 + MSE + SSIM
        total_loss = 0.5 * l1_loss + 0.2 * mse_loss + 0.3 * ssim_loss

        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=10.0, neginf=0.0)
        total_loss = torch.clamp(total_loss, 0.0, 10.0)

        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'mse_loss': mse_loss,
            'ssim_loss': ssim_loss,
            'reconstruction': total_loss
        }


class ZeroDCELoss(nn.Module):
    """Zero-DCE style loss for reference-free training"""

    def __init__(self):
        super().__init__()

    def forward(self, results, target=None, input_img=None, epoch=0):
        if isinstance(results, dict):
            enhanced = results.get('enhanced', results.get('output'))
        else:
            enhanced = results

        if enhanced is None or input_img is None:
            return {'total_loss': torch.tensor(0.0, device=enhanced.device if enhanced is not None else 'cpu')}

        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        input_img = torch.clamp(input_img, 0.0, 1.0)

        # Simple exposure loss
        exposure_loss = torch.abs(torch.mean(enhanced) - 0.6)

        return {
            'total_loss': exposure_loss,
            'exposure': exposure_loss
        }


def create_loss_function(loss_type: str, device: str = "cuda", **weights):
    """Factory function to create loss functions"""
    if loss_type in ("simplified", "simplified_perceptual"):
        return SimplifiedDistractionAwareLoss(device=device, **weights)
    elif loss_type == "recon":
        return SimpleReconstructionLoss()
    elif loss_type == "zero_dce":
        return ZeroDCELoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
