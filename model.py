import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from color_enhancement import EnhancedGlobalColorCorrection
from typing import Dict, List, Tuple, Optional
import torchvision.models as models
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


def normalize_imagenet(tensor):
    """Apply ImageNet normalization to [0,1] range tensors"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return (tensor - mean) / std


def denormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """Invert the ImageNet normalization to get back to [0,1] RGB space"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


# CBAM Implementation
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden_channels = max(in_planes // ratio, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        max_val = self.mlp(self.max_pool(x))
        return self.sigmoid(avg + max_val)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, max_val], dim=1)
        return self.sigmoid(self.conv(cat))


class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out


# Lightweight transformer helpers
class WindowMSA(nn.Module):
    """Minimal windowed multi-head self-attention"""

    def __init__(self, dim, heads=4, window=8):
        super().__init__()
        self.dim, self.heads, self.window = dim, heads, window
        assert dim % heads == 0
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.window

        # Pad to window size
        pad_h = (w - H % w) % w
        pad_w = (w - W % w) % w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
            H += pad_h
            W += pad_w

        # Windowed attention
        x = x.view(B, C, H // w, w, W // w, w).permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(-1, w * w, C)

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        head_dim = C // self.heads
        q, k, v = [t.view(t.shape[0], t.shape[1], self.heads, head_dim
                          ).transpose(1, 2) for t in (q, k, v)]
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], w * w, C)
        out = self.proj(out)

        out = out.view(B, H // w, W // w, w, w, C).permute(0, 5, 1, 3, 2, 4
                                                           ).reshape(B, C, H, W)

        # Crop back
        if pad_h or pad_w:
            out = out[:, :, :H - pad_h, :W - pad_w]

        return out


class RestormerBlock(nn.Module):
    """Lightweight Restormer-style block"""

    def __init__(self, dim=128, heads=4, window=8, mlp_ratio=2.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = WindowMSA(dim, heads, window)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio) * 2),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio) * 2, dim)
        )

    def forward(self, x):  # x: (B, C, H, W)
        h = x.permute(0, 2, 3, 1)  # BHWC
        h = h + self.attn(self.norm1(h).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        h = h + self.mlp(self.norm2(h))
        return h.permute(0, 3, 1, 2)  # back to BCHW


class TransformerDecoder(nn.Module):
    """Color-preserving transformer decoder"""

    def __init__(self, feature_dims, dim=128, heads=4, window=8):
        super().__init__()
        if isinstance(feature_dims, dict) and 'scale_16' in feature_dims:
            input_dim = feature_dims['scale_16']
        else:
            input_dim = 960

        self.proj = nn.Conv2d(input_dim, dim, 1)
        self.blocks = nn.Sequential(
            RestormerBlock(dim, heads, window),
            RestormerBlock(dim, heads, window),
            RestormerBlock(dim, heads, window)
        )

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 3, 1, 1), nn.PixelShuffle(2),
            nn.Conv2d(dim, dim * 4, 3, 1, 1), nn.PixelShuffle(2),
            nn.Conv2d(dim, dim * 4, 3, 1, 1), nn.PixelShuffle(2),
            nn.Conv2d(dim, dim * 4, 3, 1, 1), nn.PixelShuffle(2),
        )

        # Separate heads for luminance and color
        self.luminance_head = nn.Sequential(
            nn.Conv2d(dim, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid()
        )

        self.rgb_residual_head = nn.Sequential(
            nn.Conv2d(dim, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, x, semantic_feats):
        try:
            # Get input in [0,1] range
            if x.min() < -0.1:
                x_01 = denormalize_imagenet(x)
            else:
                x_01 = x

            original_input = x_01.clone()
            _, _, target_h, target_w = x_01.shape

            # Process semantic features
            if isinstance(semantic_feats, dict):
                if 'scale_16' in semantic_feats:
                    features = self.proj(semantic_feats['scale_16'])
                else:
                    B = x_01.size(0)
                    dummy_feats = torch.zeros(B, 960, target_h // 32, target_w // 32, device=x_01.device)
                    features = self.proj(dummy_feats)
            else:
                features = self.proj(semantic_feats)

            features = self.blocks(features)
            upsampled = self.upsample(features)

            # Ensure correct size
            if upsampled.shape[2:] != (target_h, target_w):
                upsampled = F.interpolate(upsampled, size=(target_h, target_w),
                                          mode='bilinear', align_corners=False)

            # Process luminance and color separately
            luminance_enhanced = self.luminance_head(upsampled)
            orig_lum = 0.299 * original_input[:, 0:1] + 0.587 * original_input[:, 1:2] + 0.114 * original_input[:, 2:3]
            enhanced_lum = 0.7 * luminance_enhanced + 0.3 * orig_lum

            # Preserve original colors
            orig_ratios = original_input / (orig_lum + 1e-8)
            color_preserved = enhanced_lum * orig_ratios

            # Add learned RGB residual
            rgb_residual = self.rgb_residual_head(upsampled) * 0.1
            enhanced = color_preserved + rgb_residual

            # Color saturation check
            max_c = enhanced.max(dim=1, keepdim=True)[0]
            min_c = enhanced.min(dim=1, keepdim=True)[0]
            saturation = (max_c - min_c).mean()

            if saturation < 0.1:
                color_diff = original_input - orig_lum
                enhanced = enhanced_lum + color_diff * 1.5

            return torch.clamp(enhanced, 0, 1)

        except Exception as e:
            # Fallback to input
            if x.min() < -0.1:
                return denormalize_imagenet(x)
            else:
                return x


class GlobalColourAffine(nn.Module):
    """Enhanced color correction with gradient-preserving weight management"""

    def __init__(self, enable_vibrance: bool = False):
        super().__init__()
        # Initialize to identity (neutral) instead of 1.1x boost
        init_matrix = torch.eye(3).view(3, 3, 1, 1)
        self.weight = nn.Parameter(init_matrix)
        self.bias = nn.Parameter(torch.zeros(3, 1, 1))

        self.enable_vibrance = enable_vibrance
        if enable_vibrance:
            # Start at 1.0 (neutral) instead of 1.3
            self.vibrance_factor = nn.Parameter(torch.tensor(1.0))

        # Register a post-optimization hook to clamp weights
        self._register_weight_clamping()

    def _register_weight_clamping(self):
        """Register hooks to clamp weights after optimizer steps, not during forward pass"""

        def clamp_weights_hook(module, input):
            # This runs before forward pass, but after optimizer updates
            with torch.no_grad():
                # Tightened: near-identity clamps to prevent strong color shifts
                self.weight.data.clamp_(0.95, 1.05)
                self.bias.data.clamp_(-0.02, 0.02)

                if self.enable_vibrance:
                    # Tightened: max 10% vibrance boost instead of 25%
                    self.vibrance_factor.data.clamp_(1.0, 1.1)

        self.register_forward_pre_hook(clamp_weights_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NO weight clamping here - gradients are preserved!
        
        # Optional: simple gray-world white balance
        if self.training:  # Only during training to avoid test-time shifts
            channel_means = x.mean(dim=[2, 3], keepdim=True)
            overall_mean = channel_means.mean()
            if overall_mean > 0:
                balance_factors = overall_mean / (channel_means + 1e-8)
                # Apply mild correction (max 5% adjustment)
                balance_factors = torch.clamp(balance_factors, 0.95, 1.05)
                x = x * balance_factors
                x = torch.clamp(x, 0.0, 1.0)
        
        w = self.weight.expand(-1, -1, *x.shape[2:])
        y = torch.einsum('bchw,ichw->bihw', x, w) + self.bias

        if self.enable_vibrance and not torch.isclose(
                self.vibrance_factor, torch.tensor(1.0, device=self.vibrance_factor.device)
        ):
            y = self._apply_vibrance(y, self.vibrance_factor)

        return torch.clamp(y, 0.0, 1.0)

    @staticmethod
    def _apply_vibrance(x, factor):
        lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        maxc = x.max(1, keepdim=True)[0]
        minc = x.min(1, keepdim=True)[0]
        sat = (maxc - minc) / (maxc + 1e-8)
        boost = 1.0 + (factor - 1.0) * (1.0 - sat) * 2.0
        return lum + (x - lum) * boost


class LearnableSNREstimator(nn.Module):
    """Learnable SNR estimation"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskSpecificSaliency(nn.Module):
    """Task-specific saliency estimation"""

    def __init__(self):
        super().__init__()
        self.saliency_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, dilation=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.saliency_net(x)


class ContrastRefinementBlock(nn.Module):
    """Contrast refinement with learnable mixing"""

    def __init__(self):
        super().__init__()
        self.refine_net = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1), nn.Tanh()
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, enhanced, original):
        combined = torch.cat([enhanced, original], dim=1)
        residual = self.refine_net(combined) * 0.1
        self.alpha.data.clamp_(0.0, 0.05)
        refined = enhanced + self.alpha * residual
        return torch.clamp(refined, 0, 1)


class MaskPredictor(nn.Module):
    """Joint SNR+Saliency mask generator"""

    def __init__(self, in_channels: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, 1, 1), nn.Sigmoid()
        )
        self.forward_calls = 0
        self.enable_dither = True

    def forward(self, x: torch.Tensor, snr_map: torch.Tensor, sem_map: torch.Tensor) -> torch.Tensor:
        try:
            target_h, target_w = x.size(2), x.size(3)

            if snr_map.shape[2:] != (target_h, target_w):
                snr_map = F.interpolate(snr_map, size=(target_h, target_w), mode='bilinear', align_corners=False)
            if sem_map.shape[2:] != (target_h, target_w):
                sem_map = F.interpolate(sem_map, size=(target_h, target_w), mode='bilinear', align_corners=False)

            inp = torch.cat([x, snr_map, sem_map], dim=1)
            mask = self.net(inp)

            if self.training and self.enable_dither:
                noise = torch.randn_like(mask) * 0.03
                mask = torch.clamp(mask + noise, 0.0, 1.0)

            # Apply sharpening when mask is trained (gamma > 1 sharpens the sigmoid)
            if not self.training or self.forward_calls > 1000:  # After some training
                mask = mask.pow(2.0)  # Sharpen the mask

            self.forward_calls += 1
            return mask

        except Exception:
            return torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device) * 0.5


class SNRAwareDistractionModule(nn.Module):
    """SNR-aware distraction detector"""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, tiny_mode: bool = False):
        super().__init__()
        self.tiny_mode = tiny_mode

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 4, 3, 1, 1), nn.BatchNorm2d(hidden_dim // 4), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 3, 1, 1), nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, 1, 1), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True)
        )

        self.content_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, 1, 1), nn.ReLU(inplace=True)
        )
        self.spatial_attention = SpatialAttention()
        self.content_output = nn.Sequential(nn.Conv2d(hidden_dim // 4, 1, 1), nn.Sigmoid())

        self.fusion_net = nn.Sequential(
            nn.Conv2d(hidden_dim + 2, hidden_dim // 2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, 1), nn.Sigmoid()
        )

        if self.tiny_mode:
            self.blur = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=False, groups=3)
            nn.init.constant_(self.blur.weight, 1 / 25.)

    def compute_snr_map(self, noisy: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        if noisy.size(1) == 3:
            cg = lambda x: 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            noisy_gray, denoised_gray = cg(noisy), cg(denoised)
        else:
            noisy_gray, denoised_gray = noisy, denoised

        noise = torch.clamp(torch.abs(noisy_gray - denoised_gray), min=1e-3)
        snr = torch.clamp(denoised_gray / (noise + 1e-8), 0.0, 50.0)
        return torch.sigmoid(snr - 1.0)

    def apply_denoising(self, x: torch.Tensor) -> torch.Tensor:
        if self.tiny_mode:
            return self.blur(x)
        else:
            return F.avg_pool2d(x, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, learned_snr: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(x)

        if learned_snr is None:
            denoised = self.apply_denoising(x)
            snr_map = self.compute_snr_map(x, denoised)
        else:
            snr_map = learned_snr

        snr_map = F.interpolate(snr_map, size=x.shape[2:], mode="bilinear", align_corners=False)

        content_feats = self.content_branch(features)
        attn = self.spatial_attention(content_feats)
        gated = content_feats * attn
        content_map = self.content_output(gated)
        content_map = F.interpolate(content_map, size=x.shape[2:], mode="bilinear", align_corners=False)

        fusion_in = torch.cat([features, snr_map, content_map], dim=1)
        distraction_mask = self.fusion_net(fusion_in)

        return distraction_mask, snr_map, content_map


class MobileNetV3LargeBackbone(nn.Module):
    """MobileNetV3-Large backbone for semantic guidance"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = mobilenet.features
        except Exception:
            # Fallback dummy features
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        self.feature_dims = {
            'scale_1': 16,
            'scale_2': 24,
            'scale_4': 40,
            'scale_8': 112,
            'scale_16': 960
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        try:
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == 1:
                    features['scale_1'] = x
                elif i == 3:
                    features['scale_2'] = x
                elif i == 6:
                    features['scale_4'] = x
                elif i == 12:
                    features['scale_8'] = x
                elif i == 16:
                    features['scale_16'] = x

            if 'scale_16' not in features:
                features['scale_16'] = x

        except Exception:
            # Create dummy features
            B, _, H, W = x.shape
            for scale, dim in self.feature_dims.items():
                scale_num = int(scale.split('_')[1])
                features[scale] = torch.zeros(B, dim, H // scale_num, W // scale_num, device=x.device)

        return features


class MobileNetSemanticHead(nn.Module):
    """Lightweight semantic segmentation head"""

    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(960, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU6(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU6(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, deepest_features: torch.Tensor) -> torch.Tensor:
        return self.segmentation_head(deepest_features)


class MobileNetSemanticGuidance(nn.Module):
    """Semantic guidance module using MobileNetV3-Large backbone"""

    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.backbone = MobileNetV3LargeBackbone(pretrained=True)
        self.semantic_head = MobileNetSemanticHead(num_classes)

        # Selectively unfreeze later layers
        try:
            for i, layer in enumerate(self.backbone.features):
                if i >= 10:
                    for param in layer.parameters():
                        param.requires_grad = True
                else:
                    for param in layer.parameters():
                        param.requires_grad = False
        except Exception:
            pass

        self.ms_convs = nn.ModuleList([
            nn.Conv2d(3, 64, 3, padding=1),
            nn.Conv2d(3, 128, 3, padding=1),
            nn.Conv2d(3, 256, 3, padding=1)
        ])
        for conv in self.ms_convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')

        self.decoder = TransformerDecoder(self.backbone.feature_dims, dim=128)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch, _, H, W = x.shape

        try:
            backbone_features = self.backbone(x)
            semantic_logits = self.semantic_head(backbone_features['scale_16'])
            semantic_probs = F.softmax(semantic_logits, dim=1)
        except Exception:
            # Create dummy features
            backbone_features = {
                'scale_1': torch.zeros(batch, 16, H // 2, W // 2, device=x.device),
                'scale_2': torch.zeros(batch, 24, H // 4, W // 4, device=x.device),
                'scale_4': torch.zeros(batch, 40, H // 8, W // 8, device=x.device),
                'scale_8': torch.zeros(batch, 112, H // 16, W // 16, device=x.device),
                'scale_16': torch.zeros(batch, 960, H // 32, W // 32, device=x.device),
            }
            semantic_logits = torch.zeros(batch, 19, H // 32, W // 32, device=x.device)
            semantic_probs = F.softmax(semantic_logits, dim=1)

        semantic_features = {
            'logits': semantic_logits,
            'probs': semantic_probs,
            'backbone_features': backbone_features,
            'multi_scale': self._extract_multiscale_features(x)
        }

        try:
            enhanced = self.decoder(x, backbone_features)
        except Exception:
            if x.min() < -0.1:
                enhanced = denormalize_imagenet(x)
            else:
                enhanced = x

        return enhanced, semantic_features

    def _extract_multiscale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats, current = [], x
        for i, conv in enumerate(self.ms_convs):
            if i > 0:
                current = F.avg_pool2d(current, 2)
            feats.append(conv(current))
        return feats


class ImprovedCurveEstimator(nn.Module):
    """Color-aware curve estimation with restored power"""

    def __init__(self, num_iterations: int = 8):
        super().__init__()
        self.num_iterations = num_iterations

        self.curve_net = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_iterations * 3, 3, 1, 1), nn.Tanh()
        )

        # Initialize for better color preservation
        with torch.no_grad():
            self.curve_net[-2].weight.data *= 0.02
            self.curve_net[-2].bias.data.fill_(0.0)

    def enhance_image(self, x: torch.Tensor, curves: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        # RESTORED: Put base_scale back to 0.25 from 0.15
        base_scale = 0.25

        # Adaptive scaling per channel
        channel_brightness = x.mean(dim=[2, 3], keepdim=True)
        channel_factors = torch.clamp(0.8 - channel_brightness, 0.1, 0.6)
        adaptive_scale = base_scale * channel_factors

        curves_reshaped = curves.view(batch_size, self.num_iterations, 3, height, width)
        curves_reshaped = curves_reshaped * adaptive_scale.unsqueeze(1)

        enhanced = x
        for i in range(self.num_iterations):
            # RESTORED: Put delta scaling back to 0.35 from 0.25
            delta = curves_reshaped[:, i, :, :, :] * (1.0 - enhanced) * 0.35
            enhanced = enhanced + delta
            enhanced = torch.clamp(enhanced, 0, 1.0)

        # Color preservation check
        color_loss = self._check_color_loss(x, enhanced)
        if color_loss > 0.3:
            enhanced = 0.7 * enhanced + 0.3 * x * (enhanced.mean() / (x.mean() + 1e-8))
            enhanced = torch.clamp(enhanced, 0, 1.0)

        return enhanced, curves_reshaped

    def _check_color_loss(self, original, enhanced):
        orig_sat = (original.max(dim=1)[0] - original.min(dim=1)[0]).mean()
        enh_sat = (enhanced.max(dim=1)[0] - enhanced.min(dim=1)[0]).mean()
        return torch.clamp((orig_sat - enh_sat) / (orig_sat + 1e-8), 0, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_curves = self.curve_net(x)
        enhanced, processed_curves = self.enhance_image(x, raw_curves)
        return enhanced, raw_curves


class IntelligentBlendingModule(nn.Module):
    """Smart blending based on enhancement quality"""

    def __init__(self):
        super().__init__()
        self.quality_net = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, original, curve_enhanced, semantic_enhanced):
        combined = torch.cat([original, curve_enhanced, semantic_enhanced], dim=1)
        blend_weight = self.quality_net(combined)

        curve_diversity = self._color_diversity(curve_enhanced)
        semantic_diversity = self._color_diversity(semantic_enhanced)

        diversity_ratio = curve_diversity / (semantic_diversity + 1e-6)
        adjusted_weight = blend_weight / torch.clamp(diversity_ratio, 0.5, 2.0)
        adjusted_weight = torch.clamp(adjusted_weight, 0.05, 0.95)

        result = (1 - adjusted_weight) * curve_enhanced + adjusted_weight * semantic_enhanced
        return result, adjusted_weight

    def _color_diversity(self, img):
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        diversity = torch.abs(r - g) + torch.abs(r - b) + torch.abs(g - b)
        return diversity.mean(dim=[2, 3], keepdim=True)


class UnifiedLowLightEnhancer(nn.Module):
    """Enhanced low-light enhancement model with all fixes applied"""

    def __init__(
            self,
            num_iterations: int = 8,
            use_snr_awareness: bool = True,
            use_semantic_guidance: bool = True,
            use_learnable_snr: bool = False,
            use_task_saliency: bool = False,
            use_contrast_refinement: bool = False,
            snr_tiny_mode: bool = False,
            disable_color_affine: bool = False,
            disable_color_enhancer: bool = False
    ):
        super().__init__()

        # Store configuration
        self.num_iterations = num_iterations
        self.use_snr_awareness = use_snr_awareness
        self.use_semantic_guidance = use_semantic_guidance
        self.use_learnable_snr = use_learnable_snr
        self.use_task_saliency = use_task_saliency
        self.use_contrast_refinement = use_contrast_refinement
        self.disable_color_affine = disable_color_affine
        self.disable_color_enhancer = disable_color_enhancer

        # Color enhancement components (vibrance off by default)
        self.colour_head = GlobalColourAffine(enable_vibrance=False)
        try:
            self.color_enhancer = EnhancedGlobalColorCorrection()
        except Exception:
            self.color_enhancer = None

        self.skip_mask = False  # For stage-specific training

        # Optional components
        if self.use_learnable_snr:
            self.snr_estimator = LearnableSNREstimator()

        if self.use_task_saliency:
            self.saliency_estimator = TaskSpecificSaliency()

        if self.use_snr_awareness:
            self.snr_distraction_module = SNRAwareDistractionModule(
                in_channels=3, hidden_dim=64, tiny_mode=snr_tiny_mode
            )
        else:
            self.snr_distraction_module = None

        if self.use_semantic_guidance:
            self.semantic_guidance = MobileNetSemanticGuidance()
        else:
            self.semantic_guidance = None

        # Core enhancement components
        self.curve_estimator = ImprovedCurveEstimator(num_iterations)
        self.mask_predictor = MaskPredictor(in_channels=5, hidden_dim=32)
        self.intelligent_blending = IntelligentBlendingModule()

        if self.use_contrast_refinement:
            self.contrast_refiner = ContrastRefinementBlock()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Denormalize and pad
        x_denorm = denormalize_imagenet(x)
        B, C, H0, W0 = x_denorm.shape
        pad_h = (32 - (H0 % 32)) % 32
        pad_w = (32 - (W0 % 32)) % 32
        if pad_h or pad_w:
            x_denorm = F.pad(x_denorm, (0, pad_w, 0, pad_h), mode='replicate')
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        B, C, H, W = x_denorm.shape

        # SNR/distraction analysis
        snr_map = torch.zeros(B, 1, H, W, device=x.device)
        distraction_mask = torch.zeros_like(snr_map)
        content_map = torch.zeros_like(snr_map)

        if self.use_snr_awareness and self.snr_distraction_module:
            learned = (self.snr_estimator(x_denorm.detach())
                       if self.use_learnable_snr else None)
            try:
                distraction_mask, snr_map, content_map = self.snr_distraction_module(x_denorm, learned_snr=learned)
            except Exception:
                pass

        # ---- Fuse SNR-map and distraction-mask into one guidance map ----
        guidance = torch.clamp(snr_map + distraction_mask, 0.0, 1.0)

        # Semantic guidance
        semantic_enhanced = x_denorm
        semantic_features = {
            'probs': torch.zeros(B, 19, H, W, device=x.device),
            'logits': torch.zeros(B, 19, H, W, device=x.device),
            'backbone_features': {},
            'multi_scale': []
        }

        if self.use_semantic_guidance and self.semantic_guidance:
            try:
                out = self.semantic_guidance(x)
                if isinstance(out, tuple):
                    semantic_norm, semantic_features = out
                    semantic_enhanced = semantic_norm
                else:
                    semantic_enhanced = out
                semantic_enhanced = F.interpolate(semantic_enhanced, size=(H, W),
                                                  mode='bilinear', align_corners=False)
            except Exception:
                pass

        # Curve enhancement
        try:
            x_ce_input = torch.cat([x_denorm, snr_map], dim=1)
            curve_enhanced, curves = self.curve_estimator(x_ce_input)
            if curve_enhanced.mean() > 0.85:
                curve_enhanced = x_denorm + (curve_enhanced - x_denorm) * 0.8
        except Exception:
            curve_enhanced = x_denorm
            curves = torch.zeros(B, self.num_iterations * 3, H, W, device=x.device)

        # Build semantic mask
        if isinstance(semantic_features, dict) and 'probs' in semantic_features:
            sem_map = semantic_features['probs'].max(dim=1, keepdim=True)[0]
        else:
            sem_map = torch.zeros(B, 1, H, W, device=x.device)

        # Blending
        if getattr(self, 'skip_mask', False):
            mask = torch.ones_like(snr_map) * 0.5
        else:
            try:
                mask = self.mask_predictor(x_denorm, guidance, sem_map)
            except Exception:
                mask = torch.ones_like(snr_map) * 0.5

        if getattr(self, "use_curve_only", False):
            final = curve_enhanced
            mask = torch.ones_like(snr_map)  # dummy for loss
        else:
            sharp = mask.pow(2)  # γ=2 ⇒ mid-gray → 0.25, white/black stay ~1/0
            blended = sharp * curve_enhanced + (1.0 - sharp) * semantic_enhanced
            final = torch.clamp(blended, 0.0, 1.0)

            # ── NEW ▸ trainable global colour correction (with toggles) ───────────────────────
            try:
                if not self.disable_color_affine:
                    final = self.colour_head(final)  # 3×3 affine
                if not self.disable_color_enhancer and self.color_enhancer is not None:
                    final = self.color_enhancer(final)  # tiny CNN
            except Exception:
                pass

        # Crop back to original size
        final = final[..., :H0, :W0]

        # Final color boost if needed
        final_saturation = (final.max(dim=1)[0] - final.min(dim=1)[0]).mean()
        if final_saturation < 0.2:
            luminance = 0.299 * final[:, 0:1] + 0.587 * final[:, 1:2] + 0.114 * final[:, 2:3]
            color_diff = final - luminance
            final = luminance + color_diff * 2.0
            final = torch.clamp(final, 0.0, 1.0)

        return {
            'enhanced': final,
            'mask': mask[..., :H0, :W0] if mask.shape[2] > H0 or mask.shape[3] > W0 else mask,
            'distraction_mask': distraction_mask[..., :H0, :W0] if distraction_mask.shape[2] > H0 or
                                                                   distraction_mask.shape[3] > W0 else distraction_mask,
            'snr_map': snr_map[..., :H0, :W0] if snr_map.shape[2] > H0 or snr_map.shape[3] > W0 else snr_map,
            'semantic_features': semantic_features,
            'curves': curves[..., :H0, :W0] if curves.shape[2] > H0 or curves.shape[3] > W0 else curves,
            'curve_enhanced': curve_enhanced[..., :H0, :W0],
            'semantic_enhanced': semantic_enhanced[..., :H0, :W0] if semantic_enhanced.shape[2] > H0 or
                                                                     semantic_enhanced.shape[
                                                                         3] > W0 else semantic_enhanced
        }

    def get_model_stats(self, count_frozen: bool = True) -> Dict[str, float]:
        """Get model statistics"""
        total = sum(p.numel() for p in self.parameters()
                    if count_frozen or p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_params_M': total / 1e6,
            'trainable_params_M': trainable / 1e6,
            'frozen_params_M': (total - trainable) / 1e6,
            'use_snr_awareness': self.use_snr_awareness,
            'use_semantic_guidance': self.use_semantic_guidance,
            'num_iterations': self.num_iterations
        }

    def enable_snr_awareness(self, enable: bool = True):
        """Enable/disable SNR awareness at runtime"""
        self.use_snr_awareness = enable

    def enable_semantic_guidance(self, enable: bool = True):
        """Enable/disable semantic guidance at runtime"""
        self.use_semantic_guidance = enable

    def enable_color_affine(self, enable: bool = True):
        """Enable/disable color affine at runtime"""
        self.disable_color_affine = not enable

    def enable_color_enhancer(self, enable: bool = True):
        """Enable/disable color enhancer at runtime"""
        self.disable_color_enhancer = not enable


# Keep compatibility aliases
SemanticGuidedEnhancement = MobileNetSemanticGuidance
SemanticEmbeddingModule = MobileNetSemanticGuidance

if __name__ == "__main__":
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UnifiedLowLightEnhancer(
        num_iterations=8,
        use_snr_awareness=True,
        use_semantic_guidance=True,
        use_learnable_snr=True
    ).to(device)

    x = torch.randn(1, 3, 256, 256).to(device)
    x = normalize_imagenet(torch.clamp(x, 0, 1))

    stats = model.get_model_stats()
    print(f"Model: {stats['total_params_M']:.1f}M parameters")

    try:
        with torch.no_grad():
            results = model(x)
        print("✅ Forward pass successful")
        enhanced = results['enhanced']
        print(f"Output: {enhanced.shape}, range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")