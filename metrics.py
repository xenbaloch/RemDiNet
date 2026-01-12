import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union
import cv2
import warnings
from metrics_core import calculate_psnr as _core_psnr, calculate_ms_ssim as _core_ms_ssim

warnings.filterwarnings('ignore')

# Silent dependency checking
try:
    import lpips

    _lpips_available = True
    _lpips_model = None
except ImportError:
    _lpips_available = False
    _lpips_model = None

try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio

    _skimage_available = True
except ImportError:
    _skimage_available = False

try:
    import pywt

    _pywt_available = True
except ImportError:
    _pywt_available = False

try:
    import scipy

    _scipy_available = True
except ImportError:
    _scipy_available = False


class CleanMetricsCalculator:
    """Clean metrics calculator for additional metrics beyond PSNR/MS-SSIM"""

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.eps = 1e-8

        # Initialize LPIPS silently
        global _lpips_model
        if _lpips_available and _lpips_model is None:
            try:
                _lpips_model = lpips.LPIPS(net='alex').to(device)
            except Exception:
                pass

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Safely convert tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            if tensor.requires_grad:
                tensor = tensor.detach()
            if tensor.is_cuda:
                tensor = tensor.cpu()
            array = tensor.numpy()
        else:
            array = tensor

        # Ensure proper format
        if array.ndim == 4:  # Batch dimension
            array = array.squeeze(0)
        if array.ndim == 3 and array.shape[0] in [1, 3]:  # Channel first
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 3 and array.shape[2] == 1:  # Grayscale
            array = array.squeeze(2)

        # Ensure [0, 1] range
        array = np.clip(array, 0.0, 1.0)
        return array

    def _prepare_images(self, img1: Union[torch.Tensor, np.ndarray],
                        img2: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare images for metric calculation"""
        try:
            arr1 = self._tensor_to_numpy(img1)
            arr2 = self._tensor_to_numpy(img2)

            # Ensure same dimensions
            if arr1.shape != arr2.shape:
                # Resize to match smaller image
                if arr1.ndim == 2:  # Grayscale
                    target_shape = (min(arr1.shape[0], arr2.shape[0]),
                                    min(arr1.shape[1], arr2.shape[1]))
                    arr1 = cv2.resize(arr1, (target_shape[1], target_shape[0]))
                    arr2 = cv2.resize(arr2, (target_shape[1], target_shape[0]))
                else:  # Color
                    target_shape = (min(arr1.shape[0], arr2.shape[0]),
                                    min(arr1.shape[1], arr2.shape[1]))
                    arr1 = cv2.resize(arr1, (target_shape[1], target_shape[0]))
                    arr2 = cv2.resize(arr2, (target_shape[1], target_shape[0]))

            return arr1, arr2

        except Exception:
            return None, None

    def calculate_psnr(self, img1, img2, *, normalized=False):
        """Delegate to metrics_core for consistency"""
        try:
            if isinstance(img1, np.ndarray):
                img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(self.device)
            if isinstance(img2, np.ndarray):
                img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(self.device)

            return float(_core_psnr(img1, img2, normalized=normalized).item())
        except Exception:
            return 0.0

    def calculate_ssim(self, img1: Union[torch.Tensor, np.ndarray],
                       img2: Union[torch.Tensor, np.ndarray]) -> float:
        """Calculate SSIM with fallback"""
        try:
            if _skimage_available:
                arr1, arr2 = self._prepare_images(img1, img2)
                if arr1 is None or arr2 is None:
                    return 0.0

                if arr1.ndim == 2:  # Grayscale
                    ssim_value = structural_similarity(arr1, arr2, data_range=1.0)
                else:  # Color
                    ssim_value = structural_similarity(arr1, arr2, data_range=1.0, channel_axis=2)
            else:
                ssim_value = self._calculate_ssim_fallback(img1, img2)

            if np.isnan(ssim_value) or np.isinf(ssim_value):
                return 0.0

            return float(ssim_value)

        except Exception:
            return 0.0

    def _calculate_ssim_fallback(self, img1: Union[torch.Tensor, np.ndarray],
                                 img2: Union[torch.Tensor, np.ndarray]) -> float:
        """Fallback SSIM implementation"""
        try:
            arr1, arr2 = self._prepare_images(img1, img2)
            if arr1 is None or arr2 is None:
                return 0.0

            # Convert to grayscale if needed
            if arr1.ndim == 3:
                arr1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
                arr2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)

            # SSIM calculation
            mu1 = cv2.GaussianBlur(arr1, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(arr2, (11, 11), 1.5)

            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = cv2.GaussianBlur(arr1 * arr1, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(arr2 * arr2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(arr1 * arr2, (11, 11), 1.5) - mu1_mu2

            C1 = (0.01 * 1.0) ** 2
            C2 = (0.03 * 1.0) ** 2

            numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

            ssim_map = numerator / denominator
            return float(np.mean(ssim_map))
        except Exception:
            return 0.0

    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate LPIPS perceptual distance"""
        global _lpips_model

        if not _lpips_available or _lpips_model is None:
            return 1.0

        try:
            # Ensure proper tensor format
            if img1.dim() == 3:
                img1 = img1.unsqueeze(0)
            if img2.dim() == 3:
                img2 = img2.unsqueeze(0)

            # Ensure 3 channels
            if img1.shape[1] == 1:
                img1 = img1.repeat(1, 3, 1, 1)
            if img2.shape[1] == 1:
                img2 = img2.repeat(1, 3, 1, 1)

            # Ensure same size
            if img1.shape != img2.shape:
                img2 = F.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)

            # Move to device and normalize to [-1, 1] for LPIPS
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            img1 = 2 * img1 - 1
            img2 = 2 * img2 - 1

            with torch.no_grad():
                lpips_value = _lpips_model(img1, img2)

            lpips_score = float(lpips_value.item())

            if np.isnan(lpips_score) or np.isinf(lpips_score) or lpips_score < 0:
                return 1.0

            return lpips_score

        except Exception:
            return 1.0

    def calculate_niqe(self, img: Union[torch.Tensor, np.ndarray]) -> float:
        """Calculate NIQE (Natural Image Quality Evaluator)"""
        if _pywt_available:
            return self._calculate_advanced_niqe(img)
        else:
            return self._calculate_basic_niqe(img)

    def _calculate_advanced_niqe(self, img: Union[torch.Tensor, np.ndarray]) -> float:
        """Advanced NIQE with wavelet features"""
        try:
            arr = self._tensor_to_numpy(img)
            if arr is None:
                return 100.0

            # Convert to grayscale if needed
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

            # Convert to 8-bit
            arr_uint8 = (arr * 255).astype(np.uint8)

            # Wavelet decomposition
            coeffs = pywt.wavedec2(arr_uint8, 'db4', mode='symmetric')
            coeffs = coeffs[:4]  # approximation + 3 detail levels

            # Calculate features from wavelet coefficients
            features = []
            for level_coeffs in coeffs[1:]:  # Skip approximation coefficients
                for subband in level_coeffs:
                    mean_val = np.mean(subband)
                    std_val = np.std(subband)
                    skewness = self._calculate_skewness(subband)
                    kurtosis = self._calculate_kurtosis(subband)
                    features.extend([mean_val, std_val, skewness, kurtosis])

            # Calculate local variance features
            if _scipy_available:
                from scipy import ndimage
                local_variance = ndimage.generic_filter(arr_uint8, np.var, size=3)
                features.extend([
                    np.mean(local_variance),
                    np.std(local_variance),
                    self._calculate_skewness(local_variance),
                    self._calculate_kurtosis(local_variance)
                ])

            # Simple NIQE score based on feature statistics
            feature_score = np.mean(np.abs(features))
            niqe_score = min(100.0, max(0.0, feature_score / 10.0))

            return float(niqe_score)

        except Exception:
            return self._calculate_basic_niqe(img)

    def _calculate_basic_niqe(self, img: Union[torch.Tensor, np.ndarray]) -> float:
        """Basic NIQE approximation using local statistics"""
        try:
            arr = self._tensor_to_numpy(img)
            if arr is None:
                return 100.0

            # Convert to grayscale if needed
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

            # Convert to 8-bit for processing
            arr_uint8 = (arr * 255).astype(np.uint8)

            # Calculate local statistics
            kernel_size = 7
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

            # Local mean and variance
            local_mean = cv2.filter2D(arr_uint8.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((arr_uint8.astype(np.float32) - local_mean) ** 2, -1, kernel)

            # Edge strength
            sobel_x = cv2.Sobel(arr_uint8, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(arr_uint8, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            # Combine statistics for NIQE approximation
            mean_local_var = np.mean(local_variance)
            std_local_var = np.std(local_variance)
            mean_edge_strength = np.mean(edge_strength)

            # Simple NIQE score (lower is better quality)
            niqe_score = (std_local_var + mean_edge_strength) / (mean_local_var + 1e-8)
            niqe_score = min(100.0, max(0.0, niqe_score))

            return float(niqe_score)

        except Exception:
            return 100.0

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        if _scipy_available:
            from scipy.stats import skew
            return float(skew(data.flatten()))
        else:
            # Manual calculation
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            skewness = np.mean(((data - mean_val) / std_val) ** 3)
            return float(skewness)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        if _scipy_available:
            from scipy.stats import kurtosis
            return float(kurtosis(data.flatten()))
        else:
            # Manual calculation
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            kurt = np.mean(((data - mean_val) / std_val) ** 4) - 3
            return float(kurt)

    def calculate_brightness_metrics(self, img: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """Calculate brightness-related metrics"""
        try:
            arr = self._tensor_to_numpy(img)
            if arr is None:
                return {'brightness': 0.0, 'contrast': 0.0}

            brightness = float(np.mean(arr))
            contrast = float(np.std(arr))

            # Dynamic range
            dynamic_range = float(np.max(arr) - np.min(arr))

            # Histogram analysis
            hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 1))
            hist_norm = hist / (hist.sum() + self.eps)

            # Entropy (information content)
            entropy = -np.sum(hist_norm * np.log2(hist_norm + self.eps))

            # Exposure metrics
            underexposed_ratio = float(np.sum(arr < 0.1) / arr.size)
            overexposed_ratio = float(np.sum(arr > 0.9) / arr.size)

            return {
                'brightness': brightness,
                'contrast': contrast,
                'dynamic_range': dynamic_range,
                'entropy': float(entropy),
                'underexposed_ratio': underexposed_ratio,
                'overexposed_ratio': overexposed_ratio
            }

        except Exception:
            return {'brightness': 0.0, 'contrast': 0.0, 'dynamic_range': 0.0,
                    'entropy': 0.0, 'underexposed_ratio': 0.0, 'overexposed_ratio': 0.0}

    def calculate_enhancement_metrics(
            self,
            enhanced: Union[torch.Tensor, np.ndarray],
            original: Union[torch.Tensor, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate enhancement-specific metrics (no-reference deltas)."""
        try:
            enh_arr = self._tensor_to_numpy(enhanced)
            orig_arr = self._tensor_to_numpy(original)

            if enh_arr is None or orig_arr is None:
                return {}

            # --- Brightness / contrast improvements ---
            orig_brightness = np.mean(orig_arr)
            enh_brightness = np.mean(enh_arr)
            brightness_improvement = enh_brightness - orig_brightness

            orig_contrast = np.std(orig_arr)
            enh_contrast = np.std(enh_arr)
            contrast_improvement = enh_contrast - orig_contrast

            # Approximate "detail" via Laplacian energy
            def _detail(x: np.ndarray) -> float:
                if x.ndim == 3:
                    xg = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
                else:
                    xg = x
                lap = cv2.Laplacian((xg * 255).astype(np.float32), cv2.CV_32F)
                return float(np.mean(np.abs(lap)))

            orig_detail = _detail(orig_arr)
            enh_detail = _detail(enh_arr)

            # --- Color enhancement (variance proxy) ---
            color_enhancement = 0.0
            if enh_arr.ndim == 3 and orig_arr.ndim == 3:
                orig_color_var = np.var(orig_arr, axis=2).mean()
                enh_color_var = np.var(enh_arr, axis=2).mean()
                color_enhancement = enh_color_var - orig_color_var

            # --- NEW: Lightness-Order Error (LOE, lower is better) ---
            def _to_gray(a: np.ndarray) -> np.ndarray:
                return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) if a.ndim == 3 else a

            og = _to_gray(orig_arr)
            eg = _to_gray(enh_arr)
            shifts = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            mismatches = []
            for dy, dx in shifts:
                og2 = np.roll(np.roll(og, dy, axis=0), dx, axis=1)
                eg2 = np.roll(np.roll(eg, dy, axis=0), dx, axis=1)
                o_rel = (og - og2) >= 0
                e_rel = (eg - eg2) >= 0
                mismatches.append((o_rel != e_rel).mean())
            loe = float(np.mean(mismatches))

            # --- NEW: Colorfulness (Hasler–Süsstrunk) and its Δ ---
            if enh_arr.ndim == 3:
                R, G, B = enh_arr[..., 0], enh_arr[..., 1], enh_arr[..., 2]
                rg = R - G
                yb = 0.5 * (R + G) - B
                std_rg, std_yb = np.std(rg), np.std(yb)
                mean_rg, mean_yb = np.mean(rg), np.mean(yb)
                colorfulness_enh = np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)

                R0, G0, B0 = orig_arr[..., 0], orig_arr[..., 1], orig_arr[..., 2]
                rg0 = R0 - G0
                yb0 = 0.5 * (R0 + G0) - B0
                std_rg0, std_yb0 = np.std(rg0), np.std(yb0)
                mean_rg0, mean_yb0 = np.mean(rg0), np.mean(yb0)
                colorfulness_orig = np.sqrt(std_rg0 ** 2 + std_yb0 ** 2) + 0.3 * np.sqrt(mean_rg0 ** 2 + mean_yb0 ** 2)

                colorfulness_delta = float(colorfulness_enh - colorfulness_orig)
            else:
                colorfulness_enh = 0.0
                colorfulness_delta = 0.0

            return {
                "brightness_improvement": float(brightness_improvement),
                "contrast_improvement": float(contrast_improvement),
                "detail_enhancement": float(enh_detail - orig_detail),
                "color_enhancement": float(color_enhancement),
                "enhancement_ratio": float(enh_brightness / (orig_brightness + self.eps)),
                "contrast_ratio": float(enh_contrast / (orig_contrast + self.eps)),
                "loe": float(loe),  # NEW
                "colorfulness": float(colorfulness_enh),  # NEW
                "colorfulness_delta": float(colorfulness_delta)  # NEW
            }

        except Exception:
            return {}

    def calculate_all_metrics(self, enhanced: Union[torch.Tensor, np.ndarray],
                              target: Optional[Union[torch.Tensor, np.ndarray]] = None,
                              original: Optional[Union[torch.Tensor, np.ndarray]] = None) -> Dict[str, float]:
        """Calculate comprehensive metrics for enhanced images"""
        metrics = {}

        # Reference-based metrics (if target available)
        if target is not None:
            metrics['psnr'] = self.calculate_psnr(enhanced, target)
            metrics['ssim'] = self.calculate_ssim(enhanced, target)

            # LPIPS (if available and inputs are tensors)
            if isinstance(enhanced, torch.Tensor) and isinstance(target, torch.Tensor):
                metrics['lpips'] = self.calculate_lpips(enhanced, target)

        # No-reference metrics
        metrics['niqe'] = self.calculate_niqe(enhanced)

        # Brightness metrics
        brightness_metrics = self.calculate_brightness_metrics(enhanced)
        metrics.update(brightness_metrics)

        # Enhancement metrics (if original available)
        if original is not None:
            enhancement_metrics = self.calculate_enhancement_metrics(enhanced, original)
            metrics.update(enhancement_metrics)

        # Filter out invalid values
        filtered_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                filtered_metrics[key] = value

        return filtered_metrics

    def batch_calculate_metrics(self, enhanced_batch: torch.Tensor,
                                target_batch: Optional[torch.Tensor] = None,
                                original_batch: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Calculate metrics for a batch of images"""
        batch_size = enhanced_batch.shape[0]
        all_metrics = []

        for i in range(batch_size):
            enhanced = enhanced_batch[i]
            target = target_batch[i] if target_batch is not None else None
            original = original_batch[i] if original_batch is not None else None

            metrics = self.calculate_all_metrics(enhanced, target, original)
            all_metrics.append(metrics)

        # Average metrics across batch
        if not all_metrics:
            return {}

        averaged_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m and m[key] is not None]
            if values:
                averaged_metrics[key] = np.mean(values)
                averaged_metrics[f'{key}_std'] = np.std(values)

        return averaged_metrics


# Legacy compatibility
class MetricsCalculator(CleanMetricsCalculator):
    """Backward compatibility wrapper"""

    def __init__(self, device='cuda'):
        super().__init__(device)

    def calculate_metrics(self, enhanced, target=None, original=None):
        """Legacy interface"""
        return self.calculate_all_metrics(enhanced, target, original)


# Simplified utility functions
def calculate_psnr_simple(img1, img2, *, normalized=False):
    """Simple PSNR calculation using metrics_core"""
    return _core_psnr(img1, img2, normalized=normalized).item()


def calculate_ssim_simple(img1, img2):
    """Simple SSIM calculation"""
    calc = CleanMetricsCalculator()
    return calc.calculate_ssim(img1, img2)


def evaluate_enhancement(enhanced_img, target_img=None, original_img=None, device='cuda'):
    """Comprehensive evaluation function"""
    calc = CleanMetricsCalculator(device=device)
    return calc.calculate_all_metrics(enhanced_img, target_img, original_img)


# Keep the same API as before but clean implementation
ImprovedMetricsCalculator = CleanMetricsCalculator