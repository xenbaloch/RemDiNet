import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Local imports
from model import UnifiedLowLightEnhancer
from metrics_core import calculate_psnr, calculate_ms_ssim

# Optional dependencies
try:
    from metrics import CleanMetricsCalculator as ImprovedMetricsCalculator
except ImportError: 
    ImprovedMetricsCalculator = None

try: 
    from color_enhancement import enhance_colors_post_process
except ImportError:
    def enhance_colors_post_process(tensor, **kwargs):
        return tensor

def denormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """Convert ImageNet normalized tensor back to [0,1] RGB"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def load_model(model_path: str, device: str = 'cuda', 
               disable_color_affine: bool = False,
               disable_color_enhancer: bool = False) -> Optional[torch.nn.Module]:
    """Load model with automatic architecture detection"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Infer num_iterations
        curve_weight_key = 'curve_estimator.curve_net.6.weight'
        if curve_weight_key in state_dict:
            output_channels = state_dict[curve_weight_key].shape[0]
            num_iterations = output_channels // 3
        else:
            num_iterations = 8

        # Infer other parameters
        use_snr_awareness = any(k.startswith('snr_distraction_module') for k in state_dict.keys())
        use_semantic_guidance = any(k.startswith('semantic_guidance') for k in state_dict.keys())
        use_learnable_snr = any(k.startswith('snr_estimator') for k in state_dict.keys())
        use_contrast_refinement = any(k.startswith('contrast_refiner') for k in state_dict.keys())

        # Initialize model
        model = UnifiedLowLightEnhancer(
            num_iterations=num_iterations,
            use_snr_awareness=use_snr_awareness,
            use_semantic_guidance=use_semantic_guidance,
            use_learnable_snr=use_learnable_snr,
            use_contrast_refinement=use_contrast_refinement,
            snr_tiny_mode=use_snr_awareness and not use_learnable_snr,
            disable_color_affine=disable_color_affine,
            disable_color_enhancer=disable_color_enhancer,
        )

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()

        # Validate model weights
        for param in model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print("❌ Model contains invalid weights")
                return None

        if 'best_psnr' in checkpoint: 
            print(f"✅ Model loaded (trained PSNR: {checkpoint['best_psnr']:.2f} dB)")
        else:
            print("✅ Model loaded successfully")

        return model

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def load_image(image_path: str, device: str = 'cuda', native_resolution: bool = True) -> tuple: 
    """
    Load and preprocess image for model input
    
    Args:
        image_path: Path to input image
        device: Device to load tensor on
        native_resolution: If True, test at native resolution (manuscript mode)
                          If False, resize to fixed size for speed
    """
    try:
        image = Image.open(image_path).convert('RGB')
        original_size = (image.height, image.width)

        # Transform:  ToTensor → Normalize → Pad (no resize for native resolution)
        tensor = transforms.ToTensor()(image)
        
        # Apply ImageNet normalization
        tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )(tensor)
        
        tensor = tensor.unsqueeze(0).to(device)

        # Pad to multiple of 32 for model compatibility
        _, _, H, W = tensor.shape
        pad_h = (32 - (H % 32)) % 32
        pad_w = (32 - (W % 32)) % 32
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='replicate')

        return tensor, original_size

    except Exception as e:
        print(f"❌ Failed to load image {image_path}: {e}")
        return None, None


def save_image(tensor: torch. Tensor, output_path: str, original_size: tuple = None) -> bool:
    """Save tensor as image"""
    try:
        # Prepare tensor
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = torch.clamp(tensor, 0.0, 1.0).cpu()

        # Convert to PIL
        image = transforms.ToPILImage()(tensor)

        # Crop to original size if needed
        if original_size: 
            orig_h, orig_w = original_size
            if image.height > orig_h or image.width > orig_w:
                image = image.crop((0, 0, orig_w, orig_h))

        # Save with high quality
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, quality=95, optimize=False)
        return True

    except Exception as e: 
        print(f"❌ Failed to save image:  {e}")
        return False


def enhance_image(model, image_path: str, output_path: str, device: str = 'cuda',
                  enable_post_process: bool = True, native_resolution: bool = True) -> Dict[str, Any]:
    """
    Enhance single image
    
    Args:
        native_resolution: Test at native resolution (manuscript:  True)
    """
    # Load image at native resolution
    image_tensor, original_size = load_image(image_path, device, native_resolution)
    if image_tensor is None:
        return {'error': f'Failed to load {image_path}'}

    try:
        start_time = time.time()

        with torch.no_grad():
            results = model(image_tensor)
            enhanced = results.get('enhanced', results) if isinstance(results, dict) else results

            # Validate output
            if torch.isnan(enhanced).any() or torch.isinf(enhanced).any():
                return {'error': 'Model produced invalid values'}

            if enhanced.abs().max() < 1e-8:
                return {'error': 'Model produced all-zero output'}

            enhanced = torch.clamp(enhanced, 0.0, 1.0)

            # Post-process color enhancement (enabled by default)
            if enable_post_process:
                enhanced = enhance_colors_post_process(
                    enhanced,
                    vibrance_boost=1.0,
                    saturation_boost=1.0,
                    gamma=1.0
                )

            # Crop to original size
            if enhanced.shape[2:] != original_size:
                orig_h, orig_w = original_size
                enhanced = enhanced[: , :, :orig_h, :orig_w]

        inference_time = time.time() - start_time

        # Save enhanced image
        if not save_image(enhanced, output_path, original_size):
            return {'error': 'Failed to save enhanced image'}

        # Calculate basic metrics
        enhanced_np = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        brightness = float(np.mean(enhanced_np))
        contrast = float(np.std(enhanced_np))

        return {
            'success': True,
            'inference_time': inference_time,
            'brightness': brightness,
            'contrast': contrast,
            'output_path': output_path,
            'resolution': f"{original_size[1]}×{original_size[0]}"
        }

    except Exception as e:
        return {'error': f'Enhancement failed: {str(e)}'}


def test_single_image(model_path: str, image_path:  str, output_path: str,
                      gt_path: str = None, device: str = 'cuda',
                      enable_post_process: bool = True,
                      native_resolution: bool = True,
                      disable_color_affine: bool = False,
                      disable_color_enhancer: bool = False) -> Dict[str, Any]:
    """Test enhancement on single image with optional GT evaluation"""
    # Load model
    model = load_model(model_path, device, disable_color_affine, disable_color_enhancer)
    if model is None: 
        return {'error': 'Failed to load model'}

    # Enhance image
    result = enhance_image(model, image_path, output_path, device, 
                          enable_post_process, native_resolution)
    if 'error' in result:
        return result

    print(f"✅ Enhanced:  {result.get('resolution', 'unknown')} @ {result['inference_time']:.3f}s")

    # Calculate metrics with GT if available
    if gt_path and os.path.exists(gt_path):
        try:
            # Load GT and enhanced at native resolution
            gt_tensor, _ = load_image(gt_path, device, native_resolution)
            if gt_tensor is not None: 
                gt_tensor = denormalize_imagenet(gt_tensor)

                enhanced_tensor, _ = load_image(output_path, device, native_resolution)
                if enhanced_tensor is not None:
                    enhanced_tensor = denormalize_imagenet(enhanced_tensor)

                    # Ensure same size
                    min_h = min(gt_tensor.size(2), enhanced_tensor.size(2))
                    min_w = min(gt_tensor.size(3), enhanced_tensor.size(3))
                    gt_tensor = gt_tensor[:, :, :min_h, :min_w]
                    enhanced_tensor = enhanced_tensor[:, :, :min_h, :min_w]

                    # Calculate metrics
                    psnr = calculate_psnr(enhanced_tensor, gt_tensor, normalized=False).item()
                    ssim = calculate_ms_ssim(enhanced_tensor, gt_tensor, normalized=False).item()

                    result['psnr'] = psnr
                    result['ssim'] = ssim

                    print(f"📊 PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

        except Exception as e:
            print(f"⚠️  Metrics calculation failed: {e}")

    return result


def get_test_images(test_dir: str) -> List[str]:
    """Get all test images from directory"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    test_path = Path(test_dir)
    images = set()

    for ext in extensions:
        images.update(test_path.glob(ext))
        images.update(test_path.glob(ext.upper()))

    return sorted([str(p) for p in images])


def find_ground_truth(image_path: str, gt_dir: str) -> Optional[str]:
    """Find corresponding ground truth image"""
    if not gt_dir or not os.path.exists(gt_dir):
        return None

    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]

    # Try multiple naming patterns
    patterns = [
        filename,
        base_name + '.png',
        base_name + '.jpg',
        base_name.replace('_low', '') + '.png',
        base_name.replace('_lowlight', '') + '.png',
        base_name.replace('_dark', '') + '.png',
        base_name + '_high.png',
        base_name + '_gt.png',
    ]

    for pattern in patterns: 
        gt_path = os.path.join(gt_dir, pattern)
        if os.path.exists(gt_path):
            return gt_path

    return None


def test_dataset(model_path: str, test_dir: str, output_dir: str,
                 gt_dir: str = None, device: str = 'cuda', max_images: int = None,
                 enable_post_process: bool = True, 
                 native_resolution: bool = True,
                 disable_color_affine: bool = False,
                 disable_color_enhancer: bool = False) -> tuple:
    """
    Test enhancement on dataset
    
    Args: 
        native_resolution: Test at native resolution (manuscript: True)
    """
    # Load model once
    model = load_model(model_path, device, disable_color_affine, disable_color_enhancer)
    if model is None:
        raise RuntimeError("Failed to load model")

    # Get test images
    test_images = get_test_images(test_dir)
    if max_images: 
        test_images = test_images[:max_images]

    print(f"🚀 Testing {len(test_images)} images")
    if native_resolution:
        print("📐 Native resolution mode (manuscript configuration)")
    if enable_post_process:
        print("🎨 Color post-processing enabled")

    # Initialize metrics calculator
    if ImprovedMetricsCalculator is not None:
        metrics_calc = ImprovedMetricsCalculator(device=device)
    else:
        metrics_calc = None

    all_results = []
    successful = 0
    failed = 0
    total_time = 0.0

    for i, image_path in enumerate(test_images, 1):
        base_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_enhanced.png")
        gt_path = None

        # Progress indicator
        if i % 10 == 0 or i == len(test_images):
            print(f"📷 Progress: {i}/{len(test_images)}")

        # Enhance image
        result = enhance_image(model, image_path, output_path, device, 
                              enable_post_process, native_resolution)
        if 'error' in result: 
            failed += 1
            continue

        successful += 1
        total_time += result.get('inference_time', 0.0)

        # No-reference metrics
        if metrics_calc is not None:
            try:
                orig_t, _ = load_image(image_path, device, native_resolution)
                enh_t, _ = load_image(output_path, device, native_resolution)
                if orig_t is not None and enh_t is not None:
                    orig_t = denormalize_imagenet(orig_t)
                    enh_t = denormalize_imagenet(enh_t)

                    no_ref = metrics_calc.calculate_all_metrics(
                        enhanced=enh_t, target=None, original=orig_t
                    )

                    for k in ("niqe", "brightness", "contrast", "enhancement_ratio",
                             "contrast_ratio", "loe", "colorfulness", "colorfulness_delta"):
                        if k in no_ref:
                            result[k] = float(no_ref[k])

                    # ΔNIQE
                    try:
                        niqe_input = float(metrics_calc.calculate_niqe(orig_t))
                        result["niqe_input"] = niqe_input
                        if "niqe" in result:
                            result["niqe_improvement"] = niqe_input - result["niqe"]
                    except Exception:
                        pass
            except Exception:
                pass

        # Calculate metrics if GT available
        if gt_dir and metrics_calc: 
            gt_path = find_ground_truth(image_path, gt_dir)
            if gt_path:
                try:
                    gt_tensor, _ = load_image(gt_path, device, native_resolution)
                    enhanced_tensor, _ = load_image(output_path, device, native_resolution)

                    if gt_tensor is not None and enhanced_tensor is not None:
                        gt_tensor = denormalize_imagenet(gt_tensor)
                        enhanced_tensor = denormalize_imagenet(enhanced_tensor)

                        # Crop to same size
                        min_h = min(gt_tensor.size(2), enhanced_tensor.size(2))
                        min_w = min(gt_tensor.size(3), enhanced_tensor.size(3))
                        gt_tensor = gt_tensor[:, : , :min_h, :min_w]
                        enhanced_tensor = enhanced_tensor[:, : , :min_h, :min_w]

                        # Calculate metrics
                        psnr = calculate_psnr(enhanced_tensor, gt_tensor, normalized=False).item()
                        ssim = calculate_ms_ssim(enhanced_tensor, gt_tensor, normalized=False).item()

                        result['psnr'] = psnr
                        result['ssim'] = ssim

                except Exception: 
                    pass

        result['input_path'] = image_path
        result['gt_path'] = gt_path if gt_dir else None
        all_results.append(result)

    # Calculate summary statistics
    summary = {
        'total_images': len(test_images),
        'successful':  successful,
        'failed': failed,
        'success_rate': successful / len(test_images) if test_images else 0.0,
        'post_processing_enabled': enable_post_process,
        'native_resolution': native_resolution,
        'avg_inference_time': total_time / successful if successful > 0 else 0.0,
    }

    # Aggregate metrics
    psnr_values = [r["psnr"] for r in all_results if "psnr" in r]
    ssim_values = [r["ssim"] for r in all_results if "ssim" in r]
    niqe_values = [r["niqe"] for r in all_results if "niqe" in r]
    niqe_impr_values = [r["niqe_improvement"] for r in all_results if "niqe_improvement" in r]
    loe_values = [r["loe"] for r in all_results if "loe" in r]
    cf_delta_values = [r["colorfulness_delta"] for r in all_results if "colorfulness_delta" in r]

    if psnr_values:
        summary.update({
            "avg_psnr": float(np.mean(psnr_values)),
            "std_psnr": float(np.std(psnr_values)),
            "avg_ssim": float(np.mean(ssim_values)) if ssim_values else 0.0,
        })
        print(f"\n📊 Results:  PSNR = {summary['avg_psnr']:.2f} ± {summary['std_psnr']:.2f} dB")
        if ssim_values:
            print(f"📊 Results: SSIM = {summary['avg_ssim']:.4f}")

    if niqe_values:
        summary.update({
            "avg_niqe": float(np.mean(niqe_values)),
            "std_niqe": float(np.std(niqe_values)),
        })
        print(f"📊 Results: NIQE = {summary['avg_niqe']:.2f} ± {summary['std_niqe']:.2f}")
    
    if niqe_impr_values:
        summary["avg_niqe_improvement"] = float(np.mean(niqe_impr_values))
        print(f"📊 ΔNIQE:  {summary['avg_niqe_improvement']:.2f} (↑ is better)")
    
    if loe_values:
        summary["avg_loe"] = float(np.mean(loe_values))
        print(f"📊 LOE: {summary['avg_loe']:.3f} (↓ is better)")
    
    if cf_delta_values: 
        summary["avg_colorfulness_delta"] = float(np.mean(cf_delta_values))
        print(f"📊 ΔColorfulness: {summary['avg_colorfulness_delta']:.3f}")

    print(f"⚡ Avg inference time: {summary['avg_inference_time']:.3f}s/image")

    # Save results
    results_file = os.path.join(output_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump({'summary': summary, 'detailed_results': all_results}, f, indent=2)

    print(f"✅ Results saved to {results_file}")

    return summary, all_results


def main():
    parser = argparse.ArgumentParser(
        description='RemDiNet Testing (Manuscript Configuration:  Native Resolution)'
    )

    # Model and data
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to model checkpoint')

    # Testing modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test_dir', type=str, help='Directory with test images')
    group.add_argument('--single_image', type=str, help='Path to single image')

    # Output paths
    parser.add_argument('--output_dir', type=str, default='./test_results', 
                       help='Output directory')
    parser.add_argument('--output_image', type=str, 
                       help='Output path for single image')

    # Ground truth
    parser.add_argument('--gt_dir', type=str, help='Ground truth directory (optional)')
    parser.add_argument('--gt_image', type=str, help='Ground truth for single image')

    # Options
    parser.add_argument('--max_images', type=int, help='Maximum images to process')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cuda', 'cpu'])

    # Resolution mode (Default:  native)
    parser.add_argument('--fixed_size', action='store_true',
                       help='Use fixed size (faster, but manuscript uses native resolution)')
    
    # Color post-processing
    parser.add_argument('--no_post_process', action='store_true',
                       help='Disable color post-processing')
    parser.add_argument('--no_color_affine', action='store_true',
                       help='Disable in-model color affine transformation')
    parser.add_argument('--no_color_enhancer', action='store_true',
                       help='Disable in-model color enhancer')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto': 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 80)
    print("RemDiNet Testing (Manuscript Configuration)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Resolution mode: {'Fixed size' if args.fixed_size else 'Native (manuscript)'}")
    print("=" * 80)

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return 1

    # Settings
    enable_post_process = not args.no_post_process
    native_resolution = not args.fixed_size  # Default: True
    disable_color_affine = args.no_color_affine
    disable_color_enhancer = args.no_color_enhancer

    if not enable_post_process: 
        print("🚫 Color post-processing disabled")
    if disable_color_affine:
        print("🚫 In-model color affine disabled")
    if disable_color_enhancer:
        print("🚫 In-model color enhancer disabled")

    try:
        if args.single_image:
            # Single image testing
            if not os.path.exists(args.single_image):
                print(f"❌ Image not found: {args.single_image}")
                return 1

            output_path = args.output_image or os.path.join(
                args.output_dir, f"{Path(args.single_image).stem}_enhanced.png"
            )

            print(f"🖼️  Processing:  {args.single_image}")
            result = test_single_image(
                args.model_path, args.single_image, output_path,
                args.gt_image, device, enable_post_process, native_resolution,
                disable_color_affine, disable_color_enhancer
            )

            if 'error' in result:
                print(f"❌ {result['error']}")
                return 1

            print(f"✅ Enhanced image saved:  {output_path}")
            print(f"⚡ Inference time: {result['inference_time']:.3f}s")

        else:
            # Dataset testing
            if not os.path.exists(args.test_dir):
                print(f"❌ Test directory not found: {args.test_dir}")
                return 1

            summary, _ = test_dataset(
                args.model_path, args.test_dir, args.output_dir,
                args.gt_dir, device, args.max_images, 
                enable_post_process, native_resolution,
                disable_color_affine, disable_color_enhancer
            )

            print(f"\n✅ Processed {summary['successful']}/{summary['total_images']} images")
            if summary['failed'] > 0:
                print(f"⚠️  {summary['failed']} images failed")

        return 0

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
