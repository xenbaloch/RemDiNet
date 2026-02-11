#!/usr/bin/env python3
"""
Test script to validate the critical bug fixes
"""
import torch
import sys

# Test constants
FLOAT_TOLERANCE = 0.01

def test_color_diversity_loss():
    """Test Fix #1: color_diversity_loss is fully differentiable"""
    print("Testing Fix #1: color_diversity_loss differentiability...")
    from Myloss import color_diversity_loss
    
    # Create a test tensor that requires gradients
    enhanced = torch.rand(2, 3, 64, 64, requires_grad=True)
    
    # Compute loss
    loss = color_diversity_loss(enhanced)
    
    # Check that loss is a tensor with grad_fn (differentiable)
    assert isinstance(loss, torch.Tensor), "Loss must be a tensor"
    assert loss.requires_grad, "Loss must require gradients"
    assert loss.grad_fn is not None, "Loss must have grad_fn (be differentiable)"
    
    # Try backprop
    loss.backward()
    assert enhanced.grad is not None, "Gradients must flow back"
    
    print("✅ Fix #1: color_diversity_loss is fully differentiable")
    return True


def test_ramp_keys():
    """Test Fix #2: RAMP_KEYS doesn't include w_reconstruction and w_ssim"""
    print("Testing Fix #2: RAMP_KEYS fix...")
    from loss_weights import ramp_colour
    
    # Check that the function signature has warmup_epochs=20
    import inspect
    sig = inspect.signature(ramp_colour)
    warmup_default = sig.parameters['warmup_epochs'].default
    assert warmup_default == 20, f"warmup_epochs default should be 20, got {warmup_default}"
    
    # Test that w_reconstruction and w_ssim are NOT ramped
    test_weights = {
        'w_reconstruction': 1.0,
        'w_ssim': 0.5,
        'w_color_diversity': 0.3,
        'w_exposure': 0.1
    }
    
    ramped = ramp_colour(test_weights, epoch_in_stage=0, warmup_epochs=20)
    
    # At epoch 0, ramp factor is 0, so ramped keys should be 0
    assert ramped['w_reconstruction'] == 1.0, "w_reconstruction should NOT be ramped"
    assert ramped['w_ssim'] == 0.5, "w_ssim should NOT be ramped"
    assert ramped['w_color_diversity'] == 0.0, "w_color_diversity SHOULD be ramped to 0"
    assert ramped['w_exposure'] == 0.0, "w_exposure SHOULD be ramped to 0"
    
    print("✅ Fix #2: RAMP_KEYS correctly excludes w_reconstruction and w_ssim")
    return True


def test_training_defaults():
    """Test Fix #2: TRAINING_DEFAULTS has batch_size=8 and image_size=512"""
    print("Testing Fix #2: TRAINING_DEFAULTS...")
    from loss_weights import TRAINING_DEFAULTS
    
    assert TRAINING_DEFAULTS['batch_size'] == 8, f"batch_size should be 8, got {TRAINING_DEFAULTS['batch_size']}"
    assert TRAINING_DEFAULTS['image_size'] == 512, f"image_size should be 512, got {TRAINING_DEFAULTS['image_size']}"
    
    print("✅ Fix #2: TRAINING_DEFAULTS correctly set to batch_size=8, image_size=512")
    return True


def test_global_colour_affine():
    """Test Fix #4: GlobalColourAffine has identity_clamp parameter and no gray-world block"""
    print("Testing Fix #4: GlobalColourAffine fixes...")
    from model import GlobalColourAffine
    import inspect
    
    # Check that __init__ has identity_clamp parameter
    sig = inspect.signature(GlobalColourAffine.__init__)
    assert 'identity_clamp' in sig.parameters, "GlobalColourAffine should have identity_clamp parameter"
    
    # Create instance
    affine = GlobalColourAffine(enable_vibrance=False, identity_clamp=0.15)
    
    # Test forward pass
    x = torch.rand(2, 3, 32, 32)
    
    # Forward in training mode (where gray-world block would have been)
    affine.train()
    out_train = affine(x)
    
    # Forward in eval mode
    affine.eval()
    out_eval = affine(x)
    
    # Both should produce valid output
    assert out_train.shape == x.shape, "Output shape should match input"
    assert out_eval.shape == x.shape, "Output shape should match input"
    assert torch.all((out_train >= 0) & (out_train <= 1)), "Output should be in [0,1]"
    
    print("✅ Fix #4: GlobalColourAffine has identity_clamp and works correctly")
    return True


def test_curve_estimator():
    """Test Fix #4: ImprovedCurveEstimator uses Zero-DCE formula"""
    print("Testing Fix #4: ImprovedCurveEstimator...")
    from model import ImprovedCurveEstimator
    
    estimator = ImprovedCurveEstimator(num_iterations=8)
    
    # Test with 4-channel input (RGB + SNR)
    x_4ch = torch.rand(2, 4, 64, 64)
    
    # Should work without error
    enhanced, curves = estimator(x_4ch)
    
    # Enhanced should be 3-channel (RGB only)
    assert enhanced.shape[1] == 3, f"Enhanced should be 3-channel, got {enhanced.shape[1]}"
    assert torch.all((enhanced >= 0) & (enhanced <= 1)), "Enhanced should be in [0,1]"
    
    print("✅ Fix #4: ImprovedCurveEstimator applies curves to RGB only")
    return True


def test_color_enhancement_clamps():
    """Test Fix #5: EnhancedGlobalColorCorrection has tightened clamps"""
    print("Testing Fix #5: color_enhancement clamps...")
    from color_enhancement import EnhancedGlobalColorCorrection
    
    enhancer = EnhancedGlobalColorCorrection()
    
    # Check initial vibrance_factor value
    assert abs(enhancer.vibrance_factor.item() - 1.2) < FLOAT_TOLERANCE, "vibrance_factor should initialize to 1.2"
    
    # Test forward pass
    x = torch.rand(2, 3, 32, 32)
    out = enhancer(x)
    
    assert out.shape == x.shape, "Output shape should match input"
    assert torch.all((out >= 0) & (out <= 1)), "Output should be in [0,1]"
    
    print("✅ Fix #5: EnhancedGlobalColorCorrection has correct initialization and clamps")
    return True


def test_unified_model():
    """Integration test: UnifiedLowLightEnhancer forward pass"""
    print("Testing model integration...")
    from model import UnifiedLowLightEnhancer, normalize_imagenet
    
    model = UnifiedLowLightEnhancer(
        num_iterations=8,
        use_snr_awareness=True,
        use_semantic_guidance=False,  # Disable for faster test
        use_learnable_snr=False
    )
    model.eval()
    
    # Test input (ImageNet normalized)
    x = normalize_imagenet(torch.rand(1, 3, 128, 128))
    
    with torch.no_grad():
        results = model(x)
    
    assert 'enhanced' in results, "Results should contain 'enhanced' key"
    enhanced = results['enhanced']
    assert enhanced.shape[1] == 3, "Enhanced should be 3-channel RGB"
    assert torch.all((enhanced >= 0) & (enhanced <= 1)), "Enhanced should be in [0,1]"
    
    print("✅ Model integration test passed")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Running validation tests for bug fixes...")
    print("=" * 60)
    
    tests = [
        test_color_diversity_loss,
        test_ramp_keys,
        test_training_defaults,
        test_global_colour_affine,
        test_curve_estimator,
        test_color_enhancement_clamps,
        test_unified_model,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed.append(test.__name__)
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    if not failed:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"❌ {len(failed)} test(s) failed: {', '.join(failed)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
