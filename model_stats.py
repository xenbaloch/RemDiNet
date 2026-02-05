"""
Model Statistics Calculator for RemDiNet
Outputs: GFLOPs, Parameters (Total/Trainable), Inference Time
"""

import torch
import time
from model import UnifiedLowLightEnhancer

# ============================================================================
# Configuration
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = (1, 3, 256, 256)  # Batch, Channels, Height, Width
WARMUP_RUNS = 20
TIMED_RUNS = 100
USE_AMP = True  # Mixed precision inference

MODEL_CONFIG = {
    'num_iterations': 8,
    'use_snr_awareness': True,
    'use_semantic_guidance': True,
    'use_learnable_snr': True,
    'use_contrast_refinement': False,
}

# ============================================================================
# Helper Functions
# ============================================================================

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def calculate_flops(model, input_tensor):
    """Calculate FLOPs using thop library"""
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return flops
    except ImportError:
        print("Warning: thop not installed. Run: pip install thop")
        return None
    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
        return None


def measure_inference_time(model, input_tensor, warmup=20, runs=100, use_amp=True):
    """Measure average inference time"""
    model.eval()

    # Warmup
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                for _ in range(warmup):
                    _ = model(input_tensor)
        else:
            for _ in range(warmup):
                _ = model(input_tensor)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    start = time.time()
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                for _ in range(runs):
                    _ = model(input_tensor)
        else:
            for _ in range(runs):
                _ = model(input_tensor)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_time = elapsed / runs
    fps = 1.0 / avg_time

    return avg_time, fps


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("RemDiNet Model Statistics")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Input Size: {INPUT_SIZE[2]}×{INPUT_SIZE[3]}")
    print(f"Mixed Precision: {'Enabled' if USE_AMP else 'Disabled'}")
    print("=" * 70)

    # Initialize model
    model = UnifiedLowLightEnhancer(**MODEL_CONFIG).to(DEVICE).eval()
    input_tensor = torch.randn(*INPUT_SIZE, device=DEVICE)

    # 1. Count parameters
    total_params, trainable_params = count_parameters(model)

    # 2. Calculate FLOPs
    flops = calculate_flops(model, input_tensor)

    # 3. Measure inference time
    avg_time, fps = measure_inference_time(
        model, input_tensor,
        warmup=WARMUP_RUNS,
        runs=TIMED_RUNS,
        use_amp=USE_AMP
    )

    # ========================================================================
    # Results
    # ========================================================================
    print("\nRESULTS")
    print("-" * 70)
    print(f"Parameters (Total):      {total_params:>15,} ({total_params/1e6:.2f} M)")
    print(f"Parameters (Trainable):  {trainable_params:>15,} ({trainable_params/1e6:.2f} M)")

    if flops:
        print(f"GFLOPs:                  {flops/1e9:>15.2f}")

    print(f"Inference Time (avg):    {avg_time*1000:>15.2f} ms")
    print(f"Throughput (FPS):        {fps:>15.1f} images/sec")
    print("=" * 70)

    # ========================================================================
    # Summary Table (Copy-paste friendly)
    # ========================================================================
    print("\nSUMMARY")
    print("-" * 70)
    print(f"{'Metric':<30} {'Value'}")
    print("-" * 70)
    print(f"{'Total Parameters (M)':<30} {total_params/1e6:.2f}")
    print(f"{'Trainable Parameters (M)':<30} {trainable_params/1e6:.2f}")
    if flops:
        print(f"{'GFLOPs':<30} {flops/1e9:.2f}")
    print(f"{'Inference Time (ms)':<30} {avg_time*1000:.2f}")
    print(f"{'FPS':<30} {fps:.1f}")
    print("-" * 70)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops,
        'inference_time_ms': avg_time * 1000,
        'fps': fps
    }


if __name__ == "__main__":
    results = main()