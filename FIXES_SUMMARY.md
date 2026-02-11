
```

### Stage Weight Updates
**Stage 1** (Reconstruction warm-up):
- `w_reconstruction`: 1.00 → 1.20
- `w_color_consistency`: 0.30 → 0.20
- `w_color_diversity`: 0.50 → 0.30
- `w_ssim`: 0.50 → 0.60

**Stage 2** (Structure + color):
- `w_edge`: 0.09 → 0.12
- `w_ssim`: 0.25 → 0.45
- `w_color_consistency`: 0.05 → 0.07
- `w_color_diversity`: 0.05 → 0.07
- `w_mask_mean`: 0.05 → 0.03

**Stage 3** (Perceptual fine-tune):
- `w_reconstruction`: 0.95 → 0.90
- `w_exposure`: 0.08 → 0.04
- `w_edge`: 0.18 → 0.20
- `w_ssim`: 0.28 → 0.45
- `w_color_consistency`: 0.15 → 0.10
- `w_color_diversity`: 0.20 → 0.12
- `w_mask_mean`: 0.10 → 0.08
- `w_perceptual`: 0.04 → 0.06

### COLOR_THRESHOLDS Updates
- `min_saturation`: 0.18 → 0.10 (reduces false LOW_COLOR flags)
- `critical_saturation`: 0.05 → 0.04
- `target_saturation`: 0.35 → 0.30
- `saturation_penalty_factor`: 1.5 → 1.0


**Impact**: Proper reconstruction supervision throughout training, better PSNR/SSIM convergence.

---



