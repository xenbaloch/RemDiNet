import math

# -----------------------------------------------------------------------------
#  Generic (stage‑independent) base weights – seldom touched during training
# -----------------------------------------------------------------------------
LOSS_WEIGHTS = {
    "w_reconstruction": 0.80,
    "w_exposure":        1.50,
    "w_edge":            0.50,
    "w_ssim":            0.40,
    "w_color_consistency": 0.60,
    "w_perceptual":      0.10,
    "w_color_diversity": 0.40,
    "w_mask_mean":       0.10,
    "w_affine_decay":    1e-4,
}

# -----------------------------------------------------------------------------
#  Stage‑specific overrides (copied & blended inside the trainer)
# -----------------------------------------------------------------------------
STAGE_WEIGHTS = {
    # Stage‑1  – reconstruction warm‑up (colour kept modest)
    1: {
        "w_reconstruction": 1.00,
        "w_color_consistency": 0.30,
        "w_color_diversity":   0.50,
        "w_exposure":          0.00,
        "w_edge":              0.00,
        "w_ssim":              0.50,
        "w_perceptual":        0.00,
        "w_mask_mean":         0.00,
    },
    # Stage‑2  – structure + colour (after switch at epoch~15)
    2: {
        "w_reconstruction":    1.35,
        "w_exposure":          0.07,
        "w_edge":              0.09,
        "w_ssim":              0.25,
        "w_color_consistency": 0.05,
        "w_color_diversity":   0.05,
        "w_mask_mean":         0.05,
        "w_perceptual":        0.00,
    },
    # Stage‑3  – perceptual fine‑tune
    3: {
        "w_reconstruction":    0.95,
        "w_exposure":          0.08,
        "w_edge":              0.18,
        "w_ssim":              0.28,
        "w_color_consistency": 0.15,
        "w_color_diversity":   0.20,
        "w_mask_mean":         0.10,
        "w_perceptual":        0.04,
    },
}

# -----------------------------------------------------------------------------
#  Early‑stopping rules (per stage)
# -----------------------------------------------------------------------------
EARLY_STOPPING_CONFIG = {
    "stage_1": {"patience": 30, "min_delta": 0.10},
    "stage_2": {"patience": 18, "min_delta": 0.15},
    "stage_3": {"patience": 15, "min_delta": 0.10},
}

# -----------------------------------------------------------------------------
#  Colour‑preservation thresholds used by validation hooks
# -----------------------------------------------------------------------------
COLOR_THRESHOLDS = {
    "min_saturation": 0.18,   # match real val‑set stats
    "critical_saturation": 0.05,
    "target_saturation":   0.35,
    "saturation_penalty_factor": 1.5,  # softer penalty → PSNR not masked
}

# -----------------------------------------------------------------------------
#  Optim‑level defaults (unchanged)
# -----------------------------------------------------------------------------
TRAINING_DEFAULTS = {
    "learning_rate":              5e-4,
    "learning_rate_curves_multiplier": 2.0,
    "weight_decay":               1e-4,
    "gradient_clip_norm":         0.5,
    "batch_size":                 4,
    "image_size":                 320,
    "num_epochs":                 60,
    "val_split":                  0.2,
}

# -----------------------------------------------------------------------------
#  Helper API used by the trainer
# -----------------------------------------------------------------------------

def get_stage_weights(stage: int) -> dict:
    """Return a *copy* of the weight dict for a given stage."""
    if stage not in STAGE_WEIGHTS:
        raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3.")
    return STAGE_WEIGHTS[stage].copy()


def get_early_stopping_config(stage: int) -> dict:
    """Early‑stopping parameters for the given stage (copy)."""
    return EARLY_STOPPING_CONFIG[f"stage_{stage}"].copy()


# ----  Cosine ramp‑in for colour weights  -----------------------------------

def ramp_colour(weights: dict, epoch_in_stage: int, warmup_epochs: int = 15) -> dict:
    """Cosine‑ramp colour‑related weights during the first *warmup_epochs*
    after a stage switch – prevents sudden PSNR dip when colour terms appear."""
    if epoch_in_stage >= warmup_epochs:
        return weights  # nothing to do

    out = weights.copy()
    ramp = 0.5 * (1 - math.cos(math.pi * epoch_in_stage / warmup_epochs))

    RAMP_KEYS = (
        "w_reconstruction", "w_color_consistency", "w_color_diversity",
        "w_exposure", "w_edge", "w_ssim"
    )

    for k in RAMP_KEYS:
        if k in out:
            out[k] *= ramp
    return out


def get_adaptive_color_weights(current_saturation: float, base_weights: dict | None = None) -> dict:
    """Optional dynamic tweak based on current image saturation."""
    if base_weights is None:
        base_weights = LOSS_WEIGHTS
    adapted = base_weights.copy()

    if current_saturation < COLOR_THRESHOLDS["min_saturation"]:
        boost = min(3.0, (COLOR_THRESHOLDS["min_saturation"] - current_saturation) * 5.0 + 1.0)
        adapted["w_color_diversity"]   *= boost
        adapted["w_color_consistency"] *= boost * 0.8
        adapted["w_reconstruction"]    *= 0.9
    return adapted

# -----------------------------------------------------------------------------
__all__ = [
    "LOSS_WEIGHTS", "STAGE_WEIGHTS", "EARLY_STOPPING_CONFIG",
    "COLOR_THRESHOLDS", "TRAINING_DEFAULTS",
    "get_stage_weights", "get_early_stopping_config",
    "get_adaptive_color_weights", "ramp_colour",
]
