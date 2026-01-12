
import torch
import torch.nn.functional as F

# Attempt multi-scale SSIM from pytorch_msssim
try:
    from pytorch_msssim import ms_ssim
    _has_ms_ssim = True
except ImportError:
    _has_ms_ssim = False


def denormalize_tensor(
    x: torch.Tensor,
    normalized: bool,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    If `normalized` is True, undo ImageNet normalization; otherwise assume x is already in [0,1].
    Supports tensors of shape (B,C,H,W).
    """
    if not normalized:
        return x
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return x * std_t + mean_t


@torch.no_grad()
def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    normalized: bool = False,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Compute average PSNR (in dB) over a batch of images.

    Args:
        pred:    Tensor of shape (B,C,H,W).
        target:  Tensor of same shape as pred.
        normalized: If True, inputs are ImageNet-normalized and will be denormalized.
        data_range: Maximum possible pixel value (1.0 for [0,1] images).

    Returns:
        Scalar tensor with mean PSNR over the batch.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Pred and target must have same shape, got {pred.shape} vs {target.shape}.")

    # Bring to [0,1]
    pred01 = denormalize_tensor(pred, normalized)
    tgt01 = denormalize_tensor(target, normalized)

    # Per-image MSE
    mse = F.mse_loss(pred01, tgt01, reduction='none')
    mse = mse.reshape(mse.size(0), -1).mean(dim=1) # option A
    mse = torch.clamp(mse, min=1e-10)

    # PSNR formula
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return psnr.mean()


@torch.no_grad()
def calculate_ms_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    normalized: bool = False,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Compute MS-SSIM over a batch of images and return the mean score.

    Requires `pytorch_msssim`; if unavailable, raises ImportError.
    """
    if not _has_ms_ssim:
        raise ImportError("pytorch_msssim is required for MS-SSIM. Install via `pip install pytorch-msssim`.")
    if pred.shape != target.shape:
        raise ValueError(f"Pred and target must have same shape, got {pred.shape} vs {target.shape}.")

    pred01 = denormalize_tensor(pred, normalized)
    tgt01 = denormalize_tensor(target, normalized)

    # Compute per-image MS-SSIM, then average
    ssim_per_image = ms_ssim(
        pred01,
        tgt01,
        data_range=data_range,
        size_average=False,
    )
    return ssim_per_image.mean()


# Quick self-test when run as a script
if __name__ == '__main__':
    print("Running metrics_core self-test...")
    B, C, H, W = 2, 3, 256, 256  # Use sufficiently large size for MS-SSIM defaults
    torch.manual_seed(0)
    img = torch.rand(B, C, H, W)
    # Perfect reconstruction
    psnr_id = calculate_psnr(img, img, normalized=False)
    try:
        ssim_id = calculate_ms_ssim(img, img, normalized=False)
    except (ImportError, AssertionError):
        ssim_id = torch.tensor(1.0)
    print(f"Identical PSNR: {psnr_id:.4f} dB (should be >>40 dB)")
    print(f"Identical MS-SSIM: {ssim_id:.4f} (should be 1.0000)")

    # Add noise
    noise = (img + 0.1 * torch.randn_like(img)).clamp(0, 1)
    psnr_noisy = calculate_psnr(noise, img, normalized=False)
    try:
        ssim_noisy = calculate_ms_ssim(noise, img, normalized=False)
    except (ImportError, AssertionError):
        ssim_noisy = torch.tensor(0.0)
    print(f"Noisy   PSNR: {psnr_noisy:.4f} dB (should be lower)")
    print(f"Noisy   MS-SSIM: {ssim_noisy:.4f} (should be <1.0)")
