import os
import sys
import argparse
import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import random_split, DataLoader

# Local imports
from model import UnifiedLowLightEnhancer, denormalize_imagenet
from dataloader import ImprovedLowLightDataset
from metrics_core import calculate_psnr, calculate_ms_ssim
from loss_weights import (
    LOSS_WEIGHTS, get_stage_weights, COLOR_THRESHOLDS,
    EARLY_STOPPING_CONFIG, ramp_colour          # ← NEW
)

def setup_seeds():
    """Set random seeds for reproducibility"""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

def create_loss_function(stage: int, device: str):
    """Create stage-specific loss function using loss_weights.py"""
    from Myloss import SimplifiedDistractionAwareLoss, SimpleReconstructionLoss

    weights = get_stage_weights(stage)

    if stage == 1:
        return SimpleReconstructionLoss()
    else:
        return SimplifiedDistractionAwareLoss(device=device, **weights)

class MinimalTrainer:
    """Minimal trainer with clean logging and warm stage transitions"""

    def __init__(self, model, device, config, resume_ckpt: str | None = None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        # Will be set at the start of train()
        self.max_epochs: int | None = None

        # Training state
        self.epoch = 0
        self.stage = 1
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        # ── NEW: stage-scoped bests and Stage-3 score ──
        self.best_psnr_s1 = 0.0
        self.best_psnr_s2 = 0.0
        self.best_score_s3 = float('-inf')

        # gradient-accumulation factor
        self.grad_accum = config.get('grad_accum', 1)

        # NEW: Stage transition tracking
        self.stage_switch_epoch = 0
        self.loss_mix = 0.0  # 0 = old loss, 1 = new loss
        self.old_criterion = None

        # ─── NEW: keep a running history ─────
        self.history = {
            "train_loss": [],
            "train_psnr": [],
            "val_psnr": [],
            "val_ssim": [],
            "learning_rate": []
        }

        # Stage thresholds (filled later from total epochs)
        self.stage_transitions = [None, None]  # will be computed in train()

        # Loss functions for each stage
        self.loss_functions = {
            1: create_loss_function(1, device),
            2: create_loss_function(2, device),
            3: create_loss_function(3, device)
        }
        self.criterion = self.loss_functions[1]

        if hasattr(self.criterion, "model_affine_ref"):
            self.criterion.model_affine_ref = self.model.colour_head

        # Optimiser – dedicate its own LR to the mask so it can catch up fast
        curve_params = [p for n, p in model.named_parameters() if 'curve_estimator' in n]
        mask_params = [p for n, p in model.named_parameters() if 'mask_predictor' in n]
        other_params = [
            p for n, p in model.named_parameters()
            if 'curve_estimator' not in n and 'mask_predictor' not in n
        ]

        self.optimizer = AdamW([
            {'params': other_params, 'lr': config['lr'] * 0.5, 'name': 'other'},
            {'params': curve_params, 'lr': config['lr'], 'name': 'curve'},
            {'params': mask_params, 'lr': config['lr'], 'name': 'mask'},
        ], weight_decay=3e-4)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=8, verbose=False
        )

        # Early stopping
        self.patience_count = 0
        self.best_score = float('-inf')

        # Output
        self.output_dir = config['output_dir']
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)

        # ── NEW: checkpointing/monitor config ──
        self.save_best_stage = int(config.get('save_best_stage', 3))
        self.monitor_stage3 = str(config.get('monitor_stage3', 'composite')).lower()
        self.s3_psnr_weight = float(config.get('s3_psnr_weight', 0.7))
        self.s3_ssim_weight = float(config.get('s3_ssim_weight', 0.3))

        # ───────── OPTIONAL RESUME ─────────
        if resume_ckpt and os.path.isfile(resume_ckpt):
            self._load_checkpoint(resume_ckpt)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        try:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception:
            pass

        # Restore counters/ best metrics
        self.epoch = ckpt.get('epoch', 0) + 1
        self.stage = ckpt.get('stage', 1)
        self.best_psnr = ckpt.get('best_psnr', 0.0)
        self.best_ssim = ckpt.get('best_ssim', 0.0)
        self.history = ckpt.get('training_history', self.history)

        # Make sure criterion matches the restored stage
        self.criterion = self.loss_functions[self.stage]
        print(f"🔄  Resumed from '{os.path.basename(path)}' "
              f"(epoch {self.epoch}, stage {self.stage})")

    def update_stage(self):
        """Update training stage with warm transitions"""
        if self.epoch < self.stage_transitions[0]:
            new_stage = 1
        elif self.epoch < self.stage_transitions[1]:
            new_stage = 2
        else:
            new_stage = 3

        if new_stage != self.stage:
            print(f"🔄 Stage {self.stage} → {new_stage} (warm transition)")

            # Store old criterion for mixing
            self.old_criterion = self.criterion

            # Switch to the new stage-specific criterion
            self.criterion = self.loss_functions[new_stage]
            if hasattr(self.criterion, "model_affine_ref"):
                self.criterion.model_affine_ref = self.model.colour_head

            self.stage = new_stage
            self.stage_switch_epoch = self.epoch
            self.loss_mix = 0.0  # Start with old loss, gradually mix in new

            # Reset early stopping
            self.patience_count = 0

            # ───────── CRITICAL FIX: Don't drop LR when changing stages ─────────
            if new_stage == 2:
                # Freeze mask predictor initially for gradual adaptation
                for p in self.model.mask_predictor.parameters():
                    p.requires_grad_(False)

                # ⤵ UN-freeze colour modules so they start receiving grads
                for p in self.model.colour_head.parameters():
                    p.requires_grad_(True)
                if hasattr(self.model, "color_enhancer") and self.model.color_enhancer:
                    for p in self.model.color_enhancer.parameters():
                        p.requires_grad_(True)

                pass

                # Refresh LR scheduler to cosine, annealing over the remaining epochs
            remaining = (self.max_epochs - self.epoch) if self.max_epochs else 60
            remaining = max(10, int(remaining))
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=remaining, eta_min=1e-6)

    def get_mixed_loss(self, results, target, input_img):
        """Get mixed loss during warm transition periods"""
        if self.stage == 1 or self.old_criterion is None:
            # No mixing in stage 1 or if no old criterion
            return self.criterion(results, target, input_img, self.epoch)

        if target is None:
            return self.criterion(results, None, input_img, self.epoch)

        # Calculate mixing factor (0 → 1) during warm transitions
        epochs_since_switch = self.epoch - self.stage_switch_epoch

        if self.stage == 2 and epochs_since_switch < 15:
            # update mix factor 0→1 and compute blended loss EVERY epoch
            self.loss_mix = epochs_since_switch / 15.0

            # compute both losses
            old_loss_out = self.old_criterion(results, target, input_img, self.epoch)
            new_loss_out = self.criterion(results, target, input_img, self.epoch)

            old_loss = old_loss_out.get('total_loss', old_loss_out) if isinstance(old_loss_out, dict) else old_loss_out
            new_loss = new_loss_out.get('total_loss', new_loss_out) if isinstance(new_loss_out, dict) else new_loss_out

            mixed_loss = (1 - self.loss_mix) * old_loss + self.loss_mix * new_loss

            if isinstance(new_loss_out, dict):
                out = new_loss_out.copy()
                out['total_loss'] = mixed_loss
                out['mix_factor'] = self.loss_mix
                return out
            else:
                return mixed_loss

        if epochs_since_switch == 15 and self.stage == 2:
            for p in self.model.mask_predictor.parameters():
                p.requires_grad_(True)

            # slow the mask by 5×
            for g in self.optimizer.param_groups:
                if g.get('name') == 'mask':
                    g['lr'] *= 0.10 # from 0.20

            # Compute both losses
            with torch.no_grad():
                old_loss_out = self.old_criterion(
                    results, target, input_img, self.epoch
                )
            new_loss_out = self.criterion(results, target, input_img, self.epoch)

            old_loss = old_loss_out.get('total_loss', old_loss_out) if isinstance(old_loss_out, dict) else old_loss_out
            new_loss = new_loss_out.get('total_loss', new_loss_out) if isinstance(new_loss_out, dict) else new_loss_out

            # Mix the losses
            mixed_loss = (1 - self.loss_mix) * old_loss + self.loss_mix * new_loss

            # Return mixed result
            if isinstance(new_loss_out, dict):
                result = new_loss_out.copy()
                result['total_loss'] = mixed_loss
                result['mix_factor'] = self.loss_mix
                return result
            else:
                return mixed_loss

        # Warm ramp for Stage‑3 as well (shorter; mainly to ease perceptual on‑ramp)
        if self.stage == 3 and epochs_since_switch < 5 and self.old_criterion is not None:
            self.loss_mix = epochs_since_switch / 5.0
            old_loss_out = self.old_criterion(results, target, input_img, self.epoch)
            new_loss_out = self.criterion(results, target, input_img, self.epoch)
            old_loss = old_loss_out.get('total_loss', old_loss_out) if isinstance(old_loss_out, dict) else old_loss_out
            new_loss = new_loss_out.get('total_loss', new_loss_out) if isinstance(new_loss_out, dict) else new_loss_out
            mixed_loss = (1 - self.loss_mix) * old_loss + self.loss_mix * new_loss
            if isinstance(new_loss_out, dict):
                out = new_loss_out.copy()
                out['total_loss'] = mixed_loss
                out['mix_factor'] = self.loss_mix
                return out
            return mixed_loss

        return self.criterion(results, target, input_img, self.epoch)


    def calculate_saturation(self, tensor):
        """Calculate color saturation using metrics"""
        with torch.no_grad():
            max_vals = tensor.max(dim=1)[0]
            min_vals = tensor.min(dim=1)[0]
            return ((max_vals - min_vals) / (max_vals + 1e-8)).mean().item()

    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # ─── Cosine-ramp colour weights for Stage-2 warm-up ───
        epoch_in_stage = self.epoch - self.stage_switch_epoch
        if self.stage == 2:
            colour_w = ramp_colour(get_stage_weights(2), epoch_in_stage, warmup_epochs=10)
            for k, v in colour_w.items():  # push into current loss object
                if hasattr(self.criterion, k):
                    setattr(self.criterion, k, v)
        elif self.stage == 3:
            warm = max(1, int(min(5, self.stage_transitions[1] - self.stage_switch_epoch)))
            perc = min(1.0, epoch_in_stage / warm)
            if hasattr(self.criterion, "w_perceptual"):
                base = get_stage_weights(3).get("w_perceptual", 0.02)
                self.criterion.w_perceptual = float(base) * float(perc)

        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_sat = 0.0
        valid_batches = 0
        supervised_batches = 0

        for i, (low_light, ground_truth) in enumerate(train_loader):
            low_light = low_light.to(self.device)
            low_light_rgb = denormalize_imagenet(low_light)
            has_gt = ground_truth is not None

            if has_gt:
                ground_truth = ground_truth.to(self.device)

            try:
                # Forward pass
                results = self.model(low_light)
                enhanced = results.get('enhanced', results) if isinstance(results, dict) else results

                # Check color saturation
                saturation = self.calculate_saturation(enhanced)
                epoch_sat += saturation

                # Use mixed loss during warm transitions
                loss_out = self.get_mixed_loss(results,
                                               ground_truth if has_gt else None,
                                               low_light_rgb)

                loss = loss_out.get('total_loss', loss_out) if isinstance(loss_out, dict) else loss_out
                loss = torch.clamp(loss, 0.0, 3.0)

                if not torch.isfinite(loss):
                    continue

                # Backward pass with grad-accumulation scaling
                (loss / self.grad_accum).backward()

                # ─── NEW: step only every grad_accum batches ───
                if (i + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Accumulate metrics
                epoch_loss += loss.item()
                valid_batches += 1

                if has_gt:
                    psnr = calculate_psnr(enhanced, ground_truth, normalized=False)
                    epoch_psnr += psnr.item()
                    supervised_batches += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()  # keep training, just skip this batch
                    print("⚠️  OOM – batch skipped")
                    continue

                # Anything else: show the traceback once and abort the epoch
                import traceback, sys
                traceback.print_exc()
                sys.exit(1)

        # Step once more if there is a remainder micro-batch
        total_batches = valid_batches
        if total_batches % self.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Return epoch metrics
        return {
            'loss': epoch_loss / max(valid_batches, 1),
            'psnr': epoch_psnr / max(supervised_batches, 1),
            'saturation': epoch_sat / max(valid_batches, 1),
            'supervised': supervised_batches,
            'total': valid_batches
        }

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()

        total_psnr = 0.0
        total_ssim = 0.0
        total_sat = 0.0
        count = 0

        with torch.no_grad():
            for low_light, ground_truth in val_loader:
                if ground_truth is None:
                    continue

                try:
                    low_light = low_light.to(self.device)
                    ground_truth = ground_truth.to(self.device)

                    results = self.model(low_light)
                    enhanced = results.get('enhanced', results) if isinstance(results, dict) else results

                    # Ensure same size
                    h = min(enhanced.size(2), ground_truth.size(2))
                    w = min(enhanced.size(3), ground_truth.size(3))
                    enhanced = enhanced[:, :, :h, :w]
                    ground_truth = ground_truth[:, :, :h, :w]

                    # Calculate metrics using metrics_core
                    psnr = calculate_psnr(enhanced, ground_truth, normalized=False)
                    ssim = calculate_ms_ssim(enhanced, ground_truth, normalized=False)
                    saturation = self.calculate_saturation(enhanced)

                    total_psnr += psnr.item()
                    total_ssim += ssim.item()
                    total_sat += saturation
                    count += 1

                except Exception:
                    continue

        if count == 0:
            return {'psnr': 0.0, 'ssim': 0.0, 'saturation': 0.0}

        return {
            'psnr': total_psnr / count,
            'ssim': total_ssim / count,
            'saturation': total_sat / count
        }

    def check_early_stopping(self, score):
        """Check early stopping using loss_weights.py config"""
        config = EARLY_STOPPING_CONFIG[f'stage_{self.stage}']
        patience = config['patience']
        min_delta = config['min_delta']

        if score > self.best_score + min_delta:
            self.best_score = score
            self.patience_count = 0
            return False
        else:
            self.patience_count += 1
            return self.patience_count >= patience

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'stage': self.stage,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'config': self.config,
            'training_history': self.history
        }

        filepath = os.path.join(self.output_dir, 'checkpoints', filename)
        torch.save(checkpoint, filepath)

    def train(self, train_loader, val_loader, max_epochs):
        """Main training loop with minimal logging"""
        # remember total planned epochs and compute stage boundaries
        self.max_epochs = int(max_epochs)
        # Fractions (S1, S3) are configurable; Stage‑2 gets the remainder
        s1_frac = float(self.config.get('stage1_frac', 0.25))  # 25% by default
        s3_frac = float(self.config.get('stage3_frac', 0.20))  # 20% by default
        s1 = max(1, int(round(s1_frac * self.max_epochs)))
        s3 = max(1, int(round(s3_frac * self.max_epochs)))
        s2_end = max(s1 + 1, self.max_epochs - s3)
        # Stage 1: [0, s1-1], Stage 2: [s1, s2_end-1], Stage 3: [s2_end, max_epochs-1]
        self.stage_transitions = [s1, s2_end]
        print(f"🧭 Stage plan — S1:0–{s1 - 1}, S2:{s1}–{s2_end - 1}, S3:{s2_end}–{self.max_epochs - 1}")

        print(f"Training: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")
        print("Epoch | Stage | Train PSNR | Val PSNR | Val SSIM | Saturation | Mix | Status")
        print("-" * 80)

        start_time = time.time()

        for epoch in range(max_epochs):
            self.epoch = epoch
            self.update_stage()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # ─── NEW: record training metrics ──────
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_psnr"].append(train_metrics["psnr"])
            self.history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"]
            )

            # Validate every few epochs or at key points
            should_validate = (
                    epoch % 5 == 0 or  # Every 5 epochs
                    epoch < 15 or  # First 15 epochs
                    epoch in self.stage_transitions or  # Stage transitions
                    epoch == max_epochs - 1  # Last epoch
            )

            if should_validate:
                val_metrics = self.validate(val_loader)

                # ─── NEW: record validation metrics ────────
                self.history["val_psnr"].append(val_metrics["psnr"])
                self.history["val_ssim"].append(val_metrics["ssim"])

                # Update best scores
                val_psnr = val_metrics['psnr']
                val_ssim = val_metrics['ssim']
                val_sat = val_metrics['saturation']

                # Always track global bests for reporting only
                if val_psnr > self.best_psnr: self.best_psnr = val_psnr
                if val_ssim > self.best_ssim: self.best_ssim = val_ssim

                # Color penalty for early stopping
                color_penalty = max(0, (COLOR_THRESHOLDS['min_saturation'] - val_sat) * COLOR_THRESHOLDS[
                    'saturation_penalty_factor'])
                adjusted_psnr = val_psnr - color_penalty

                # ── NEW: stage-aware BEST saving ─────────────────────────────
                if self.stage < self.save_best_stage:
                    if self.stage == 1 and val_psnr > self.best_psnr_s1:
                        self.best_psnr_s1 = val_psnr
                        self.save_checkpoint('best_s1.pth')
                    monitor_for_es = adjusted_psnr
                else:
                    if self.monitor_stage3 == 'psnr':
                        s3_score = adjusted_psnr
                    elif self.monitor_stage3 == 'ssim':
                        s3_score = (20.0 * val_ssim) - color_penalty
                    else:
                        s3_score = (self.s3_psnr_weight * val_psnr) + \
                                   (self.s3_ssim_weight * (20.0 * val_ssim)) - color_penalty
                        monitor_for_es = s3_score
                        if s3_score > self.best_score_s3:
                            self.best_score_s3 = s3_score
                            self.save_checkpoint('best_s3.pth')
                            self.save_checkpoint('best_model.pth')

                            # Status indicators
                status_parts = []
                gap = train_metrics['psnr'] - val_psnr
                if epoch > 5 and gap > 4.0:  # calmer heuristic
                    status_parts.append("OVERFIT")
                if val_sat < COLOR_THRESHOLDS['min_saturation']:
                    status_parts.append("LOW_COLOR")
                if color_penalty > 0:
                    status_parts.append(f"PENALTY:{color_penalty:.1f}")

                status = " ".join(status_parts) if status_parts else "OK"

                # Show mixing factor if in transition
                mix_str = f"{self.loss_mix:.2f}" if hasattr(self, 'loss_mix') and self.loss_mix < 1.0 else "1.00"

                # Clean log line
                print(
                    f"{epoch:5d} | {self.stage:5d} | {train_metrics['psnr']:10.1f} | {val_psnr:8.1f} | "
                    f"{val_ssim:8.3f} | {val_sat:10.2f} | {mix_str:3s} | {status} "
                    f"({train_metrics['supervised']}/{train_metrics['total']} supervised)"
                )

                # Learning rate scheduling
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_psnr)
                else:
                    self.scheduler.step()

                # Early stopping
                if self.check_early_stopping(monitor_for_es):
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Periodic checkpoints
            if epoch % 10 == 0 and epoch > 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

        # Final summary
        total_time = time.time() - start_time
        print("-" * 80)
        print(f"Training completed in {total_time / 3600:.1f} hours")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Best SSIM: {self.best_ssim:.4f}")


def create_datasets(config):
    """Create train/validation datasets"""
    full_dataset = ImprovedLowLightDataset(
        data_dir=config['train_low'],
        gt_dir=config['train_gt'],
        image_size=config['image_size'],
        mode='train',
        augment=True,
        gt_pairing_strategy=config['gt_pairing']
    )

    if len(full_dataset) == 0:
        raise ValueError("No training data found!")

    # Split dataset
    total_size = len(full_dataset)
    val_size = int(config['val_split'] * total_size)
    train_size = total_size - val_size

    train_indices, val_indices = random_split(
        range(total_size), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_subset = torch.utils.data.Subset(full_dataset, train_indices.indices)

    # Validation dataset (no augmentation)
    val_dataset = ImprovedLowLightDataset(
        data_dir=config['train_low'],
        gt_dir=config['train_gt'],
        image_size=config['image_size'],
        mode='val',
        augment=False,
        gt_pairing_strategy=config['gt_pairing']
    )
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)

    return train_subset, val_subset


def main():
    parser = argparse.ArgumentParser(description='Minimal Low-Light Enhancement Training')

    # Essential arguments
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, default='minimal_training')
    parser.add_argument('--output_dir', type=str, default='./experiments')

    # Model configuration
    parser.add_argument('--num_iterations', type=int, default=8)
    parser.add_argument('--use_semantic_guidance', action='store_true', default=False)
    parser.add_argument('--use_snr_awareness', action='store_true', default=False)
    parser.add_argument('--use_learnable_snr', action='store_true', default=True)
    parser.add_argument('--use_contrast_refinement', action='store_true', default=False)

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--gt_pairing', type=str, default='flexible')
    parser.add_argument('--val_split', type=float, default=0.2)

    # Stage lengths as fractions of total epochs
    parser.add_argument('--stage1_frac', type=float, default=0.25, help='Fraction of total epochs for Stage‑1')
    parser.add_argument('--stage3_frac', type=float, default=0.20, help='Fraction of total epochs for Stage‑3')

    # Stage-3 checkpointing/monitoring
    parser.add_argument('--save_best_stage', type=int, default=3, choices=[1, 2, 3],
                        help='Only this stage may update best_model.pth')
    parser.add_argument('--monitor_stage3', type=str, default='composite',
                        choices=['psnr', 'ssim', 'composite'],
                        help='Metric used to pick best in Stage-3')
    parser.add_argument('--s3_psnr_weight', type=float, default=0.7)
    parser.add_argument('--s3_ssim_weight', type=float, default=0.3)

    # ─── NEW: accumulate N mini-batches before each optimiser step ───
    parser.add_argument('--accum_batches', type=int, default=1,
                        help='Accumulate gradients over N batches (≃ larger virtual batch size)')

    # System
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    # Setup
    setup_seeds()
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {device}")

    # Configuration
    config = {
        'lr': args.learning_rate,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'gt_pairing': args.gt_pairing,
        'val_split': args.val_split,
        'grad_accum': args.accum_batches,
        'output_dir': experiment_dir,
        'save_best_stage': args.save_best_stage,
        'monitor_stage3': args.monitor_stage3,
        's3_psnr_weight': args.s3_psnr_weight,
        's3_ssim_weight': args.s3_ssim_weight,
        'train_low': os.path.join(args.data_root, 'train_data'),
        'train_gt': os.path.join(args.data_root, 'train_gt'),
    }

    # Initialize model
    model = UnifiedLowLightEnhancer(
        num_iterations=args.num_iterations,
        use_snr_awareness=args.use_snr_awareness,
        use_semantic_guidance=args.use_semantic_guidance,
        use_learnable_snr=args.use_learnable_snr,
        use_contrast_refinement=args.use_contrast_refinement
    )

    # Initialize color preservation
    if hasattr(model, 'colour_head') and hasattr(model.colour_head, 'weight'):
        with torch.no_grad():
            color_matrix = torch.tensor([
                [1.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1]
            ]).view(3, 3, 1, 1)
            model.colour_head.weight.data.copy_(color_matrix)

    model = model.to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create datasets
    try:
        train_dataset, val_dataset = create_datasets(config)

        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'],
            shuffle=True, num_workers=0, drop_last=True, pin_memory=False
        )

        val_loader = DataLoader(
            val_dataset, batch_size=config['batch_size'],
            shuffle=False, num_workers=0, pin_memory=False
        )

    except Exception as e:
        print(f"Data loading failed: {e}")
        return 1

    # Train
    try:
        trainer = MinimalTrainer(model, device, config, resume_ckpt=args.resume)
        trainer.train(train_loader, val_loader, args.num_epochs)
        return 0

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())