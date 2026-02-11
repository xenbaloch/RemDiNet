import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
from pathlib import Path
from typing import Optional, Tuple, Dict
import random


class ImprovedLowLightDataset(Dataset):
    """Clean dataset loader with synchronized augmentations"""

    def __init__(self, data_dir: str, gt_dir: Optional[str] = None,
                 image_size: int = 320, mode: str = 'train', augment: bool = False,
                 gt_pairing_strategy: str = 'strict'):
        """
        Args:
            data_dir: Path to low-light images
            gt_dir: Path to ground truth images (optional)
            image_size: Target image size
            mode: 'train' or 'val'
            augment: Apply data augmentation
            gt_pairing_strategy: 'strict', 'flexible', or 'none'
        """
        self.data_dir = Path(data_dir)
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.image_size = image_size
        self.gt_pairing_strategy = gt_pairing_strategy
        self.mode = mode
        self.augment = augment

        # Get image files
        self.image_files = self._get_image_files()

        # Build GT mapping
        self.gt_mapping = self._build_gt_mapping()

        # Base transforms (applied manually for sync)
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # Input normalization (ImageNet)
        self.input_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _get_image_files(self):
        """Get all valid image files"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        files = [p for p in self.data_dir.rglob('*')
                 if p.suffix.lower() in extensions and p.is_file()]
        return sorted(str(p) for p in files)

    def _build_gt_mapping(self) -> Dict[str, Optional[str]]:
        """Build ground truth mapping based on pairing strategy"""
        mapping = {}

        if not self.gt_dir or not self.gt_dir.exists():
            return {img_path: None for img_path in self.image_files}

        if self.gt_pairing_strategy == 'none':
            return {img_path: None for img_path in self.image_files}

        # Get all GT files (RECURSIVE) – some LOL variants nest images in subfolders
        gt_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        gt_files = {p.stem: str(p) for p in self.gt_dir.rglob('*')
                    if p.suffix.lower() in gt_extensions and p.is_file()}

        for img_path in self.image_files:
            img_stem = Path(img_path).stem
            gt_path = None

            if self.gt_pairing_strategy == 'strict':
                # Exact filename match
                if img_stem in gt_files:
                    gt_path = gt_files[img_stem]

            elif self.gt_pairing_strategy == 'flexible':
                # Try multiple matching strategies
                gt_path = self._flexible_gt_match(img_stem, gt_files)

            mapping[img_path] = gt_path

        return mapping

    def _flexible_gt_match(self, img_stem: str, gt_files: Dict[str, str]) -> Optional[str]:
        """Flexible GT matching with suffix removal"""
        # Try exact match first
        if img_stem in gt_files:
            return gt_files[img_stem]

        # Remove common low-light suffixes
        suffixes_to_remove = ['_low', '_lowlight', '_dark', '_input', '_noisy']
        base_name = img_stem

        for suffix in suffixes_to_remove:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                if base_name in gt_files:
                    return gt_files[base_name]

        # Try adding common GT suffixes
        gt_suffixes = ['_high', '_normal', '_gt', '_clean', '_target']
        for suffix in gt_suffixes:
            candidate = base_name + suffix
            if candidate in gt_files:
                return gt_files[candidate]

        # Try numeric matching
        import re
        numeric_match = re.match(r'(\d+)', base_name)
        if numeric_match:
            numeric_base = numeric_match.group(1)
            for gt_stem in gt_files:
                if gt_stem.startswith(numeric_base):
                    return gt_files[gt_stem]

        return None

    def _apply_synchronized_augmentations(self, low_img: Image.Image,
                                          gt_img: Optional[Image.Image]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply synchronized augmentations to both images"""
        if not self.augment or self.mode != 'train':
            # No augmentation - just resize and convert
            low_tensor = self.input_normalize(self.base_transform(low_img))
            gt_tensor = self.base_transform(gt_img) if gt_img else None
            return low_tensor, gt_tensor

        # Synchronized augmentations with reduced intensity

        # 1. Random resized crop with same parameters
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            low_img, scale=(0.95, 1.0), ratio=(0.95, 1.05)
        )
        low_cropped = TF.resized_crop(low_img, i, j, h, w, (self.image_size, self.image_size))
        gt_cropped = TF.resized_crop(gt_img, i, j, h, w, (self.image_size, self.image_size)) if gt_img else None

        # 2. Random horizontal flip with same decision
        if random.random() < 0.5:
            low_cropped = TF.hflip(low_cropped)
            if gt_cropped:
                gt_cropped = TF.hflip(gt_cropped)

        # 3. Reduced random rotation
        angle = random.uniform(-2, 2)
        interp = transforms.InterpolationMode.BILINEAR
        low_cropped = TF.rotate(low_cropped, angle, interpolation=interp, fill=0)
        if gt_cropped:
            gt_cropped = TF.rotate(gt_cropped, angle, interpolation=interp, fill=0)

        # 4. Stronger color/exposure jitter (INPUT ONLY)
        brightness_factor = random.uniform(0.90, 1.10)    
        contrast_factor = random.uniform(0.90, 1.10)     
        saturation_factor = random.uniform(0.90, 1.10)    
        hue_factor = random.uniform(-0.03, 0.03)          

        # Apply jitter to INPUT only; keep GT photometrically clean
        for fn, fac in [
            (TF.adjust_brightness, brightness_factor),
            (TF.adjust_contrast, contrast_factor),
            (TF.adjust_saturation, saturation_factor),
            (TF.adjust_hue, hue_factor),
        ]:
            low_cropped = fn(low_cropped, fac)


        # Convert to tensors
        low_tensor = TF.to_tensor(low_cropped)
        gt_tensor = TF.to_tensor(gt_cropped) if gt_cropped else None

        # Optional gamma/exposure augmentation (INPUT ONLY)
        if self.augment and self.mode == 'train' and random.random() < 0.4:
            gamma = random.uniform(0.85, 1.20)
            low_tensor = low_tensor.pow(gamma)

        # Optional color temperature shift (white-balance variation)
        if self.augment and self.mode == 'train' and random.random() < 0.5:
            rgb_gain = torch.tensor([
                random.uniform(0.9, 1.1),  # R
                random.uniform(0.9, 1.1),  # G
                random.uniform(0.9, 1.1)  # B
            ]).view(3, 1, 1)
            low_tensor = (low_tensor * rgb_gain).clamp(0, 1)

        # Optional Poisson–Gaussian noise (INPUT ONLY) – **zero‑mean** shot noise
        if self.augment and self.mode == 'train' and random.random() < 0.2:
            sigma = random.uniform(0.0, 0.01)
            Q = 255.0
            # shot noise: sample counts then subtract expectation → zero mean
            counts = torch.poisson((low_tensor * Q).clamp(0, Q))
            shot = (counts - (low_tensor * Q)) / Q
            gauss = torch.randn_like(low_tensor) * sigma
            low_tensor = (low_tensor + shot + gauss).clamp(0, 1)


        # 90° rotation with reduced probability
        if self.augment and self.mode == 'train' and random.random() < 0.3:
            rot_k = random.choice([1, 3])  # 90° or -90°
            low_tensor = torch.rot90(low_tensor, rot_k, [1, 2])
            if gt_tensor is not None:
                gt_tensor = torch.rot90(gt_tensor, rot_k, [1, 2])

        # Apply ImageNet normalization to input only
        low_tensor = self.input_normalize(low_tensor)

        return low_tensor, gt_tensor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Load image pair with synchronized augmentations"""
        try:
            # Load low-light image
            img_path = self.image_files[idx]
            low_image = Image.open(img_path).convert('RGB')

            # Load ground truth if available
            gt_path = self.gt_mapping[img_path]
            gt_image = None

            if gt_path and os.path.exists(gt_path):
                try:
                    gt_image = Image.open(gt_path).convert('RGB')
                except Exception:
                    gt_image = None

            # Apply synchronized augmentations
            low_tensor, gt_tensor = self._apply_synchronized_augmentations(low_image, gt_image)

            return low_tensor, gt_tensor

        except Exception:
            # Return dummy data if loading fails
            dummy_input = torch.zeros(3, self.image_size, self.image_size)
            dummy_input = self.input_normalize(dummy_input)
            return dummy_input, None

    def get_pairing_stats(self) -> Dict[str, int]:
        """Get statistics about GT pairing"""
        paired = sum(1 for gt_path in self.gt_mapping.values() if gt_path is not None)
        unpaired = len(self.gt_mapping) - paired

        return {
            'total_images': len(self.gt_mapping),
            'paired_images': paired,
            'unpaired_images': unpaired,
            'pairing_ratio': paired / len(self.gt_mapping) if self.gt_mapping else 0.0
        }


class UnifiedTrainingConfig:
    """Simplified training configuration"""

    def __init__(self, data_root: str = './data'):
        # Core dataset paths
        self.DATASETS = {
            'main': {
                'train_low': os.path.join(data_root, 'train_data'),
                'train_gt': os.path.join(data_root, 'train_gt'),
                'val_low': os.path.join(data_root, 'val_data'),
                'val_gt': os.path.join(data_root, 'val_gt'),
                'test_low': os.path.join(data_root, 'test_data'),
                'test_gt': os.path.join(data_root, 'test_gt'),
            }
        }

        # Training configuration from loss_weights.py
        from loss_weights import TRAINING_DEFAULTS
        self.TRAINING = TRAINING_DEFAULTS.copy()

        # Output directory
        self.OUTPUT_DIR = './experiments'

        # GT pairing strategy
        self.gt_pairing_strategy = 'flexible'


class UnifiedDataLoader:
    """Simplified data loader"""

    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config

    def create_datasets(self, mode: str = 'train') -> Tuple[Optional[Dataset], Dict[str, int]]:
        """Create dataset for specified mode"""

        if mode == 'train':
            data_dir = self.config.DATASETS['main']['train_low']
            gt_dir = self.config.DATASETS['main']['train_gt']
            augment = True
        elif mode == 'val':
            data_dir = self.config.DATASETS['main']['val_low']
            gt_dir = self.config.DATASETS['main']['val_gt']
            augment = False
        elif mode == 'test':
            data_dir = self.config.DATASETS['main']['test_low']
            gt_dir = self.config.DATASETS['main']['test_gt']
            augment = False
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not os.path.exists(data_dir):
            return None, {}

        # Create dataset
        dataset = ImprovedLowLightDataset(
            data_dir=data_dir,
            gt_dir=gt_dir if os.path.exists(gt_dir or '') else None,
            image_size=320,
            mode=mode,
            augment=augment,
            gt_pairing_strategy=self.config.gt_pairing_strategy
        )

        if len(dataset) == 0:
            return None, {}

        # Get pairing statistics
        stats = dataset.get_pairing_stats()
        return dataset, stats

    def create_dataloaders(self) -> Tuple[Optional[torch.utils.data.DataLoader],
    Optional[torch.utils.data.DataLoader], Dict]:
        """Create train and validation dataloaders"""
        from torch.utils.data import DataLoader

        # Create datasets
        train_dataset, train_stats = self.create_datasets('train')
        val_dataset, val_stats = self.create_datasets('val')

        train_loader = None
        val_loader = None

        # Create train loader
        if train_dataset:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.TRAINING['batch_size'],
                shuffle=True,
                num_workers=2,
                drop_last=True,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=lambda wid: (
                    random.seed(torch.initial_seed() % 2 ** 32)
                )
            )

        # Create validation loader
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.TRAINING['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

        stats = {
            'train_stats': train_stats,
            'val_stats': val_stats
        }

        return train_loader, val_loader, stats


# Keep backward compatibility

TrainingConfig = UnifiedTrainingConfig
