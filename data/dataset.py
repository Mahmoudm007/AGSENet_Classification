import os
from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter
from sklearn.model_selection import train_test_split

class RoadPondingDataset(Dataset):
    """
    Dataset loader for road ponding image classification.
    Expects a folder structure where subdirectories are class names.
    Automatically handles splitting if specified.
    """
    def __init__(self, root_dir, transform=None, split='train', val_ratio=0.2, test_ratio=0.1, seed=42):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory {self.root_dir} not found.")

        # Discover classes
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        # Collect all images
        all_samples = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    all_samples.append((str(img_path), self.class_to_idx[cls_name]))

        if not all_samples:
            raise ValueError(f"No valid images found in {self.root_dir}")

        # Stratified Splitting
        labels = [s[1] for s in all_samples]
        
        # Train / Temp split
        temp_ratio = val_ratio + test_ratio
        if temp_ratio > 0:
            train_samples, temp_samples, train_labels, temp_labels = train_test_split(
                all_samples, labels, test_size=temp_ratio, stratify=labels, random_state=seed
            )
            
            if test_ratio > 0 and val_ratio > 0:
                rel_test_ratio = test_ratio / temp_ratio
                val_samples, test_samples = train_test_split(
                    temp_samples, test_size=rel_test_ratio, stratify=temp_labels, random_state=seed
                )
            elif val_ratio > 0:
                val_samples = temp_samples
                test_samples = []
            else:
                test_samples = temp_samples
                val_samples = []
        else:
            train_samples = all_samples
            val_samples = []
            test_samples = []

        # Assign split
        if self.split == 'train':
            self.samples = train_samples
        elif self.split == 'val':
            self.samples = val_samples
        elif self.split == 'test':
            self.samples = test_samples
        else:
            raise ValueError(f"Invalid split {split}. Must be train, val, or test.")

        print(f"[{split.upper()}] Loaded {len(self.samples)} images from {len(self.classes)} classes.")
        self.print_class_distribution()

    def print_class_distribution(self):
        counter = Counter([s[1] for s in self.samples])
        print("Class distribution:")
        for idx in range(len(self.classes)):
            count = counter[idx]
            print(f"  - {self.idx_to_class[idx]}: {count}")

    def get_class_frequencies(self):
        """Returns the number of samples for each class in this split."""
        counter = Counter([s[1] for s in self.samples])
        counts = [counter[i] for i in range(len(self.classes))]
        return counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Handle corrupt images by falling back to another random image in dataset
            print(f"Warning: Failed to load {img_path}. Error: {e}")
            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx)

        if self.transform:
            image = self.transform(image)

        return image, label
