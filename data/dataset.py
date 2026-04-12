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
    def __init__(self, root_dir, transform=None, split='train', merge_classes=True, **kwargs):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.merge_classes = merge_classes
        
        # Determine the split directory (test uses val as requested)
        split_dir_name = 'val' if self.split == 'test' else self.split
        self.split_dir = self.root_dir / split_dir_name
        
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Directory {self.split_dir} not found.")

        # Discover classes from directory names
        raw_classes = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        
        # Remap classes if merge_classes is used
        if self.merge_classes:
            # We want to map:
            # "1 Centre - Partly", "2 Two Track - Partly", "3 One Track - Partly" -> "1 Partly"
            # "0 Bare" -> "0 Bare"
            # "4 Fully" -> "2 Fully"
            self.classes = ["0 Bare", "1 Partly", "2 Fully"]
            
            raw_to_mapped = {}
            for rc in raw_classes:
                rc_lower = rc.lower()
                if "partly" in rc_lower:
                    raw_to_mapped[rc] = "1 Partly"
                elif "bare" in rc_lower:
                    raw_to_mapped[rc] = "0 Bare"
                elif "fully" in rc_lower:
                    raw_to_mapped[rc] = "2 Fully"
                else:
                    raw_to_mapped[rc] = rc # Fallback
        else:
            self.classes = raw_classes
            raw_to_mapped = {rc: rc for rc in raw_classes}

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        # Collect all images
        self.samples = []
        for rc in raw_classes:
            cls_dir = self.split_dir / rc
            mapped_cls = raw_to_mapped[rc]
            mapped_idx = self.class_to_idx[mapped_cls]
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    self.samples.append((str(img_path), mapped_idx))

        if not self.samples:
            raise ValueError(f"No valid images found in {self.split_dir}")

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

        return image, label, img_path
