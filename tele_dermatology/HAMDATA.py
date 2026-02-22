import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class HAMDataset(Dataset):
    def __init__(self, img_dir, metadata_path, mask_dir=None, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.metadata = pd.read_csv(metadata_path)

        # Map labels (binary example: melanoma vs others)
        self.metadata["label"] = self.metadata["dx"].apply(
            lambda x: 1 if x == "mel" else 0
        )

        self.samples = self.metadata[["image_id", "label"]].reset_index(drop=True)

        print("Final usable images:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id = self.samples.iloc[idx]["image_id"]
        label = self.samples.iloc[idx]["label"]

        img_name = img_id + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.mask_dir:
            # IMPORTANT: HAM masks are named like:
            # ISIC_xxxxxx_segmentation.png
            mask_name = img_id + "_segmentation.png"
            mask_path = os.path.join(self.mask_dir, mask_name)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            mask = Image.open(mask_path).convert("L")

            if self.transform:
                mask = self.transform(mask)

            return image, mask

        return image, torch.tensor(label, dtype=torch.long)