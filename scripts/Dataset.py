import glob
import os
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class TrainingDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform_img=None, transform_mask=None):
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.transform_img is not None:
            img = self.transform_img(img)
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)

        mask = (mask > 0.5).float()

        if random.random() > 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)

        if random.random() > 0.5:
            img = F.vflip(img)
            mask = F.vflip(mask)

        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-10, 10)

            img = F.affine(img, angle=angle, translate=translate, scale=scale, shear=shear)
            mask = F.affine(mask, angle=angle, translate=translate, scale=scale, shear=shear)

        return img, mask


class TestDataset(Dataset):
    def __init__(self, images_dir, transform_img=None):
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.transform_img = transform_img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform_img is not None:
            img = self.transform_img(img)

        return img
