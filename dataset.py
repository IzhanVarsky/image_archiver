import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root_dirs, transform=None, return_img_paths=False):
        self.image_files = []
        for root_dir in root_dirs:
            self.get_all_img_paths(root_dir)
        self.transform = transform
        self.return_img_paths = return_img_paths

    def get_all_img_paths(self, root_dir):
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.image_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.return_img_paths:
            return image, img_path
        return image
