import os
import numpy as np
from PIL import Image
import torch

class TripleImageDataset:
    """
    dataset.txt 를 읽어서 (cnt_img, char_img, back_img, cnt_path, char_path, back_path) 반환
    """
    def __init__(self, root_dir, dataset_txt="dataset.txt", image_size=512, device="cuda"):
        self.root_dir = root_dir
        self.dataset_txt = os.path.join(root_dir, dataset_txt)
        self.device = device
        self.image_size = image_size
        self.triples = self._load_triples()

    def _load_triples(self):
        triples = []
        with open(self.dataset_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cnt_name, char_name, back_name = line.split()
                cnt_path = os.path.join(self.root_dir, "cnt", cnt_name)
                char_path = os.path.join(self.root_dir, "sty/char", char_name)
                back_path = os.path.join(self.root_dir, "sty/back", back_name)

                if not os.path.isfile(cnt_path):
                    raise FileNotFoundError(f"Content image not found: {cnt_path}")
                if not os.path.isfile(char_path):
                    raise FileNotFoundError(f"Char image not found: {char_path}")
                if not os.path.isfile(back_path):
                    raise FileNotFoundError(f"Back image not found: {back_path}")

                triples.append((cnt_path, char_path, back_path))
            triples.sort(key=lambda x: (x[0], x[1], x[2]))
        return triples

    def _center_crop(self, image):
        """이미지를 정중앙에서 min(x,y) 크기로 crop"""
        x, y = image.size
        crop_size = min(x, y)
        left = (x - crop_size) // 2
        top = (y - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        return image.crop((left, top, right, bottom))

    def _load_img(self, path):
        image = Image.open(path).convert("RGB")
        x, y = image.size
        print(f"Loaded input image of size ({x}, {y}) from {path}")

        # 중앙 crop 후 resize
        image = self._center_crop(image)
        image = image.resize((self.image_size, self.image_size), resample=Image.Resampling.LANCZOS)

        # numpy → torch 변환
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)  # (1, C, H, W)
        image = torch.from_numpy(image)

        return 2. * image - 1.  # [-1, 1] 범위로 정규화

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        cnt_path, char_path, back_path = self.triples[idx]
        cnt_img = self._load_img(cnt_path)
        char_img = self._load_img(char_path)
        back_img = self._load_img(back_path)
        return cnt_img, char_img, back_img, cnt_path, char_path, back_path


class SingleImageDataset:
    """
    A dataset that loads all images in a folder.
    """
    def __init__(self, folder, image_size=512, device="cuda"):
        self.folder = folder
        self.image_size = image_size
        self.device = device
        valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]
        self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                            if os.path.splitext(f)[1].lower() in valid_exts]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in folder: {folder}")

    def _center_crop(self, image):
        x, y = image.size
        crop_size = min(x, y)
        left = (x - crop_size) // 2
        top = (y - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        return image.crop((left, top, right, bottom))

    def _load_img(self, path):
        image = Image.open(path).convert("RGB")
        image = self._center_crop(image)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2. * image - 1.

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = self._load_img(path)
        return img, path
