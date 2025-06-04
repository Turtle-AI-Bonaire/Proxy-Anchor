import os
import re
import torch
import PIL.Image
from pandas import pandas
import random

class AmvrakikosDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, mode: str, transform=None):
        self.root = root + '/amv_c_224'
        self.mode = mode.lower()
        self.transform = transform

        meta_path_csv = os.path.join(self.root, "annotations.csv")
        if not os.path.isfile(meta_path_csv):
            raise FileNotFoundError("No metadata file found (expected 'annotations.csv').")

        self.im_paths = []
        self._y_strs = []
        self.positions = []  # empty because not parsed, adjust if needed
        self.I = []

        def extract_turtle_id(filename):
            base = os.path.basename(filename)
            match = re.match(r"^(\d+)_", base)
            return match.group(1) if match else None

        data_df = pandas.read_csv(meta_path_csv)

        index = 0
        for _, row in data_df.iterrows():
            filename = row.get("image_name", "").strip()
            if not filename:
                continue

            turtle_id = extract_turtle_id(filename)
            if not turtle_id:
                continue

            img_path = os.path.join(self.root, "images", filename)
            if not os.path.isfile(img_path):
                continue

            identity = f"{turtle_id}"

            self.im_paths.append(img_path)
            self._y_strs.append(identity)
            self.positions.append("")  # placeholder, no side info currently
            self.I.append(index)
            index += 1

        if not self.im_paths:
            raise RuntimeError("Dataset is empty — check folder structure and metadata file.")

        # Train/validation split at the class (identity) level
        all_classes = sorted(set(self._y_strs))
        class_count = len(all_classes)
        train_class_count = int(class_count * 0.8) if class_count > 1 else class_count

        rng = random.Random(42)
        train_classes = rng.sample(all_classes, train_class_count)
        val_classes = [cls for cls in all_classes if cls not in train_classes]

        if self.mode == "train":
            selected = set(train_classes)
        elif self.mode == "eval":
            selected = set(val_classes)
        else:
            selected = set(all_classes)

        filtered = [i for i, lbl in enumerate(self._y_strs) if lbl in selected]
        self.im_paths = [self.im_paths[i] for i in filtered]
        self._y_strs = [self._y_strs[i] for i in filtered]
        self.positions = [self.positions[i] for i in filtered]
        self.I = list(range(len(self._y_strs)))

        self.classes = sorted(selected)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.ys = [self.class_to_idx[s] for s in self._y_strs]

    def __getitem__(self, index: int):
        img = PIL.Image.open(self.im_paths[index])
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        target = self.ys[index]
        return img, target

    def __len__(self):
        return len(self.ys)

    def nb_classes(self) -> int:
        return len(self.classes)

    def get_position(self, index: int) -> str:
        return self.positions[index]
