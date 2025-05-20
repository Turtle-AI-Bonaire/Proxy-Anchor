import os
import json
import torch
import PIL.Image
from .base import BaseDataset
import random

class SeaTurtleIDHeadsDataset(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform

        annotation_path = os.path.join(self.root, 'annotations.json')
        with open(annotation_path) as f:
            dataset = json.load(f)

        annotations = dataset["annotations"]
        images = dataset["images"]

        image_paths_map = {img["id"]: img["path"] for img in images}

        self.im_paths = []
        self.ys = []
        self.I = []
        self.positions = []  # store position for potential use


        index = 0
        allowed_positions = {"left", "right"}

        for ann in annotations:
            pos = ann.get("position", "").lower()
            if pos not in allowed_positions:
                continue

            img_id = ann["image_id"]
            if img_id not in image_paths_map:
                continue

            img_rel_path = image_paths_map[img_id]
            img_path = os.path.join(self.root, img_rel_path)

            identity = ann["identity"] + "_" + pos

            self.im_paths.append(img_path)
            self.ys.append(identity)
            self.positions.append(pos)
            self.I.append(index)
            index += 1

        all_classes = sorted(set(self.ys))
        classes_len = len(all_classes)
        count_train_classes = int(classes_len * 0.8)

        # Freeze random seed for reproducibility if desired
        random.seed(42)

        train_classes = random.sample(all_classes, count_train_classes)
        val_classes = [c for c in all_classes if c not in train_classes]

        if self.mode == 'train':
            selected_classes = set(train_classes)
        elif self.mode == 'eval':
            selected_classes = set(val_classes)
        else:
            # For other modes, take all classes
            selected_classes = set(all_classes)

        # Filter samples based on selected classes
        filtered_indices = [i for i, y in enumerate(self.ys) if y in selected_classes]

        # Keep only selected samples
        self.im_paths = [self.im_paths[i] for i in filtered_indices]
        self.ys = [self.ys[i] for i in filtered_indices]
        self.positions = [self.im_paths[i] for i in filtered_indices]
        self.I = list(range(len(self.ys)))  # fresh indices

        self.classes = sorted(selected_classes)

        super().__init__(self.root, self.mode, self.transform)


    def __getitem__(self, index):
        # Load image as in base
        im = PIL.Image.open(self.im_paths[index])
        if len(list(im.split())) == 1:
            im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        target = self.ys[index]
        # Optionally return position if needed, for now just return image and target
        return im, target

    def get_position(self, index):
        return self.positions[index]

    def nb_classes(self):
        return len(self.classes)
