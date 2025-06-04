import os
import csv
import random
import torch
import PIL.Image
from pandas import pandas
# Retain import for backwards-compatibility with any existing subclassing.


class BonaireTurtlesDataset(torch.utils.data.Dataset):


    #: Sides accepted by the loader and their canonical labels
    _ALLOWED_SIDES = {"L", "R"}

    def __init__(self, root: str, mode: str, transform=None, ignoreThreshold = 1):
        # ------------------------------------------------------------------
        # File system bookkeeping
        # ------------------------------------------------------------------
        self.root = root + '/sbh2'
        self.mode = mode.lower()
        self.transform = transform

        # Prefer *.csv but keep compatibility with the *.csf typo/variant
        meta_path_csv = os.path.join(self.root, "data.csv")
        if os.path.isfile(meta_path_csv):
            meta_path = meta_path_csv
        else:
            raise FileNotFoundError(
                "No metadata file found (expected 'metadata.csv' or 'metadata.csf')."
            )


        # ------------------------------------------------------------------
        # Parse CSV and build sample list
        # ------------------------------------------------------------------
        self.im_paths: list[str] = []
        self._y_strs: list[str] = []  # keep the string labels until mapping
        self.positions: list[str] = []
        self.I: list[int] = []

        with open(meta_path, newline="", encoding="utf-8") as csvfile:
            data_df = pandas.DataFrame(list(csv.DictReader(csvfile)))

        grouped = data_df.groupby(['internal_turtle_id', 'side'])
        group_counts = grouped.size()
        if ignoreThreshold > 0: 
            multi_sample_groups = group_counts[group_counts > ignoreThreshold]
            print("Removing single-sample classes for BonaireTurtlesDataset.")
        else: multi_sample_groups = group_counts

        index = 0
        for (turtle_id, side), group_df in grouped:
            # Only process groups that have more than 2 samples
            if (turtle_id, side) not in multi_sample_groups.index:
                continue

            turtle_id = turtle_id.strip()
            side = side.upper().strip()

            # if side not in self._ALLOWED_SIDES or not turtle_id:
            #     continue            
            if not turtle_id:
                continue

            for _, row in group_df.iterrows():
                filename = row.get("filename", "").strip()
                if not filename:
                    continue

                img_path = os.path.join(self.root, "images", filename)
                if not os.path.isfile(img_path):
                    continue

                identity = f"{turtle_id}"  # preserve old label pattern
                # identity = f"{turtle_id}_{side}"  # preserve old label pattern

                self.im_paths.append(img_path)
                self._y_strs.append(identity)
                self.positions.append(side)
                self.I.append(index)
                index += 1


        if not self.im_paths:
            raise RuntimeError("Dataset is empty — check folder structure and metadata file.")

        # ------------------------------------------------------------------
        # Train/validation split at the *class* (identity) level
        # ------------------------------------------------------------------
        all_classes = sorted(set(self._y_strs))
        class_count = len(all_classes)
        train_class_count = int(class_count * 0.8) if class_count > 1 else class_count

        # Deterministic split for reproducibility (seed can be exposed if needed)
        rng = random.Random(42)
        train_classes = rng.sample(all_classes, train_class_count)
        val_classes = [cls for cls in all_classes if cls not in train_classes]

        if self.mode == "train":
            selected = set(train_classes)
        elif self.mode == "eval":
            selected = set(val_classes)
        else:  # e.g. "all" or any other custom mode collects everything
            selected = set(all_classes)

        # ------------------------------------------------------------------
        # Filter samples to the split we decided on
        # ------------------------------------------------------------------
        filtered = [i for i, lbl in enumerate(self._y_strs) if lbl in selected]
        self.im_paths = [self.im_paths[i] for i in filtered]
        self._y_strs = [self._y_strs[i] for i in filtered]
        # print(self._y_strs)
        self.positions = [self.positions[i] for i in filtered]
        self.I = list(range(len(self._y_strs)))  # re-index consecutive

        # ------------------------------------------------------------------
        # Convert string class labels → integer indices
        # ------------------------------------------------------------------
        self.classes = sorted(selected)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.ys = [self.class_to_idx[s] for s in self._y_strs]

    # ---------------------------------------------------------------------
    # Dataset protocol implementation
    # ---------------------------------------------------------------------
    def __getitem__(self, index: int):
        """Return ``(image_tensor, target_int)`` for the sample at *index*."""
        img = PIL.Image.open(self.im_paths[index])
        # Some JPEGs are single-channel — convert to RGB for consistency
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        target = self.ys[index]
        return img, target

    def __len__(self):
        return len(self.ys)

    # ------------------------------------------------------------------
    # Convenience helpers matching the legacy interface
    # ------------------------------------------------------------------
    def nb_classes(self) -> int:
        """Return the number of (photo-side) identities in the dataset."""
        return len(self.classes)

    def get_position(self, index: int) -> str:
        """Return the raw *side* label (``"L"`` or ``"R"``) for *index*."""
        return self.positions[index]
