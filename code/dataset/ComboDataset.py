import os
import csv
import random
import torch
import PIL.Image
from dataset.BonaireTurtlesDataset import BonaireTurtlesDataset
from dataset.SeaTurtleIDHeadsDataset import SeaTurtleIDHeadsDataset

class CombinedTurtlesDataset(torch.utils.data.Dataset):
    """
    Combines SeaTurtleIDHeadsDataset and BonaireTurtlesDataset.
    """
    def __init__(self, root: str, mode: str, transform=None, seed: int = 42):
        """
        Args:
            root_heads (str): Root directory for SeaTurtleIDHeadsDataset (e.g., path to 'scaled_heads_dataset').
            root_bonaire (str): Root directory for BonaireTurtlesDataset (e.g., path to 'sbh2').
            mode (str): One of 'train', 'eval', or 'all'.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed (int): Random seed for train/val split reproducibility.
        """
        self.mode = mode.lower()
        self.transform = transform
        self.seed = seed

        # Load all data from both datasets internally by passing a special flag or mode
        # The individual datasets are modified to support 'load_all_for_combination'
        # which skips their internal train/val split and loads all string labels.
        
        # Path adjustments:
        # SeaTurtleIDHeadsDataset expects 'root' to be the path to 'scaled_heads_dataset'
        # BonaireTurtlesDataset expects 'root' to be the path to 'sbh2'

        dataset_heads = SeaTurtleIDHeadsDataset(root, mode='all', transform=None)
        dataset_bonaire = BonaireTurtlesDataset(root, mode='all', transform=None)

        # --- Combine data from both datasets ---
        self.im_paths_combined: list[str] = []
        self.ys_str_combined: list[str] = [] # String labels, prefixed for uniqueness
        self.positions_combined: list[str] = []

        # Add data from SeaTurtleIDHeadsDataset with prefix
        if hasattr(dataset_heads, 'im_paths') and dataset_heads.im_paths: # Check if dataset loaded data
            self.im_paths_combined.extend(dataset_heads.im_paths)
            self.ys_str_combined.extend([f"heads_{s}" for s in dataset_heads.classes])
            self.positions_combined.extend(dataset_heads.positions)
        else:
            print(f"Warning: SeaTurtleIDHeadsDataset loaded no data.")


        # Add data from BonaireTurtlesDataset with prefix
        if hasattr(dataset_bonaire, 'im_paths') and dataset_bonaire.im_paths: # Check if dataset loaded data
            self.im_paths_combined.extend(dataset_bonaire.im_paths)
            self.ys_str_combined.extend([f"bonaire_{s}" for s in dataset_bonaire._y_strs])
            self.positions_combined.extend(dataset_bonaire.positions)
        else:
            print(f"Warning: BonaireTurtlesDataset loaded no data.")


        if not self.im_paths_combined:
            # print("Combined dataset is empty after attempting to load from both sources.")
            self.im_paths = []
            self.ys = []
            self.I = []
            self.positions = []
            self.classes = []
            self.class_to_idx = {}
            return
            # raise RuntimeError("Combined dataset is empty. Check source dataset paths and content.")


        # --- Train/validation split at the *combined class* level ---
        all_classes_combined = sorted(list(set(self.ys_str_combined)))
        
        if not all_classes_combined: # Handles case where sources loaded data but all were filtered/empty somehow
            print("Warning: Combined dataset has image paths but no unique classes derived. This is unusual.")
            self.im_paths = []
            self.ys = []
            self.I = []
            self.positions = []
            self.classes = []
            self.class_to_idx = {}
            return

        class_count_combined = len(all_classes_combined)
        train_class_count_combined = int(class_count_combined * 0.8) if class_count_combined > 1 else class_count_combined
        
        rng = random.Random(self.seed)
        train_classes_combined = rng.sample(all_classes_combined, train_class_count_combined)
        val_classes_combined = [cls for cls in all_classes_combined if cls not in train_classes_combined]

        if self.mode == "train":
            selected_classes_combined = set(train_classes_combined)
        elif self.mode == "eval":
            selected_classes_combined = set(val_classes_combined)
        elif self.mode == "all":
            selected_classes_combined = set(all_classes_combined)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'train', 'eval', 'all'.")

        # --- Filter samples to the split we decided on ---
        self.im_paths: list[str] = []
        self._y_strs_final: list[str] = [] # Store final string labels for this split
        self.positions: list[str] = []
        
        for i, original_y_str in enumerate(self.ys_str_combined):
            if original_y_str in selected_classes_combined:
                self.im_paths.append(self.im_paths_combined[i])
                self._y_strs_final.append(original_y_str)
                self.positions.append(self.positions_combined[i])
        
        self.I = list(range(len(self._y_strs_final))) # Re-index consecutively

        if not self.im_paths:
            # print(f"Warning: Combined dataset for mode '{self.mode}' is empty after class filtering.")
            # Ensure attributes exist even if empty
            self.ys = []
            self.classes = []
            self.class_to_idx = {}
            return


        # --- Convert string class labels to integer indices ---
        self.classes = sorted(list(selected_classes_combined)) # These are the actual classes in the current split
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.ys = [self.class_to_idx[s] for s in self._y_strs_final]


    def __getitem__(self, index: int):
        """Return (image_tensor, target_int) for the sample at *index*."""
        if not self.im_paths:
            raise IndexError("Dataset is empty.")
        img_path = self.im_paths[index]
        try:
            img = PIL.Image.open(img_path)
        except FileNotFoundError:
            print(f"ERROR: Image file not found at {img_path} during __getitem__ for index {index}.")
            # Handle this case, e.g., return a placeholder or raise an error
            # For now, re-raising to make it explicit
            raise FileNotFoundError(f"Image file not found: {img_path}")
        except Exception as e:
            print(f"ERROR: Could not open image {img_path} during __getitem__ for index {index}: {e}")
            raise e


        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        
        target = self.ys[index]
        return img, target

    def __len__(self):
        return len(self.im_paths)

    def nb_classes(self) -> int:
        """Return the number of unique identities in the current dataset mode."""
        return len(self.classes)

    def get_position(self, index: int) -> str:
        """Return the raw side label ('left', 'right', 'L', or 'R') for the sample at *index*."""
        if not self.positions:
            raise IndexError("Dataset positions are empty.")
        return self.positions[index]

    def get_original_identity(self, index: int) -> str:
        """Return the original prefixed string identity for the sample at *index*."""
        if not hasattr(self, '_y_strs_final') or not self._y_strs_final:
            raise RuntimeError("Original string identities are not available. Dataset might be empty or not properly initialized.")
        return self._y_strs_final[index]