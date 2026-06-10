# data.py
# bigclass taxonomy (cen et al. 2021 "bigclasses"), manifest building over the
# 1000images dataset, the FundusDataset over cached crops, and frozen stratified
# 5-fold splits. restricted to bigclasses 0-9 (the populous ones we compare to
# fig 2a/2b of the paper).

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

# human-readable names for bigclasses 0-9 (integer prefix of the folder name).
# the dataset folders encode a 2-level taxonomy: parent "N" -> leaves "N.x".
bigclass_names = {
    0: "Normal / Tessellated / Large optic cup / DR1",
    1: "DR2-DR3 (moderate-severe NPDR)",
    2: "BRVO / CRVO",
    3: "RAO",
    4: "Rhegmatogenous RD",
    5: "CSCR / VKH",
    6: "Maculopathy",
    7: "ERM",
    8: "MH",
    9: "Pathological myopia",
}

# the nested duplicate folder inside the dataset that must be excluded to avoid
# double-counting and train/test leakage
duplicate_dirname = "1000images"

image_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def folder_to_bigclass(folder_name):
    # "1.0.DR2" -> 1 ; "29.0.Blur..." -> 29 ; "3.RAO" -> 3
    head = folder_name.split(".")[0].strip()
    try:
        return int(head)
    except ValueError:
        return None


def build_manifest(dataset_root, keep_bigclasses=range(10)):
    # walk the top-level class folders only (skip the nested duplicate copy),
    # case-insensitive, keep images whose bigclass is in keep_bigclasses.
    keep = set(keep_bigclasses)
    manifest = []
    for entry in sorted(os.listdir(dataset_root)):
        full = os.path.join(dataset_root, entry)
        if not os.path.isdir(full):
            continue
        if entry == duplicate_dirname:
            continue  # exclude the duplicate
        bc = folder_to_bigclass(entry)
        if bc is None or bc not in keep:
            continue
        for root, _dirs, files in sorted(os.walk(full)):
            for f in sorted(files):  # sorted for deterministic, reproducible order
                if f.lower().endswith(image_exts):
                    manifest.append({
                        "path": os.path.join(root, f),
                        "bigclass": bc,
                        "source_folder": entry,
                    })
    return manifest


def make_splits(manifest, n_splits=5, seed=42, out_path=None):
    # frozen stratified k-fold. assigns each sample a test-fold index in [0, n_splits).
    # writes {index: fold} plus metadata to out_path (json). returns the fold array.
    labels = np.array([row["bigclass"] for row in manifest])
    folds = np.full(len(manifest), -1, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (_train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        folds[test_idx] = fold_idx
    if out_path is not None:
        payload = {
            "n_splits": n_splits,
            "seed": seed,
            "fold_of_index": folds.tolist(),
            "bigclass_of_index": labels.tolist(),
        }
        with open(out_path, "w") as fh:
            json.dump(payload, fh)
    return folds


def load_splits(splits_path):
    with open(splits_path) as fh:
        return json.load(fh)


def fold_indices(folds, test_fold, val_fold=None):
    # for a given test fold, build train/val/test index lists.
    # val defaults to the next fold (cyclic); train is everything else.
    folds = np.asarray(folds)
    n_splits = int(folds.max()) + 1
    if val_fold is None:
        val_fold = (test_fold + 1) % n_splits
    test_idx = np.where(folds == test_fold)[0]
    val_idx = np.where(folds == val_fold)[0]
    train_idx = np.where((folds != test_fold) & (folds != val_fold))[0]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


class FundusDataset(Dataset):
    # reads cached crop pngs (key "cache_path") and applies a torchvision transform.
    # labels are remapped to contiguous 0..K-1 via class_to_index for the model head.
    def __init__(self, manifest, indices, transform, class_to_index):
        self.rows = [manifest[i] for i in indices]
        self.transform = transform
        self.class_to_index = class_to_index

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = row.get("cache_path", row["path"])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.class_to_index[row["bigclass"]]
        return img, label


def class_index_maps(keep_bigclasses=range(10)):
    # contiguous index <-> bigclass mappings for the classification head
    classes = sorted(set(keep_bigclasses))
    class_to_index = {c: i for i, c in enumerate(classes)}
    index_to_class = {i: c for c, i in class_to_index.items()}
    return class_to_index, index_to_class
