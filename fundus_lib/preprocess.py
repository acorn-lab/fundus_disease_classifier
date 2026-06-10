# preprocess.py
# paper-style fundus preprocessing (linchundan / cen et al. 2021):
# crop the circular retinal region out of the black border, pad to a square,
# resize to a fixed size. run this once to build a cached crop directory that
# every model notebook reads from, so all models see pixel-identical inputs.

import os
import glob
import cv2
import numpy as np

# image extensions we accept (case-insensitive). the kaggle set mixes .JPG/.jpg/.tif
image_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def list_images(folder):
    # case-insensitive recursive listing of image files under a folder
    out = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(image_exts):
                out.append(os.path.join(root, f))
    return out


def crop_from_gray(img, tol=7):
    # crop the black border by keeping rows/cols whose grayscale exceeds tol.
    # img is an rgb uint8 array. returns the cropped rgb array.
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mask = gray > tol
    if mask.sum() == 0:
        # all black / unreadable -> return as-is
        return img

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    cropped = img[r0:r1 + 1, c0:c1 + 1]

    # guard against degenerate crops (a single bright pixel etc.)
    if cropped.shape[0] < 10 or cropped.shape[1] < 10:
        return img
    return cropped


def pad_to_square(img, fill=0):
    # pad the shorter side so the fundus circle is centered in a square frame
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=[fill, fill, fill])


def circle_mask(img):
    # zero out the corners outside the inscribed circle (mirrors the paper's
    # clean circular fundus appearance). img is a square rgb array.
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 1, thickness=-1)
    return img * mask[..., None]


def preprocess_fundus(path_in, out_size=512, tol=7, apply_circle_mask=True):
    # full single-image pipeline: read -> crop border -> square pad ->
    # optional circle mask -> resize. returns an rgb uint8 array, or None on failure.
    img = cv2.imread(path_in, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_from_gray(img, tol=tol)
    img = pad_to_square(img, fill=0)
    if apply_circle_mask:
        img = circle_mask(img)
    img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return img


def build_cache(manifest, out_dir, out_size=512, tol=7, apply_circle_mask=True,
                verbose=True):
    # manifest: list of dicts with keys "path" (source image) and "bigclass" (int).
    # writes one png per image to out_dir/<bigclass>/<index>.png and returns a
    # new manifest with the cached png paths added under key "cache_path".
    os.makedirs(out_dir, exist_ok=True)
    out_manifest = []
    n_fail = 0
    for i, row in enumerate(manifest):
        cls_dir = os.path.join(out_dir, str(row["bigclass"]))
        os.makedirs(cls_dir, exist_ok=True)
        cache_path = os.path.join(cls_dir, f"{i:05d}.png")
        proc = preprocess_fundus(row["path"], out_size=out_size, tol=tol,
                                 apply_circle_mask=apply_circle_mask)
        if proc is None:
            n_fail += 1
            continue
        # cv2 writes bgr, so convert back
        cv2.imwrite(cache_path, cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
        new_row = dict(row)
        new_row["cache_path"] = cache_path
        out_manifest.append(new_row)
        if verbose and (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(manifest)} images")
    if verbose:
        print(f"done: {len(out_manifest)} cached, {n_fail} failed")
    return out_manifest
