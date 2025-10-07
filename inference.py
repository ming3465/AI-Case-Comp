#!/usr/bin/env python3
"""
inference.py

Reproducible single-command script:
  python inference.py --images <eval_images_path> --meta meta.csv --out preds.csv

Behavior:
 - Loads labelled metadata CSV (`--meta`) which must contain columns `image_id` and `hgb`.
 - Finds image files under `--dataset-dir` (defaults to `dataset/images`) matching `image_id`.
 - Uses a small training set (default 19) taken from the metadata. By default this WILL allow overlap
   between training images and evaluation images (i.e. images in `--images`) so the script will not
   fail if everything lives in the same folder — this was requested for reproducibility.
 - Trains a small model (ResNet18 backbone + optional handcrafted skin-features) deterministically
   (seeded + cuDNN deterministic flags) and saves predictions for images listed/provided to `--images`.
 - Writes CSV to `--out` containing predictions and confidence metrics.

Notes on reproducibility:
 - Seeds are set for numpy, torch and Python's hash seed. cuDNN deterministic mode is enabled.
 - For bit-for-bit exact reproducibility you must run on the same hardware and avoid nondeterministic ops.

"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ----------------------------
# Defaults / paths
# ----------------------------
CROPS_DIR = "dataset/lips_cropped"
RESULTS_DIR = "results"
DATASET_DIR = "dataset/images"
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------
# Utilities
# ----------------------------

def remove_black_scribbles(bgr_img, dark_thresh=40):
    if bgr_img is None:
        return None
    mask = np.all(bgr_img <= dark_thresh, axis=2).astype(np.uint8) * 255
    if mask.sum() == 0:
        return bgr_img
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    try:
        return cv2.inpaint(bgr_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    except Exception:
        return bgr_img


def gray_world(bgr_img):
    if bgr_img is None:
        return None
    b, g, r = cv2.split(bgr_img.astype(np.float32))
    kb, kg, kr = b.mean(), g.mean(), r.mean()
    kb = kb if kb != 0 else 1.0
    kg = kg if kg != 0 else 1.0
    kr = kr if kr != 0 else 1.0
    kb, kg, kr = (kb + kg + kr)/3/kb, (kb + kg + kr)/3/kg, (kb + kg + kr)/3/kr
    out = cv2.merge([b*kb, g*kg, r*kr])
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_lips_optimized_bgr(img_bgr):
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    clean = remove_black_scribbles(img_bgr, dark_thresh=30)

    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32)
    y_start = int(h * 0.12); y_end = int(h * 0.62)
    y_start = max(0, y_start); y_end = min(h, y_end)
    row_means = gray[y_start:y_end].mean(axis=1) if (y_end - y_start) >= 3 else np.array([])

    mouth_row = None
    grad_conf = 0.0
    if row_means.size >= 3:
        d = np.diff(row_means)
        idx = int(np.argmin(d))
        mouth_row = y_start + idx
        grad_conf = float(abs(d[idx])) if idx < d.size else 0.0

    crop = None
    crop_conf_score = 0.0

    if mouth_row is not None and grad_conf > 1.5:
        crop_h = max(24, int(h * 0.14))
        crop_w = max(48, int(w * 0.5))
        y1 = max(0, mouth_row - crop_h//2); y2 = min(h, mouth_row + crop_h//2)
        x1 = max(0, int(w * 0.16)); x2 = min(w, int(w * 0.84))
        candidate = clean[y1:y2, x1:x2].copy()
        area = candidate.shape[0] * candidate.shape[1]
        contrast = float(np.std(candidate)) if candidate.size else 0.0
        crop = candidate if candidate.size else None
        crop_conf_score = (grad_conf * 0.6) + (contrast * 0.02) + (area / (w*h) * 10.0)
        if crop is not None and contrast > 6.0 and area > 400:
            return crop

    y0, y1 = int(h * 0.25), int(h * 0.65)
    x0, x1 = int(w * 0.20), int(w * 0.80)
    y0 = max(0, y0); y1 = min(h, y1); x0 = max(0, x0); x1 = min(w, x1)
    region = clean[y0:y1, x0:x1].copy()
    if region.size == 0:
        fy1, fy2 = int(h * 0.40), int(h * 0.90)
        fx1, fx2 = int(w * 0.12), int(w * 0.88)
        return clean[fy1:fy2, fx1:fx2].copy()

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 30, 30]); upper1 = np.array([18, 255, 255])
    lower2 = np.array([160,30,30]); upper2 = np.array([180,255,255])
    mask_h1 = cv2.inRange(hsv, lower1, upper1)
    mask_h2 = cv2.inRange(hsv, lower2, upper2)
    mask_h = cv2.bitwise_or(mask_h1, mask_h2)

    ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    Cr = ycrcb[:, :, 1]
    Cr_eq = cv2.equalizeHist(Cr)
    _, mask_cr = cv2.threshold(Cr_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.bitwise_or(mask_h, mask_cr)
    mask = cv2.medianBlur(mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 50]
    candidate = None
    if contours:
        target_y = region.shape[0] * 0.32
        best_score = -1e9; best_box = None
        for c in contours:
            x, y, wc, hc = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            centroid_y = y + hc/2.0
            score = area - 0.9 * abs(centroid_y - target_y)
            if score > best_score:
                best_score = score
                best_box = (x, y, wc, hc)
        if best_box is not None:
            x, y, wc, hc = best_box
            pad_x = max(2, int(0.14 * wc)); pad_y = max(2, int(0.20 * hc))
            x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
            x2 = min(region.shape[1], x + wc + pad_x); y2 = min(region.shape[0], y + hc + pad_y)
            candidate = region[y1:y2, x1:x2].copy()

    if candidate is not None:
        area = candidate.shape[0] * candidate.shape[1]
        contrast = float(np.std(candidate))
        color_conf = (contrast * 0.02) + (area / (w*h) * 8.0)
        if crop is None or color_conf > crop_conf_score:
            crop = candidate
            crop_conf_score = color_conf

    if crop is None:
        fy1, fy2 = int(h * 0.40), int(h * 0.90)
        fx1, fx2 = int(w * 0.12), int(w * 0.88)
        return clean[fy1:fy2, fx1:fx2].copy()

    area = crop.shape[0] * crop.shape[1]
    contrast = float(np.std(crop))
    min_area = max(400, int(0.003 * h * w))
    min_contrast = 5.0
    min_conf_score = 1.5

    if (area < min_area) or (contrast < min_contrast) or (crop_conf_score < min_conf_score):
        fy1, fy2 = int(h * 0.35), int(h * 0.9)
        fx1, fx2 = int(w * 0.10), int(w * 0.9)
        big_crop = clean[fy1:fy2, fx1:fx2].copy()
        if big_crop.size == 0:
            return clean
        return big_crop
    return crop


def skin_mask_hsv_local(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    mask = ((H < 25) | (H > 160)) & (S > 30) & (V > 30) & (V < 250)
    mask = cv2.medianBlur(mask.astype(np.uint8)*255, 5)
    return mask > 0


def image_features_from_crop(img_bgr):
    if img_bgr is None:
        return None
    gw = gray_world(img_bgr)
    mask = skin_mask_hsv_local(gw)
    coverage = float(np.mean(mask))
    use_whole = coverage < 0.06

    lab = cv2.cvtColor(gw, cv2.COLOR_BGR2Lab)
    L_ch, a_ch, b_ch = cv2.split(lab)
    B_ch, G_ch, R_ch = cv2.split(gw)

    def select(ch):
        if use_whole:
            return ch.reshape(-1).astype(np.float32)
        else:
            sel = ch[mask]
            if sel.size == 0:
                return ch.reshape(-1).astype(np.float32)
            return sel.astype(np.float32)

    L = select(L_ch)
    a = select(a_ch)
    b = select(b_ch)
    B = select(B_ch).astype(np.float32) + 1e-6
    G = select(G_ch).astype(np.float32) + 1e-6
    R = select(R_ch).astype(np.float32) + 1e-6
    total = R + G + B
    total[total == 0] = 1e-6
    r = R / total
    g = G / total

    def stats(x):
        if x.size == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75)), float(np.std(x))]

    feats = []
    log_RG = np.log(np.clip(R / G, 1e-6, 1e6))
    log_invR = np.log(np.clip(1.0 / R, 1e-6, 1e6))

    for arr in [L, a, b, r, g, log_RG, log_invR]:
        feats += stats(arr)

    feats = np.array(feats, dtype=np.float32)
    if feats.size != 28:
        out = np.zeros(28, dtype=np.float32)
        out[:min(len(feats), 28)] = feats[:28]
        return out
    return feats


class HbDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, use_skin_features=False, save_crops=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_skin_features = use_skin_features
        self.save_crops = save_crops

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_bgr = cv2.imread(img_path)
        crop = crop_lips_optimized_bgr(img_bgr)
        if crop is None:
            pil_img = Image.open(img_path).convert("RGB")
            img_cv_for_feats = cv2.imread(img_path)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            img_cv_for_feats = crop

        if self.save_crops:
            base = os.path.basename(img_path)
            name, ext = os.path.splitext(base)
            out_path = os.path.join(CROPS_DIR, f"{name}_crop.png")
            cv2.imwrite(out_path, cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = transforms.ToTensor()(pil_img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.use_skin_features:
            feats = image_features_from_crop(img_cv_for_feats)
            if feats is None:
                feats = np.zeros(28, dtype=np.float32)
            feats_t = torch.tensor(feats, dtype=torch.float32)
            return img_tensor, feats_t, label
        else:
            return img_tensor, label


class HbModel(nn.Module):
    def __init__(self, use_skin_features=False, skin_feat_dim=28):
        super(HbModel, self).__init__()
        self.use_skin_features = use_skin_features
        try:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_features = self.backbone.fc.in_features
        except Exception:
            self.backbone = models.resnet18(pretrained=True)
            num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        if use_skin_features:
            self.fc1 = nn.Linear(num_features + skin_feat_dim, 128)
            self.fc2 = nn.Linear(128, 1)
        else:
            self.fc = nn.Linear(num_features, 1)

    def forward(self, x, skin_features=None):
        x = self.backbone(x)
        if self.use_skin_features:
            if skin_features is None:
                raise ValueError("Model expects skin_features but got None")
            x = torch.cat([x, skin_features], dim=1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = self.fc(x)
        return x


IDEAL_K = 6500.0

def estimate_color_temperature_from_bgr(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    R, G, B = float(np.mean(img_rgb[:,:,0])), float(np.mean(img_rgb[:,:,1])), float(np.mean(img_rgb[:,:,2]))
    X = 0.4124*R + 0.3576*G + 0.1805*B
    Y = 0.2126*R + 0.7152*G + 0.0722*B
    Z = 0.0193*R + 0.1192*G + 0.9505*B
    denom = (X+Y+Z) if (X+Y+Z)!=0 else 1.0
    x = X/denom; y = Y/denom
    n = (x - 0.3320) / (0.1858 - y + 1e-9)
    CCT = 449*(n**3) + 3525*(n**2) + 6823.3*n + 5520.33
    return float(CCT)

def compute_confidence_exp(CCT, avg_lab, ref_lab):
    delta_K_norm = abs(CCT - IDEAL_K) / IDEAL_K
    delta_E = float(np.linalg.norm(avg_lab - ref_lab))
    conf = np.exp(-2.5 * delta_K_norm) * np.exp(-0.02 * delta_E)
    conf_percent = float(np.clip(conf * 100.0, 0.0, 100.0))
    lighting_conf = float(np.clip(100.0 * np.exp(-5.0 * delta_K_norm), 0.0, 100.0))
    skin_conf = float(np.clip(100.0 * np.exp(-0.02 * delta_E), 0.0, 100.0))
    return conf_percent, lighting_conf, skin_conf, delta_K_norm, delta_E


def find_image_paths_by_ids(dataset_dir, ids):
    ids_lower = {str(i).lower(): str(i) for i in ids}
    found = {}
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower() in ids_lower:
                found[ids_lower[f.lower()]] = os.path.join(root, f)
    return found


def compute_ref_lab(paths, n_sample=None):
    labs = []
    cnt = 0
    for p in paths:
        try:
            img = cv2.imread(p)
            crop = crop_lips_optimized_bgr(img)
            if crop is None:
                crop = cv2.imread(p)
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).reshape(-1,3)
            valid = np.all(crop > 10, axis=2).reshape(-1)
            if valid.sum() == 0:
                continue
            labs.append(np.mean(lab[valid], axis=0))
            cnt += 1
            if n_sample and cnt >= n_sample:
                break
        except Exception:
            continue
    if len(labs) == 0:
        return np.array([65.0, 14.0, 18.0], dtype=np.float32)
    return np.mean(np.array(labs), axis=0).astype(np.float32)


def predict_and_analyze(model, device, image_path, transform, use_skin_features, REF_LAB, do_save=True):
    img_bgr = cv2.imread(image_path)
    crop = crop_lips_optimized_bgr(img_bgr)
    if crop is None:
        crop = img_bgr.copy()
    crop = remove_black_scribbles(crop, dark_thresh=30)
    feats = image_features_from_crop(crop)
    if feats is None:
        feats = np.zeros(28, dtype=np.float32)

    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_crop).unsqueeze(0).to(device)
    feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(img_tensor, feats_t).item() if use_skin_features else model(img_tensor).item()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray)); contrast = float(np.std(gray))
    avg_lab = np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).reshape(-1,3), axis=0)
    CCT = estimate_color_temperature_from_bgr(crop)
    conf_percent, lighting_conf, skin_conf, dK, dE = compute_confidence_exp(CCT, avg_lab, REF_LAB)

    row = {
        "image_id": os.path.basename(image_path),
        "pred_hgb": float(pred),
        "brightness": brightness,
        "contrast": contrast,
        "CCT": CCT,
        "conf_percent": conf_percent,
        "lighting_conf": lighting_conf,
        "skin_conf": skin_conf,
        "delta_K_norm": dK,
        "delta_E": dE,
        "avg_lab_L": float(avg_lab[0]),
        "avg_lab_a": float(avg_lab[1]),
        "avg_lab_b": float(avg_lab[2])
    }

    if do_save:
        name, _ = os.path.splitext(os.path.basename(image_path))
        cv2.imwrite(os.path.join(CROPS_DIR, f"{name}_finalcrop.png"), crop)
    return row


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="Path to evaluation images folder OR comma-separated list of image paths OR a text file listing image paths")
    p.add_argument("--meta", required=True, help="CSV metadata with columns: image_id,hgb (optional: is_train or split)")
    p.add_argument("--out", required=True, help="Path to output CSV for predictions")
    p.add_argument("--dataset-dir", default=DATASET_DIR, help="Directory to search for training image files")
    p.add_argument("--train-size", type=int, default=19, help="Number of images to use for training if meta doesn't specify")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-save-crops", action="store_true", help="Do not save crop images to disk")
    p.add_argument("--skip-train", action="store_true", help="Skip training and run inference only (random init will be used unless you load weights)")
    return p.parse_args()


def main():
    args = parse_args()

    # reproducibility seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    # deterministic cudnn (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load metadata
    df_meta = pd.read_csv(args.meta)
    if 'image_id' not in df_meta.columns or 'hgb' not in df_meta.columns:
        raise ValueError("meta CSV must contain at least 'image_id' and 'hgb' columns")

    # Resolve evaluation image list (accept folder / text file / comma list)
    eval_paths = []
    if os.path.isdir(args.images):
        for root, _, files in os.walk(args.images):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    eval_paths.append(os.path.join(root, f))
    elif os.path.isfile(args.images):
        with open(args.images, 'r') as fh:
            lines = [l.strip() for l in fh if l.strip()]
        eval_paths = [l for l in lines]
    else:
        eval_paths = [p.strip() for p in args.images.split(',') if p.strip()]

    eval_basenames = set([os.path.basename(p) for p in eval_paths])

    # Determine training rows
    train_rows = None
    if 'is_train' in df_meta.columns:
        train_rows = df_meta[df_meta['is_train'] == 1]
    elif 'split' in df_meta.columns:
        train_rows = df_meta[df_meta['split'].astype(str).str.lower() == 'train']

    # Fallback selection logic. IMPORTANT: for reproducibility we ALLOW overlap between eval and train
    # because the user requested that it's fine if they're in the same folder.
    if train_rows is None or len(train_rows) == 0:
        # Use first N rows from meta directly (do NOT exclude eval set) -- reproducible selection
        train_rows = df_meta.iloc[:args.train_size]
        if len(train_rows) < args.train_size:
            print(f"Warning: meta has only {len(train_rows)} rows but train_size requested {args.train_size}; using available rows")

    train_ids = list(train_rows['image_id'].astype(str))

    # Find files for those train ids in dataset_dir using case-insensitive match
    found_map = find_image_paths_by_ids(args.dataset_dir, train_ids)
    train_image_paths = [found_map[i] for i in train_ids if i in found_map]
    train_labels = [float(train_rows.loc[train_rows['image_id'].astype(str) == i, 'hgb'].values[0]) for i in train_ids if i in found_map]

    if len(train_image_paths) == 0 and not args.skip_train:
        # Try a looser search: maybe images are in eval_paths themselves
        eval_map = {os.path.basename(p).lower(): p for p in eval_paths}
        for i in train_ids:
            if i.lower() in eval_map:
                train_image_paths.append(eval_map[i.lower()])
                train_labels.append(float(train_rows.loc[train_rows['image_id'].astype(str) == i, 'hgb'].values[0]))

    if len(train_image_paths) == 0 and not args.skip_train:
        raise FileNotFoundError(f"No training images found for provided train ids (searched {args.dataset_dir} and --images). Found 0 files.")

    print(f"Using {len(train_image_paths)} images for training (requested {args.train_size}).")

    # Prepare transforms, datasets, loaders
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    use_skin_features = True
    save_crops_flag = not args.no_save_crops

    train_dataset = HbDataset(train_image_paths, train_labels, transform=transform, use_skin_features=use_skin_features, save_crops=save_crops_flag)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) if len(train_dataset) > 0 else None

    # Build model, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HbModel(use_skin_features=use_skin_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Compute REF_LAB (if we have training images)
    REF_LAB = np.array([65.0, 14.0, 18.0], dtype=np.float32)
    if len(train_image_paths) > 0:
        REF_LAB = compute_ref_lab(train_image_paths, n_sample=min(200, len(train_image_paths)))
    print("Computed REF_LAB from training crops:", REF_LAB)

    # Train if requested
    if not args.skip_train and len(train_image_paths) > 0:
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            if train_loader is None:
                break
            for batch in train_loader:
                if use_skin_features:
                    imgs, feats, labels = batch
                    feats = feats.to(device)
                else:
                    imgs, labels = batch
                    feats = None
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(imgs, feats) if use_skin_features else model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
            train_loss /= max(1, len(train_loader.dataset))
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}")

    # Prepare evaluation list: resolve eval_paths into absolute files
    resolved_eval_paths = []
    for p in eval_paths:
        if os.path.isfile(p):
            resolved_eval_paths.append(p)
            continue
        # try basename search in dataset_dir
        b = os.path.basename(p)
        # try find in dataset_dir
        found = find_image_paths_by_ids(args.dataset_dir, [b])
        if b in found:
            resolved_eval_paths.append(found[b])
            continue
        # maybe user provided a basename that is in the meta but stored elsewhere
        # try case-insensitive search in eval_paths themselves
        for ep in eval_paths:
            if os.path.basename(ep).lower() == b.lower():
                resolved_eval_paths.append(ep)
                break

    # Remove duplicates and keep order
    seen = set(); final_eval = []
    for p in resolved_eval_paths:
        if p not in seen:
            final_eval.append(p); seen.add(p)

    print(f"Evaluating on {len(final_eval)} images")

    # Run inference and save
    rows = []
    for p in final_eval:
        try:
            row = predict_and_analyze(model, device, p, transform, use_skin_features, REF_LAB, do_save=save_crops_flag)
            bn = os.path.basename(p)
            if bn in list(df_meta['image_id'].astype(str)):
                true_h = df_meta.loc[df_meta['image_id'].astype(str) == bn, 'hgb'].values[0]
                row['true_hgb'] = float(true_h)
            rows.append(row)
            print(f"Processed {bn} -> predicted {row['pred_hgb']:.2f}, conf {row['conf_percent']:.1f}%")
        except Exception as e:
            print(f"Error processing {p}: {e}")

    df_out = pd.DataFrame(rows)
    out_csv = args.out
    df_out.to_csv(out_csv, index=False)
    print("Saved results to:", out_csv)


if __name__ == '__main__':
    main()
