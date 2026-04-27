import os
import glob
import random
import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, log_loss, jaccard_score,confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

AOI_BOUNDS = [-64.57541111111111, -31.82181111111111,-64.48602222222222, -31.767622222222222]
ESRI_LULC_CLASSES = {
    0:  ("Unlabeled Data","#5A5A5A"),
    1:  ("Water", "#02487E"),
    2:  ("Trees", "#05701E"),
    3:  ("Grass", "#A8D46F"),
    4:  ("Flooded Veg.", "#6FDAE3"),
    5:  ("Crops", "#48D806"),
    6:  ("Scrub or Shrub", "#C9A227"),
    7:  ("Built Area", "#A2A19D"),
    8:  ("Bare Ground", "#683B05"),
    9:  ("Snow or Ice", "#FFFFFF"),
    10: ("Clouds", "#B5EEFF"),
    11: ("Rangeland", "#E5B116"),
}

def load_data(base_path="../landsat_data/", lulc_path="lulc.tif"):
    if not os.path.exists(base_path):
        tifs = glob.glob("*_SR_B2.TIF")
        if not tifs:
            raise FileNotFoundError("Landsat TIF files not found. Run data_ingestion notebook first.")
        base_path = ""

    band_patterns = {"blue":  "*_SR_B2.TIF","green": "*_SR_B3.TIF","red":   "*_SR_B4.TIF","nir08": "*_SR_B5.TIF",}
    band_files = {}
    for name, pat in band_patterns.items():
        matches = glob.glob(f"{base_path}{pat}")
        if not matches:
            raise FileNotFoundError(f"Missing band: {pat}")
        band_files[name] = matches[0]

    def read_band(path):
        return rioxarray.open_rasterio(path, masked=True).rio.clip_box(*AOI_BOUNDS, crs="EPSG:4326")

    ls_bands = []
    for name in ["blue", "green", "red", "nir08"]:
        ls_bands.append(read_band(band_files[name]).assign_coords(band=[name]))
    ls_data = xr.concat(ls_bands, dim="band").squeeze()

    if not os.path.exists(lulc_path):
        raise FileNotFoundError(f"{lulc_path} not found. Run data_ingestion notebook first.")
    lulc_data = rioxarray.open_rasterio(lulc_path, masked=True).squeeze()

    if ls_data.shape[1:] != lulc_data.shape:
        lulc_data = lulc_data.rio.reproject_match(ls_data)

    return ls_data, lulc_data

def preprocess(ls_data):
    ls_scaled = ls_data.astype(np.float32) * 0.0000275 - 0.2
    ls_scaled = np.clip(ls_scaled, 0, 1)

    red = ls_scaled.sel(band="red")
    nir = ls_scaled.sel(band="nir08")
    ndvi = (nir - red) / (nir + red + 1e-8)

    rgb = ls_scaled.sel(band=["red", "green", "blue"]).values
    rgb_vis = np.clip(rgb / 0.3, 0, 1)

    return ls_scaled, ndvi, rgb_vis

def create_patches(features, labels, patch_size=32, stride=4):
    X, y, coords = [], [], []
    pad = patch_size // 2
    feat_pad = np.pad(features, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    label_pad = np.pad(labels, ((pad,pad), (pad,pad)), mode="reflect")
    h, w = labels.shape

    for i in range(0, h, stride):
        for j in range(0, w, stride):
            if i + patch_size <= h + 2 * pad and j + patch_size <= w + 2 * pad:
                X.append(feat_pad[:, i:i + patch_size, j:j + patch_size])
                y.append(label_pad[i:i+patch_size, j:j+patch_size])
                coords.append((i,j))
                
    return np.array(X), np.array(y), np.array(coords)

def print_split_summary(splits):
    ps = splits["patch_size"]
    ic = splits.get("in_channels", "N/A")
    print(f"Patch size : {ps} × {ps}")
    print(f"Channels : {ic}\n")
    print(f"{'Subset':<10} {'Samples':>10} {'X_Shape':>22} {'y_Shape':>22}")
    print("─" * 70)
    for name in ("train", "val", "test"):
        X = splits[f"X_{name}"]
        y = splits[f"y_{name}"]
        print(f"{name:<10} {len(X):>10,}    {str(X.shape):>22} {str(y.shape):>22}")
    if "label_map" in splits:
        print(f"\nClasses labels changed : "
              f"{[f'{int(k)} to {v}' for k, v in splits['label_map'].items()]}")

def load_processed_data(directory="../processed_data"):
    print(f"Loading dataset from {directory}")
    X_train = np.load(os.path.join(directory, "X_train.npy"))
    y_train = np.load(os.path.join(directory, "y_train.npy"))
    X_val = np.load(os.path.join(directory, "X_val.npy"))
    y_val = np.load(os.path.join(directory, "y_val.npy"))
    X_test = np.load(os.path.join(directory, "X_test.npy"))
    y_test = np.load(os.path.join(directory, "y_test.npy"))
    unique_labels = np.load(os.path.join(directory, "unique_labels.npy"))
    
    c_train = np.load(os.path.join(directory, "c_train.npy")) if os.path.exists(os.path.join(directory, "c_train.npy")) else None
    c_val   = np.load(os.path.join(directory, "c_val.npy")) if os.path.exists(os.path.join(directory, "c_val.npy")) else None
    c_test  = np.load(os.path.join(directory, "c_test.npy")) if os.path.exists(os.path.join(directory, "c_test.npy")) else None

    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    
    return {"X_train": X_train, "y_train": y_train, "c_train": c_train,
        "X_val": X_val, "y_val": y_val, "c_val": c_val,
        "X_test": X_test, "y_test": y_test, "c_test": c_test,
        "unique_labels": unique_labels,
        "label_map": label_map,
        "num_classes": len(unique_labels),
        "patch_size": X_train.shape[-1],
        "in_channels": X_train.shape[1]}

class AugmentedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def evaluate(model, X, y_true, num_classes, device, class_names, subset_name="Test", batch_size=64, show_cm=True, show_roc=True):
    model.eval()
    ds = AugmentedDataset(X, y_true)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for bx, by in loader:
            out = model(bx.to(device))
            prob = torch.softmax(out, dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)
            all_true.extend(by.numpy().flatten())
            all_pred.extend(pred.flatten())
            all_prob.extend(prob.transpose(0, 2, 3, 1).reshape(-1, num_classes))

    yt = np.array(all_true)
    yp = np.array(all_pred)
    yprob = np.array(all_prob)

    acc = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, average="weighted", zero_division=0)
    rec = recall_score(yt, yp, average="weighted", zero_division=0)
    f1 = f1_score(yt, yp, average="weighted", zero_division=0)
    ll = log_loss(yt, yprob, labels=list(range(num_classes)))
    iou = jaccard_score(yt, yp, average="weighted", zero_division=0)
    dice = 2 * iou / (1 + iou) 

    print(f"\n{'─'*52}")
    print(f"Evaluation    {subset_name}")
    print(f"{'─'*52}")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision       : {prec:.4f}")
    print(f"Recall          : {rec:.4f}")
    print(f"F1 Score        : {f1:.4f}")
    print(f"Log Loss        : {ll:.4f}")
    print(f"IoU (Jaccard)   : {iou:.4f}")
    print(f"Dice Score      : {dice:.4f}")
    print(f"{'─'*52}")

    if show_cm:
        cm = confusion_matrix(yt, yp, labels=range(num_classes))
        fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix – {subset_name}")
        plt.tight_layout(); plt.show()

    if show_roc:
        yt_bin = label_binarize(yt, classes=range(num_classes))
        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(num_classes):
            if yt_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(yt_bin[:, i], yprob[:, i])
            ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc(fpr,tpr):.2f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC Curves – {subset_name}")
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout(); plt.show()

    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "LogLoss": ll, "IoU": iou, "Dice": dice}
