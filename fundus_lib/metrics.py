# metrics.py
# evaluation metrics styled to line up with cen et al. 2021: per-bigclass
# one-vs-rest AUC (fig 2a/2b), macro AUC, and frequency-weighted F1 (their
# headline 0.923). all five model notebooks call compute_metrics so the numbers
# are computed identically and are therefore comparable.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score,
    roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score, precision_recall_fscore_support,
)

# short labels for bigclasses 0-9 used in the per-class roc legend so the
# figure stays readable. mirrors data.bigclass_names but trimmed for plotting.
bigclass_short = {
    0: "Normal/Tessellated/DR1",
    1: "DR2-DR3",
    2: "BRVO/CRVO",
    3: "RAO",
    4: "Rhegmatogenous RD",
    5: "CSCR/VKH",
    6: "Maculopathy",
    7: "ERM",
    8: "MH",
    9: "Pathological myopia",
}


def one_hot(y_true, n_classes):
    oh = np.zeros((len(y_true), n_classes), dtype=int)
    oh[np.arange(len(y_true)), y_true] = 1
    return oh


def compute_metrics(y_true, y_prob, index_to_class=None):
    # y_true: (N,) int labels in [0, K). y_prob: (N, K) softmax probabilities.
    # returns a dict of scalar metrics plus per-class AUC. classes absent from
    # y_true (can happen in a small fold) get AUC = nan and are skipped in macro.
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = y_prob.shape[1]
    y_pred = y_prob.argmax(axis=1)
    oh = one_hot(y_true, n_classes)

    per_class_auc = {}
    for k in range(n_classes):
        name = index_to_class[k] if index_to_class else k
        if oh[:, k].sum() == 0 or oh[:, k].sum() == len(y_true):
            per_class_auc[name] = float("nan")  # undefined (all neg or all pos)
        else:
            per_class_auc[name] = roc_auc_score(oh[:, k], y_prob[:, k])

    valid = [v for v in per_class_auc.values() if not np.isnan(v)]
    macro_auc = float(np.mean(valid)) if valid else float("nan")

    # weighted (a.k.a. frequency-weighted) one-vs-rest AUC
    try:
        weighted_auc = roc_auc_score(oh, y_prob, average="weighted", multi_class="ovr")
    except ValueError:
        weighted_auc = float("nan")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        "per_class_auc": per_class_auc,
    }


def roc_points(y_true, y_prob, class_index):
    # fpr/tpr for one class (one-vs-rest), for plotting fig-2a-style overlays
    y_true = np.asarray(y_true)
    binary = (y_true == class_index).astype(int)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return None
    fpr, tpr, _ = roc_curve(binary, np.asarray(y_prob)[:, class_index])
    return fpr, tpr


def pr_points(y_true, y_prob, class_index):
    # recall/precision for one class (one-vs-rest) + auprc (average precision),
    # for plotting fig-2b-style precision-recall overlays. mirrors roc_points.
    y_true = np.asarray(y_true)
    binary = (y_true == class_index).astype(int)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return None
    scores = np.asarray(y_prob)[:, class_index]
    precision, recall, _ = precision_recall_curve(binary, scores)
    ap = average_precision_score(binary, scores)
    return recall, precision, ap


def pr_per_class_table(y_true, y_prob, index_to_class=None):
    # per-class precision/recall/f1 at the argmax decision plus the
    # threshold-independent auprc (average precision) and the prevalence baseline
    # (the auprc a no-skill classifier would get for that class). returns a list
    # of dicts, one per class, ready to drop into a DataFrame.
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = y_prob.shape[1]
    y_pred = y_prob.argmax(axis=1)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0)
    rows = []
    for k in range(n_classes):
        binary = (y_true == k).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            ap = float("nan")
        else:
            ap = average_precision_score(binary, y_prob[:, k])
        name = index_to_class[k] if index_to_class else k
        rows.append({
            "class": name,
            "support": int(support[k]),
            "prevalence": float(binary.mean()),
            "precision": float(prec[k]),
            "recall": float(rec[k]),
            "f1": float(f1[k]),
            "auprc": float(ap),
        })
    return rows


def auprc_summary(y_true, y_prob):
    # macro and frequency-weighted average precision across classes. weights are
    # class supports, matching how weighted_auc is computed in compute_metrics.
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = y_prob.shape[1]
    aps, weights = [], []
    for k in range(n_classes):
        binary = (y_true == k).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        aps.append(average_precision_score(binary, y_prob[:, k]))
        weights.append(binary.sum())
    if not aps:
        return {"macro_auprc": float("nan"), "weighted_auprc": float("nan")}
    aps = np.array(aps, dtype=float); weights = np.array(weights, dtype=float)
    return {
        "macro_auprc": float(aps.mean()),
        "weighted_auprc": float(np.average(aps, weights=weights)),
    }


def plot_pr_per_class(y_true, y_prob, index_to_class=None,
                      title=None, save_path=None, legend_outside=True):
    # fig-2b-style per-class one-vs-rest precision-recall overlay with auprc in
    # the legend. unlike roc there is no single chance line (the no-skill level is
    # each class's prevalence), so we annotate auprc per curve instead.
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = y_prob.shape[1]
    if index_to_class is None:
        index_to_class = {k: str(k) for k in range(n_classes)}

    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
    figsize = (9.5, 6) if legend_outside else (7, 6)
    fig, ax = plt.subplots(figsize=figsize)
    for k in range(n_classes):
        binary = (y_true == k).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        precision, recall, _ = precision_recall_curve(binary, y_prob[:, k])
        ap = average_precision_score(binary, y_prob[:, k])
        ax.plot(recall, precision, color=cmap(k % cmap.N), lw=1.5,
                label=f"{index_to_class[k]} (auprc = {ap:.4f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=8, framealpha=0.95, borderaxespad=0.0)
    else:
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def confusion(y_true, y_prob, n_classes):
    y_pred = np.asarray(y_prob).argmax(axis=1)
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def aggregate_folds(fold_metrics):
    # fold_metrics: list of dicts from compute_metrics across folds.
    # returns mean and std for each scalar metric (per_class_auc handled separately).
    scalar_keys = [k for k, v in fold_metrics[0].items() if not isinstance(v, dict)]
    summary = {}
    for k in scalar_keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        summary[k + "_mean"] = float(np.nanmean(vals))
        summary[k + "_std"] = float(np.nanstd(vals))
    return summary


# ----------------------------------------------------------------------------
# plotting helpers
# ----------------------------------------------------------------------------
def plot_history(histories, model_name="model", save_path=None, smooth=False):
    # histories: either a single fit() history (list of dicts) or a list of
    # such histories (one per fold). plots train/val loss and train/val accuracy
    # across epochs in the same style as classic keras training curves. when
    # multiple folds are passed, each fold is drawn as a thin line and the mean
    # across folds is overlaid as a thicker line.
    if len(histories) == 0:
        raise ValueError("empty histories")
    # normalize to list-of-histories
    if isinstance(histories[0], dict):
        histories = [histories]

    def _series(h, key):
        return np.array([row[key] for row in h], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax_acc, ax_loss = axes

    for i, h in enumerate(histories):
        epochs = _series(h, "epoch")
        ax_acc.plot(epochs, _series(h, "train_acc"), color="tab:blue", alpha=0.35, lw=1)
        ax_acc.plot(epochs, _series(h, "val_acc"),   color="tab:orange", alpha=0.35, lw=1)
        ax_loss.plot(epochs, _series(h, "train_loss"), color="tab:blue", alpha=0.35, lw=1)
        ax_loss.plot(epochs, _series(h, "val_loss"),  color="tab:orange", alpha=0.35, lw=1)

    # mean across folds (only over the common epoch range)
    if len(histories) > 1:
        min_len = min(len(h) for h in histories)
        ep = np.arange(min_len)
        def _stack(key): return np.stack([_series(h, key)[:min_len] for h in histories], axis=0)
        ax_acc.plot(ep, _stack("train_acc").mean(0), color="tab:blue", lw=2.2, label="Training Accuracy")
        ax_acc.plot(ep, _stack("val_acc").mean(0),   color="tab:orange", lw=2.2, label="Validation Accuracy")
        ax_loss.plot(ep, _stack("train_loss").mean(0), color="tab:blue", lw=2.2, label="Training Loss")
        ax_loss.plot(ep, _stack("val_loss").mean(0),   color="tab:orange", lw=2.2, label="Validation Loss")
    else:
        h = histories[0]
        ax_acc.plot(_series(h, "epoch"), _series(h, "train_acc"), color="tab:blue", lw=2.2, label="Training Accuracy")
        ax_acc.plot(_series(h, "epoch"), _series(h, "val_acc"),   color="tab:orange", lw=2.2, label="Validation Accuracy")
        ax_loss.plot(_series(h, "epoch"), _series(h, "train_loss"), color="tab:blue", lw=2.2, label="Training Loss")
        ax_loss.plot(_series(h, "epoch"), _series(h, "val_loss"),   color="tab:orange", lw=2.2, label="Validation Loss")

    ax_acc.set_title(f"{model_name} Accuracy")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="lower right")

    ax_loss.set_title(f"{model_name} Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="upper right")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(y_true, y_prob, index_to_class=None,
                          normalize="true", title=None, save_path=None,
                          annotate=True, annotate_counts=True, figsize=None):
    # row-normalized confusion matrix with per-cell percentages (and raw counts
    # in parentheses by default). normalize="true" -> rows sum to 1 (sensitivity
    # per class). normalize="pred" -> columns sum to 1 (precision per class).
    # normalize=None -> raw counts.
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    n_classes = y_prob.shape[1]
    if index_to_class is None:
        index_to_class = {k: str(k) for k in range(n_classes)}
    labels = [str(index_to_class[k]) for k in range(n_classes)]

    cm_counts = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    if normalize == "true":
        denom = cm_counts.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1, denom)
        cm = cm_counts.astype(float) / denom
        cbar_label = "Row-normalized (sensitivity per true class)"
    elif normalize == "pred":
        denom = cm_counts.sum(axis=0, keepdims=True)
        denom = np.where(denom == 0, 1, denom)
        cm = cm_counts.astype(float) / denom
        cbar_label = "Column-normalized (precision per predicted class)"
    else:
        cm = cm_counts.astype(float)
        cbar_label = "Count"

    if figsize is None:
        figsize = (max(7, 0.6 * n_classes + 4), max(6, 0.55 * n_classes + 3))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=(1 if normalize else cm.max()))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    if title is not None:
        ax.set_title(title)

    if annotate:
        # color text white on dark cells for legibility
        thresh = (cm.max() if normalize is None else 1.0) * 0.55
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm[i, j]
                if normalize:
                    s = f"{val:.2f}"
                    if annotate_counts:
                        s += f"\n({cm_counts[i, j]})"
                else:
                    s = f"{int(val)}"
                ax.text(j, i, s, ha="center", va="center",
                        fontsize=8,
                        color="white" if val > thresh else "black")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_per_class(y_true, y_prob, index_to_class=None,
                       xlim=(0.0, 0.20), ylim=(0.80, 1.0),
                       title=None, save_path=None,
                       legend_outside=True, autoscale=True):
    # fig-2a-style per-class one-vs-rest ROC overlay. by default zoomed into the
    # top-left corner like cen et al., but with two safety nets so the figure
    # stays readable for weaker models whose curves don't hug the corner:
    #   - legend_outside=True puts the legend to the right of the axes so it
    #     never covers a curve
    #   - autoscale=True falls back to the full (0,1) range whenever a curve
    #     would be clipped (any tpr below ylim[0] or fpr above xlim[1])
    # pass xlim=None / ylim=None to force full range. pass autoscale=False to
    # force the zoom regardless of clipping.
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = y_prob.shape[1]
    if index_to_class is None:
        index_to_class = {k: str(k) for k in range(n_classes)}

    # compute curves once so we can detect clipping before choosing limits
    curves = []
    for k in range(n_classes):
        binary = (y_true == k).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        fpr, tpr, _ = roc_curve(binary, y_prob[:, k])
        auc_k = roc_auc_score(binary, y_prob[:, k])
        curves.append((k, fpr, tpr, auc_k))

    # decide axis limits
    use_full = (xlim is None) or (ylim is None)
    if not use_full and autoscale:
        # if any curve enters the visible x window with a tpr below ylim[0],
        # the zoom is misleading -- expand to full range
        for _, fpr, tpr, _ in curves:
            visible = fpr <= xlim[1]
            if visible.any() and tpr[visible].min() < ylim[0]:
                use_full = True
                break
    if use_full:
        xlim_eff, ylim_eff = (0.0, 1.0), (0.0, 1.0)
    else:
        xlim_eff, ylim_eff = xlim, ylim

    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
    figsize = (9.5, 6) if legend_outside else (7, 6)
    fig, ax = plt.subplots(figsize=figsize)
    for k, fpr, tpr, auc_k in curves:
        ax.plot(fpr, tpr, color=cmap(k % cmap.N), lw=1.5,
                label=f"{index_to_class[k]} (auc = {auc_k:.4f})")
    if use_full:
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls="--")  # chance line

    ax.set_xlim(*xlim_eff)
    ax.set_ylim(*ylim_eff)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=8, framealpha=0.95, borderaxespad=0.0)
    else:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
