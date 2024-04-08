import argparse
import matplotlib.pyplot as plt

from utils import read_jsonl
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

def tpr_at_fpr(fpr, tpr, fpr_target):
    fpr_tpr_interpolation = interpolate.interp1d(fpr, tpr, kind='linear')
    return fpr_tpr_interpolation(fpr_target)

def f1_at_fpr(y_true, y_scores, fpr_target):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Finding the threshold for our target FPR
    threshold = thresholds[next(i for i in range(len(fpr)) if fpr[i] > fpr_target) - 1]
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)

    # Interpolating to find precision and recall at the target threshold
    precision_interp = interpolate.interp1d(thresholds_pr, precision[:-1], fill_value="extrapolate")
    recall_interp = interpolate.interp1d(thresholds_pr, recall[:-1], fill_value="extrapolate")
    precision_at_threshold = precision_interp(threshold)
    recall_at_threshold = recall_interp(threshold)

    # Calculate F1 score
    f1 = 2 * (precision_at_threshold * recall_at_threshold) / (precision_at_threshold + recall_at_threshold)

    return f1

def main(args):
    hm_list = read_jsonl(args.hm_zscore)
    wm_list = read_jsonl(args.wm_zscore)

    hm_zscore = [x["z_score"] if x["z_score"] is not None else 0 for x in hm_list]
    hm_true = [0 for x in hm_list]

    wm_zscore = [x["z_score"] if x["z_score"] is not None else 0 for x in wm_list]
    wm_true = [1 for x in wm_list]

    y_true = hm_true + wm_true
    y_scores = hm_zscore + wm_zscore

    auc = roc_auc_score(y_true, y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    print(f"""AUC: {auc:.3f}

TPR@FPR=0.1: {tpr_at_fpr(fpr, tpr, 0.1):.3f}
TPR@FPR=0.01: {tpr_at_fpr(fpr, tpr, 0.01):.3f}

F1@FPR=0.1: {f1_at_fpr(y_true, y_scores, 0.1):.3f}
F1@FPR=0.01: {f1_at_fpr(y_true, y_scores, 0.01):.3f}
"""
    )

    if args.roc_curve:
        with open(args.roc_curve, "w") as f:
            f.write(f"FPR\tTPR\n")
            for i in range(len(fpr)):
                f.write(f"{fpr[i]:.3f}\t{tpr[i]:.3f}\n")

        plt.plot(fpr, tpr)
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.grid()
        plt.savefig(f"{args.roc_curve}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate with watermarking')
    parser.add_argument("--hm_zscore", type=str, required=True, help="Human zscore file")
    parser.add_argument("--wm_zscore", type=str, required=True, help="Watermark zscore file")
    parser.add_argument("--roc_curve", type=str, default=None, help="ROC curve file")

    args = parser.parse_args()
    main(args)