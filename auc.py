import json
import argparse
from sklearn.metrics import roc_auc_score

def main(args):
    with open(args.ref_zscore, "r") as f:
        ref_list = json.load(f)
    
    with open(args.wm_zscore, "r") as f:
        wm_list = json.load(f)

    ref_zscore = [x["z_score"] if x["z_score"] is not None else 0 for x in ref_list]
    ref_true = [0 for x in ref_list]

    wm_zscore = [x["z_score"] if x["z_score"] is not None else 0 for x in wm_list]
    wm_true = [1 for x in wm_list]

    y_true = ref_true + wm_true
    y_scores = ref_zscore + wm_zscore

    auc = roc_auc_score(y_true, y_scores)

    print(f"AUC: {auc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate with watermarking')
    parser.add_argument("--ref_zscore", type=str, required=True, help="Reference zscore file")
    parser.add_argument("--wm_zscore", type=str, required=True, help="Watermark zscore file")

    args = parser.parse_args()
    main(args)