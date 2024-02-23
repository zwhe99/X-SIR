import json
import tqdm
import torch
import argparse

from transformers import AutoTokenizer
from watermark import WatermarkWindow, WatermarkContext

def get_length(text, tokenizer):
    return len(tokenizer.encode(text))

def is_nan(nan):
    return nan != nan

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    if args.watermark_type == "window": # use a window of previous tokens to hash, e.g. KGW
        watermark_model = WatermarkWindow(device, args.window_size, tokenizer)
    elif args.watermark_type == "context":
        watermark_model = WatermarkContext(device, args.chunk_size, tokenizer, mapping_file=args.mapping_file,delta = args.delta,transform_model_path=args.transform_model, embedding_model=args.embedding_model)
    else:
        watermark_model = None

    detect_data = json.load(open(args.detect_file))
    result = []
    with torch.no_grad():
        for dd in tqdm.tqdm(detect_data):
            z_score = watermark_model.detect(dd)
            if is_nan(z_score):
                z_score = None

            result.append({"z_score": z_score, "text": dd})

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare the z-scores of strings in detect_file.')
    parser.add_argument('--watermark_type', type=str, default="context")
    parser.add_argument('--detect_file', type=str, required=True)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--mapping_file', type=str, required=True)
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--transform_model', type=str, default="model/transform_model_x-sbert_test.pth")
    parser.add_argument('--embedding_model', type=str, default="paraphrase-multilingual-mpnet-base-v2")

    args = parser.parse_args()
    main(args)