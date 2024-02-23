import os
import tqdm
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from watermark import WatermarkLogitsProcessor, WatermarkWindow, WatermarkContext

MAX_LENGTH = 512

def main(args):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", trust_remote_code=True)

    if args.watermark_type == "window": # use a window of previous tokens to hash, e.g. KGW
        watermark_model = WatermarkWindow(device, args.window_size, tokenizer)
        logits_processor = WatermarkLogitsProcessor(watermark_model)
    elif args.watermark_type == "context":
        watermark_model = WatermarkContext(device, args.chunk_size, tokenizer, mapping_file=args.mapping_file,delta = args.delta,transform_model_path=args.transform_model, embedding_model=args.embedding_model)
        logits_processor = WatermarkLogitsProcessor(watermark_model)
    else:
        watermark_model, logits_processor = None, None

    generation_config = {
        "do_sample": True,
        "min_new_tokens":2,
        "max_new_tokens": MAX_LENGTH,
        "eos_token_id": tokenizer.eos_token_id,
        "no_repeat_ngram_size": 5,
        "logits_processor": LogitsProcessorList([logits_processor])
    }

    # Load data
    with open(args.prmopt_file, "r") as f:
        prompt_list = json.load(f)

    output_list = []
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            output_list = json.load(f)
    else:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if len(output_list) == len(prompt_list):
        print("Data already generated. Skipping...")
        return

    prompt_list = prompt_list[len(output_list):]
    for prompt in tqdm.tqdm(prompt_list):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
            outputs = outputs[:, inputs["input_ids"].shape[-1]:]
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_list.append(generated_text)

            with open(args.output_file, "w") as f:
                json.dump(output_list, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate with watermarking')
    parser.add_argument('--watermark_type', type=str, default="context")
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True, help="""Output file to save the generated text. A json file of list of strings.""")
    parser.add_argument('--prmopt_file', type=str, required=True, help="Prompt file to generate the text. A json file of list of strings.")
    parser.add_argument('--mapping_file', type=str, required=True)
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--transform_model', type=str, default="model/transform_model_x-sbert.pth")
    parser.add_argument('--embedding_model', type=str, default="c-bert")

    args = parser.parse_args()
    main(args)