
"""
This script generates text using an LLM and supports watermarking techniques.
We follow KGW (Kirchenbauer et al., 2023) to constrain length of the generated text to ~200 tokens.
"""
import os
import sys
import tqdm
import torch
import argparse
from transformers.utils import is_flash_attn_2_available
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, GenerationConfig
from src_watermark.xsir.watermark import (
    WatermarkWindow as XSIRWindow,
    WatermarkContext as XSIRContext,
    WatermarkLogitsProcessor as XSIRLogitsProcessor
)
from src_watermark.kgw.extended_watermark_processor import (
    WatermarkLogitsProcessor as KGWLogitsProcessor
)
from src_watermark.uw import (
    Delta_Reweight,
    Gamma_Reweight,
    WatermarkLogitsProcessor as UWLogitsProcessor,
    PrevN_ContextCodeExtractor,
    patch_model
)

from utils import read_jsonl, append_jsonl

OUTPUT_LENGTH = 200

def main(args):
    print(args)
    assert not (args.fp16 and args.bf16), "Cannot use both fp16 and bf16"

    # seed & device
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    input_data = read_jsonl(args.input_file)

    output_data = []
    if os.path.exists(args.output_file):
        output_data = read_jsonl(args.output_file)
    else:
        if os.path.dirname(args.output_file) != "":
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if len(input_data) == len(output_data):
        print("Data already generated. Skipping...")
        return

    prompt_list = [d["prompt"] for d in input_data[len(output_data):]]

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() and (args.fp16 or args.bf16) else "eager",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
            trust_remote_code=True
        )
    except ValueError as e:
        if "does not support Flash Attention 2.0" in str(e):
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                device_map="auto",
                torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
                trust_remote_code=True
            )
        else:
            raise e

    # Set padding_side & pad token for batch generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        print("Set pad token to eos token")

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # Load watermark
    if args.watermark_method in ["xsir", "sir"]:
        if args.watermark_type == "window": # use a window of previous tokens to hash, e.g. KGW
            watermark_model = XSIRWindow(
                device,
                args.window_size,
                tokenizer
            )
            logits_processor = XSIRLogitsProcessor(watermark_model)
        elif args.watermark_type == "context":
            watermark_model = XSIRContext(
                device,
                args.chunk_size,
                tokenizer,
                mapping_file=args.mapping_file,
                delta=args.delta,
                transform_model_path=args.transform_model,
                embedding_model=args.embedding_model
            )
            logits_processor = XSIRLogitsProcessor(watermark_model)
        else:
            raise ValueError(f"Incorrect watermark type: {args.watermark_type}")
    elif args.watermark_method == "kgw":
        logits_processor = KGWLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            delta=args.delta,
            seeding_scheme=args.seeding_scheme
        )
    elif args.watermark_method == "uw":
        logits_processor = UWLogitsProcessor(
            b"42",
            Delta_Reweight(),
            PrevN_ContextCodeExtractor(5),
        )
    elif args.watermark_method == "no":
        logits_processor = None
    else:
        raise ValueError(f"Incorrect watermark method: {args.watermark_method}")

    # Generate
    generation_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=OUTPUT_LENGTH + 5,
        min_new_tokens=OUTPUT_LENGTH - 5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=4,
        repetition_penalty=1.05, # reduce repetition (we found that repetition might result in high z-score accidentially, even for non-watermarked text)
    )

    for batch in tqdm.tqdm(range(0, len(prompt_list), args.batch_size)):
        batch_prompts = prompt_list[batch:batch+args.batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False).to(device)
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]

        # Remove the last token if it is eos token
        input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
        attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                generation_config=generation_config,
                logits_processor=LogitsProcessorList([logits_processor]) if logits_processor is not None else None
            )

            for i, (in_ids, gen_ids) in enumerate(zip(input_ids, generated_ids)):
                # Remove input tokens from generated tokens
                in_text = tokenizer.decode(in_ids, skip_special_tokens=True)
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                new_text = gen_text[len(in_text):]

                # Append to output file
                append_jsonl(args.output_file, {"prompt": batch_prompts[i], "response": new_text})

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Generate with watermarking')
    # Model
    parser.add_argument('--base_model', type=str, required=True, help="Base model to generate text from")
    parser.add_argument('--fp16', action="store_true", help="Use fp16")
    parser.add_argument('--bf16', action="store_true", help="Use bf16")

    # Data
    parser.add_argument('--input_file', type=str, required=True, help="Input file containing prompts")
    parser.add_argument('--output_file', type=str, required=True, help="Output file to save generated text")

    # Watermark
    parser.add_argument('--watermark_method', type=str, choices=["xsir", "sir", "kgw", "uw", "no"], default="no", help="Watermarking method")
    parser.add_argument('--delta', type=float, default=None, help="bias of logit")

    # X-SIR
    parser.add_argument('--watermark_type', type=str, default="context")
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--mapping_file', type=str, default="mapping.json")
    parser.add_argument('--transform_model', type=str, default="model/transform_model_x-sbert.pth")
    parser.add_argument('--embedding_model', type=str, default="paraphrase-multilingual-mpnet-base-v2")

    # KGW
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--seeding_scheme', type=str, default="minhash")

    # Generation
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()

    # Manually set default value for delta based on watermark_method
    if args.watermark_method == "kgw" and args.delta is None:
        args.delta = 2
    elif args.watermark_method in ["xsir", "sir"] and args.delta is None:
        args.delta = 1

    main(args)