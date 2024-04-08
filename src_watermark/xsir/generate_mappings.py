import os
import json
import random
import argparse
from transformers import AutoTokenizer

# set seed
random.seed(0)

def generate_mapping(size=30000, dimension=300):
    return [random.randint(0, dimension-1) for _ in range(size)]

def main():
    parser = argparse.ArgumentParser(description='Generate mappings.')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mapping = generate_mapping(tokenizer.vocab_size, 300)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(mapping, f, indent=4)

if __name__ == '__main__':
    main()