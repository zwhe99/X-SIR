import os
import json
import torch
import argparse
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

class SentenceEmbeddings:

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sbert = None
        assert self.args.model_path in ["perceptiveshawty/compositional-bert-large-uncased", "paraphrase-multilingual-mpnet-base-v2"], f"embedding_model {self.args.model_path} not supported"
        if self.args.model_path == "perceptiveshawty/compositional-bert-large-uncased":
            # C-BERT
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
            self.model = AutoModel.from_pretrained(self.args.model_path).to(self.device)
            self.sbert = False
        else:
            # S-BERT
            self.model = SentenceTransformer(self.args.model_path).to(self.device)
            self.sbert = True

    def get_embedding(self, sentence):
        """Generate embedding for a sentence."""
        if not self.sbert:
            input_ids = self.tokenizer.encode(sentence, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model(input_ids)
            return output[0][:, 0, :].cpu().numpy()
        else:
            return self.model.encode(sentence, show_progress_bar=False)

    def generate_embeddings(self, input_path, output_path, generate_size=1000):
        """Generate embeddings for all sentences in the input file."""
        all_embeddings = []
        with open(input_path, 'r') as f:
            lines = f.readlines()

        pbar = tqdm(total=generate_size, desc="Embeddings generated")
        for line in lines:
            data = json.loads(line)
            all_embeddings.append(self.get_embedding(data['sentence1']))
            all_embeddings.append(self.get_embedding(data['sentence2']))
            pbar.update(2)
            if len(all_embeddings) >= generate_size:
                break
        pbar.close()

        all_embeddings = np.vstack(all_embeddings)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savetxt(output_path, all_embeddings, delimiter=" ")


def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for sentences.')
    parser.add_argument('--input_path', type=str, required=True, help='Input file path')
    parser.add_argument('--output_path', type=str, required=True, help='Output file path')
    parser.add_argument('--model_path', type=str, required=True, help='Path of the embedding model')
    parser.add_argument('--size', type=int, required=False, default=1000, help='Size of the data to generate embeddings for')
    args = parser.parse_args()

    sentence_embeddings = SentenceEmbeddings(args)
    sentence_embeddings.generate_embeddings(args.input_path, args.output_path, args.size)

if __name__ == '__main__':
    main()
