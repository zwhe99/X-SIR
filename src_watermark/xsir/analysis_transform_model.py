import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from train_watermark_model import TransformModel

def plot_and_save(x, y, title='graph', xlabel='X', ylabel='Y', filename='output.png'):
    plt.figure()
    bins = np.arange(-0.4, 1.05, 0.05)
    indices = np.digitize(x, bins)
    
    y_avg = []
    numbers = []
    for i in range(1, len(bins)):
        bin_y_values = np.array(y)[indices == i]
        avg_value = np.mean(bin_y_values) if bin_y_values.size > 0 else np.nan  # Handle empty bins
        y_avg.append(avg_value)
        numbers.append(bin_y_values.size)

    valid_bins = bins[:-1][~np.isnan(y_avg)]
    valid_y_avg = np.array(y_avg)[~np.isnan(y_avg)]

    plt.bar(valid_bins, valid_y_avg, width=0.05, align='edge', color='blue', label='average y in bin')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    print("file_name:", filename)
    plt.savefig(filename)

def scale_vector_batch(tensor_batch):
    mean = torch.mean(tensor_batch, dim=-1, keepdim=True)
    v_minus_mean = tensor_batch - mean
    v_minus_mean = torch.tanh(1000 * v_minus_mean)
    return v_minus_mean

def cosine_similarity_matrix(batch):
    norm = torch.norm(batch, dim=1).view(-1, 1)
    normed_batch = batch / norm
    similarity = torch.mm(normed_batch, normed_batch.t())
    return similarity

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_model = TransformModel(input_dim=args.input_dim)
    transform_model.load_state_dict(torch.load(args.checkpoint))
    transform_model = transform_model.to(device)
    all_token_embedding = np.loadtxt(args.embedding_file)
    all_token_embedding = torch.tensor(all_token_embedding, device='cuda', dtype=torch.float32)

    with torch.no_grad():
        similarities_origin = cosine_similarity_matrix(all_token_embedding)
        similarities_x = similarities_origin[similarities_origin<0.99]
        similarities_x = similarities_x.view(-1)
        transformed_embedding = transform_model(all_token_embedding)
        similarities_y1 = cosine_similarity_matrix(transformed_embedding)
        similarities_y1 = similarities_y1[similarities_origin<0.99]
        similarities_y1 = similarities_y1.view(-1)


        similarities_y2 = cosine_similarity_matrix(scale_vector_batch(transformed_embedding))
        similarities_y2 = similarities_y2[similarities_origin<0.99]
        similarities_y2 = similarities_y2.view(-1)

        path_origin = os.path.join(args.figure_dir, "origin_graph.png")
        path_scale = os.path.join(args.figure_dir, "scale_graph.png")
        os.makedirs(os.path.dirname(path_origin), exist_ok=True)

        plot_and_save(similarities_x.detach().cpu().numpy(), similarities_y1.detach().cpu().numpy(), filename=path_origin)
        plot_and_save(similarities_x.detach().cpu().numpy(), similarities_y2.detach().cpu().numpy(), filename=path_scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect watermark in texts")
    parser.add_argument('--embedding_file', type=str, default="embeddings/sbert-embeddings-new.txt")
    parser.add_argument('--checkpoint', type=str, default="watermark/transform_model_cbert.pth")
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--figure_dir", type=str, default="figure/")
    args = parser.parse_args()
    main(args)