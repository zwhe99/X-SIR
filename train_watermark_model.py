import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
from torch.optim import lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = out + x 
        return out

class TransformModel(nn.Module):
    def __init__(self, num_layers=4, input_dim=1024, hidden_dim=500, output_dim=300):
        super(TransformModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(ResidualBlock(hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        
        return x

def cosine_similarity(x, y):
    dot_product = torch.sum(x * y, dim=-1)
    norm_x = torch.norm(x, p=2, dim=-1)
    norm_y = torch.norm(y, p=2, dim=-1)
    return dot_product / (norm_x * norm_y)

def row_col_mean_penalty(output):
    row_mean_penalty = torch.mean(output, dim=1).pow(2).sum()
    col_mean_penalty = torch.mean(output, dim=0).pow(2).sum()
    return row_mean_penalty + col_mean_penalty

def abs_value_penalty(output):
    penalties = torch.relu(0.05 - torch.abs(output))
    mask = (penalties > 0).float()  # Create a mask where penalties are non-zero
    non_zero_count = torch.max(mask.sum(), torch.tensor(1.0))  # Avoid division by zero
    return (penalties * mask).sum() / non_zero_count

def vector_transform(vec):
    mean = vec.mean(dim=0, keepdim=True)
    centered_x = vec - mean
    transformed_x = torch.tanh(30*centered_x)
    return transformed_x

def loss_fn(output_a, output_b, input_a, input_b, lambda1=0.1, lambda2=1, median_value=0.4):
    original_similarity = cosine_similarity(input_a, input_b)
    original_similarity = torch.tanh(20*(original_similarity - median_value))
    transformed_similarity = cosine_similarity(output_a, output_b)
    original_loss = torch.abs(original_similarity - transformed_similarity).mean()

    mean_penalty = mean_penalty = row_col_mean_penalty(output_a) + row_col_mean_penalty(output_b)
    
    range_penalty_a = abs_value_penalty(output_a)
    range_penalty_b = abs_value_penalty(output_b)

    total_loss = original_loss + lambda1 * mean_penalty + lambda2*(range_penalty_a+range_penalty_b)
    return total_loss

def cosine_similarity_matrix(batch):
    norm = torch.norm(batch, dim=1).view(-1, 1)
    normed_batch = batch / norm
    similarity = torch.mm(normed_batch, normed_batch.t())
    return similarity

def get_median_value_of_similarity(all_token_embedding):
    similarity = cosine_similarity_matrix(all_token_embedding)
    median_value = torch.median(similarity)
    print(median_value)
    return median_value

class VectorDataset(Dataset):
    def __init__(self, vectors):
        self.vectors = vectors
    
    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, idx):
        return self.vectors[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect watermark in texts")
    parser.add_argument("--input_path", type=str, default="embeddings/gpt2_sentence_embedding_bert.txt")
    parser.add_argument("--output_model", type=str, default="transform_model5.pth")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--input_dim", type=int, default=1)

    args = parser.parse_args()
    embedding_data = np.loadtxt(args.input_path)
    data = torch.tensor(embedding_data, device='cuda', dtype=torch.float32)
    median_value = get_median_value_of_similarity(data)
    dataset = VectorDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  

    model = TransformModel(input_dim=args.input_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.2)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    epochs = args.epochs
    for epoch in range(epochs):
        batch_iterator = iter(dataloader)
        
        for _ in range(len(dataloader) // 2):  
            input_a = next(batch_iterator).to(device)  
            input_b = next(batch_iterator).to(device)  
            if input_a.shape[0] != input_b.shape[0]:
                continue
            output_a = model(input_a)
            output_b = model(input_b)
            loss = loss_fn(output_a, output_b, input_a, input_b, median_value=median_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if _ %100 ==0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{_ + 1}/{len(dataloader) // 2}], Loss: {loss.item()}")

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(model.state_dict(), args.output_model)
