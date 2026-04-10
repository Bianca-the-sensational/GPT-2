import requests
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from BPETokeniser import train_data , val_data , test_data , vocab_size

# DATALOADER
batch_size = 32
block_size = 128
#num_batches = 200
#dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split , batch_size , block_size):
    if (split == "train"):
        data = train_data
    elif (split == "val"):
        data = val_data
    else:
        data = test_data

    if not isinstance(data , torch.Tensor):
        data = torch.tensor(data , dtype = torch.long)

    ix = torch.randint(len(data) - block_size , (batch_size , ))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i+1 : i + block_size + 1 ]for i in ix])
    x  , y = x.to(device) , y.to(device)
    return x , y

 # TRANFORMER ARCHITECTURE

#n_heads = 4
n_embed = 256
n_blocks = 6

import math
import random
import time

random.seed(42)
print ("EMBEDDINGS GETTING GENERATED........")
start_time = time.time()
class GetEmbedding(nn.Module):

    def __init__(self, vocab_size, n_embed, block_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embed)

        pe = torch.zeros((block_size, n_embed) , device = device)
        position = torch.arange(0, block_size, dtype=torch.float , device = device).unsqueeze(1) # (block_size, 1)

        div_term = torch.exp(torch.arange(0, n_embed, 2).float().to(device) * (-math.log(10000.0) / n_embed))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, idx):
        B, T = idx.shape

        x = self.embedding(idx) # (B, T, n_embed)

        return x + self.pe[:, :T, :] # (B , T , n_embed) -> pe is same across all the batches

idx , y = get_batch("train" , batch_size , block_size) # batch = 64 , block = 128
embedding_model = GetEmbedding(vocab_size , n_embed , block_size).to(device)
embedding_layer = embedding_model(idx)
print(embedding_layer.shape)

head_size = 64
num_heads = 4
dropout = 0.3

def save_checkpoint(state, filename="best_model.pth"):
    print(f"--> Saving checkpoint to {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimiser):
    print(f"--> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser'])
    return checkpoint['iter'], checkpoint['best_loss']
