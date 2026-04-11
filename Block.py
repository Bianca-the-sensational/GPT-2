import requests
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from Preprocessing import dropout , GetEmbedding , batch_size , vocab_size , n_embed , block_size , num_heads , head_size , device , n_blocks

# LAYER NORMALISATION IMPLEMENTATION :
class LayerNorm(nn.Module):

    def __init__(self , features , eps = 1e-6):
        super().__init__()
        # nn.Parameter -> to let gamma , beta be treated as features
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self , x : torch.Tensor):
        mean = x.mean(-1 , keepdim = True)
        std = x.std(-1 , keepdim = True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class MultiHeadAttention(nn.Module):

    def __init__(self, n_embed, num_heads, head_size, block_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        # Fused Q, K, V projection for all heads in one linear layer -> converts to the Q , K , V (all combined) for all the heads combined 
        self.qkv_projection = nn.Linear(n_embed, 3 * n_embed, bias=False) # (B , T , 3 * n_embed)
        self.output_projection = nn.Linear(n_embed, n_embed)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # C = n_embed # (B , T , 3 * n_embed)
        q, k, v = self.qkv_projection(x).split(C, dim=2) # splits into the Q , K, V vectors for all heads (B , T , n_embed)

        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # isolates the query vector for each head (B , num_heads , T , head_size)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # isolates the key vector for each head (B , num_heads , T , head_size)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # isolate the value vector for each head (B , num_heads , T , head_size)

        dot = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)

        dot = dot.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        dot = torch.softmax(dot, dim=-1)  # (B, num_heads, T, T)
        dot = self.dropout(dot)
        out = dot @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C) # concatenating all the heads output to form the original dimension (B , T, n_embed)
        
        out = self.output_projection(out)  # (B, T, n_embed)

        return out
    
# IMPLEMENT THE FEED FORWARD NETWORK
class MLP(nn.Module):

    def __init__(self , n_embed):
        super().__init__()
        self.l1 = nn.Linear(n_embed , 4 * n_embed)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(4 * n_embed , n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self , idx):
        x1 = self.dropout(self.relu(self.l1(idx)))
        x3 = self.l3(x1)

        return x3
    
# IMPLEMENTING A BLOCK (ATTENTION(MHA) + MLP)
class Block(nn.Module):

    def __init__(self , num_heads , head_size , n_embed , block_size):
        super().__init__()
        self.MHA = MultiHeadAttention(n_embed , num_heads , head_size , block_size)
        self.FFN = MLP(n_embed)
        self.ln1 = LayerNorm(n_embed)
        self.ln2 = LayerNorm(n_embed)
        self.drop = nn.Dropout(dropout)

    def forward(self , idx):
        # residual connections (adding the outputs of the MHA network to the initial idx)
        x1 = idx + self.drop(self.MHA(self.ln1(idx)))
        x2 = x1 + self.drop(self.FFN(self.ln2(x1)))

        return x2
    
# IMPLEMENTING THE FINAL GPT MODEL

class GPT(nn.Module):

    def __init__(self , vocab_size , n_embed , block_size , num_heads  , head_size):
        super().__init__()
        self.emb = GetEmbedding(vocab_size , n_embed , block_size)
        self.blocks = nn.Sequential(*[Block(num_heads  , head_size , n_embed , block_size) for _ in range(n_blocks)])
        self.layer = nn.Linear(n_embed , vocab_size)
        self.ln = LayerNorm(n_embed)

        self.apply(self._init_weights)

    def _init_weights(self , module):
        # to initialise weights (for the linear layer and the lookup table) such that the mean is 0.0 and std is 0.02
        if isinstance(module , nn.Linear):
            torch.nn.init.normal_(module.weight , mean = 0.0 , std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module , nn.Embedding):
            torch.nn.init.normal_(module.weight , mean = 0.0 , std = 0.02)

    # function to provide the final layer of neurons (for prediciting the final token) and predict the loss for training data
    def forward(self , idx , targets = None):
        embed = self.emb(idx) # (B , T , n_embed)
        x1 = self.blocks(embed) # residual connections already present in the BLOCK class (B , T, N_embed)
        x1 = self.ln(x1)
        x2 = self.layer(x1) # (B , T , vocab_size)

        if targets is None:
            loss = None
        else:
            B , T , C = x2.shape
            x2 = x2.view(B*T , C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(x2 , targets)

        return x2 , loss

    def generate(self , idx , num_new_tokens , temperature = 0.8 , top_k = 40):
        for _ in range(num_new_tokens):
            idx1 = idx[: , -block_size:]
            logits , loss = self(idx1)
            logits = logits[: , -1 , :]
            logits = logits / temperature

            if top_k is not None :
                top_k_values , _ = torch.topk(logits , top_k)
                min_top_k = top_k_values[: , -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_top_k , float('-inf'))

            prob = F.softmax(logits , dim = -1)
            pred = torch.multinomial(prob , num_samples = 1)
            idx = torch.cat((idx , pred) , dim = 1)

        return idx
    
model = GPT(vocab_size , n_embed , block_size , num_heads , head_size).to(device)


