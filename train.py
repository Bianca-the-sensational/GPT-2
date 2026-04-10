import requests
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from Block import model
from Preprocessing import get_batch
from Preprocessing import dropout , GetEmbedding , batch_size , vocab_size , n_embed , block_size , num_heads , head_size , device , n_blocks
from Preprocessing import load_checkpoint , save_checkpoint 
from BPETokeniser import tokeniser , test_data

eval_iters = 200
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train' , 'val']:
        losses = torch.zeros(eval_iters)
        for i in range (eval_iters):
            X , Y = get_batch(split , batch_size , block_size)
            logits , loss = model(X , Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# TRAINING LOOP
lr = 3e-4
optimiser = torch.optim.AdamW(model.parameters() , lr = lr , weight_decay = 0.1)
num_iters = 15000
eval_interval = 500
best_loss = float('inf')
start_iter = 0

CHECKPOINT_FILE = "best_model.pth"
if os.path.exists(CHECKPOINT_FILE):
    start_iter, best_loss = load_checkpoint(CHECKPOINT_FILE, model, optimiser)
    start_iter += 1 

for i in range(start_iter, num_iters):
    if i % eval_interval == 0 or i == num_iters - 1:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if losses['val'] < best_loss:
            best_loss = losses['val']
            checkpoint = {
                'iter': i,
                'state_dict': model.state_dict(),
                'optimiser': optimiser.state_dict(),
                'best_loss': best_loss,
            }
            save_checkpoint(checkpoint, filename=CHECKPOINT_FILE)

    idx, y = get_batch("train", batch_size, block_size)
    logits, loss = model.forward(idx, y)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print ("-------------------------TEXT GENERATED WITHOUT CONTEXT---------------------------------")
generated_indices = model.generate(context, num_new_tokens=1000)
print(tokeniser.decode(generated_indices[0].tolist()))

input_data = torch.tensor(test_data[:200], dtype=torch.long).unsqueeze(0).to(device)
print ("--------------------------TEXT GENERATED WITH PROVIDED CONTEXT--------------------------------")
generated_indices = model.generate(input_data, num_new_tokens=1000)

print(tokeniser.decode(generated_indices[0].tolist()))

# to use the final model saved with the best val loss
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval() 

user_input = input(print("Enter prompt : "))
context = torch.tensor(tokeniser.encode(user_input), dtype=torch.long, device=device).unsqueeze(0)

generated_indices = model.generate(context, num_new_tokens=500)
print(tokeniser.decode(generated_indices[0].tolist()))
