"""

"""

import torch
from pathlib import Path
from gpt import GPTLanguageModel
from load_data import get_random_batch
from gpt_settings import *

torch.cuda.empty_cache()

@torch.no_grad()
def _estimate_batch_loss(model, train_data, val_data, batch_size, block_size):
    out = {}

    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_random_batch(split, train_data, val_data, batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()      # average losses over eval_iters
    model.train()

    return out


######## Load training text
text = ""
for file_path in Path('./data/hedh').glob('*.txt'):
    with file_path.open('r', encoding='utf-8') as f:
        text += f.read() + "\n"
print("length of dataset in characters: ", len(text))

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
print(''.join(chars))

vocab_size = len(chars)
print(vocab_size)

##### ENCODING
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode_fn = lambda s: [stoi[c] for c in s]             # encoder: take a string, output a list of integers
decode_fn = lambda l: ''.join([itos[i] for i in l])


####  text -> Tensor
# TBD: add another tokenizer
data_tensor = torch.tensor(encode_fn(text), dtype=torch.long)
print(data_tensor.shape, data_tensor.dtype)


### train / validation
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]


#### Load gpt model
model = GPTLanguageModel(vocab_size)
print(sum(p.numel() for p in model.parameters())/1e6, 'Million parameters')

# Print the first embeddings for the first 2 tokens
first_embeddings = model.token_embedding_table.weight.data[:1]  # Adjust 'embedding' if the name is different
#print('First embeddings for the first 1 token:')
#print(first_embeddings)

# or you can use this one if you have cuda device
m = model.to(device)


#### TRAIN
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # sample a batch of data
    xb, yb = get_random_batch('train', train_data, val_data, batch_size, block_size)

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()         # compute gradients
    optimizer.step()        # update parameters

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = _estimate_batch_loss(m, train_data, val_data, batch_size, block_size)
        print(f"step {iter} finished: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# save the model
model_save_path = 'gpt2_hedh_model_10000.pth'  # Specify your desired file name and path
torch.save(m.state_dict(), model_save_path)


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode_fn(m.generate(context, max_new_tokens=600)[0].tolist()))
