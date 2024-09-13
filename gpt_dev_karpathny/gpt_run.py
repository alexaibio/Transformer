import torch
from gpt import GPTLanguageModel
from load_data import get_batch
from _encode_decode import decode_fn, encode_fn


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


######## Load training text
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
print(''.join(chars))

vocab_size = len(chars)
print(vocab_size)


###### hyperparameters
batch_size = 64     # how many independent sequences will we process in parallel?
block_size = 256    # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

## text -> Tensor
data_tensor = torch.tensor(encode_fn(text), dtype=torch.long)
print(data_tensor.shape, data_tensor.dtype)


### train / validation
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]


#### Load gpt model
model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


#### TRAIN
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train', train_data, val_data, batch_size, block_size)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()     # compute gradients
    optimizer.step()    # update parameters

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode_fn(m.generate(context, max_new_tokens=500)[0].tolist()))
