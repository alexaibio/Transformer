import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


######## Load training text
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
print(''.join(chars))

vocab_size = len(chars)
print(vocab_size)


def get_batch(split, train_data, val_data, batch_size, block_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data

    # generate 4 starting indexes to take a block of size 8
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # take those 4 batches x block size, then convert list of tensors into one tensor
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y