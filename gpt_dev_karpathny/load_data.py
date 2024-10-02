import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def get_random_batch(split, train_data, val_data, batch_size, block_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data

    # generate random 64 (block size) starting indexes to take a block of size batch_size=8
    ix = torch.randint(high=(len(data) - block_size), size=(batch_size,))

    # take those 4 batches x block size, then convert list of tensors into one tensor
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


if __name__ == '__main__':
    x, y = get_random_batch()