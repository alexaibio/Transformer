import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

######  architecture parameters
block_size = 512        #256 the maximum context length
n_embd = 384
n_head = 6              # e_embd must be divided by n_head without residial
n_layer = 7             #6
DROPOUT = 0.2


###### training hyperparameters
batch_size = 64     #64 how many independent sequences will we process in parallel?

max_iters = 10000
learning_rate = 2e-4

eval_interval = 500
eval_iters = 200

# val loss 1.4