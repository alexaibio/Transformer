import torch
import torch.nn as nn
from torch.nn import functional as F


#### Now - SELF-ATTENTION - DECODER
'''
# version 4: self-attention!
in previous example we just averaged past token's embeddings
here instead of assigning uniform weigths to each past token, we do it as follow
- each token emits Query (what am I looking for?) and Key(what do I contain?) vectors
- affinity = dot(Query_of_token_1 @ Key_of_token_2)
- that dot product becomes wei
'''

torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16  # what is intuition on that?
key = nn.Linear(C, head_size, bias=False)   # (C, 16) - in fact this is multiple dot-products
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
# for every batch B we have a (T,T) wei matrix of affinities
wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
# so now initial value of wei is not zero - but calculated affinities
#wei = torch.zeros((T,T))

# then - standard way to apply past information
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf')) # comment this line for ENCODER
wei = F.softmax(wei, dim=-1)

# instead of simple
#out = wei @ x
# we add scaling coefficient
# v also add 16 dimensions, number of heads
v = value(x)
out = wei @ v

'''
NOTE
- this is decoder block, becasue it masked previous information
- decoder block is used for generation, that is why it dont have to know previous information
- ENcoder block is the same with exception of it has no masking - all info is available
'''



