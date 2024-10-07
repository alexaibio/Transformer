"""
Explains how to mask attention, i.e. only take care about the past
and disregard the future tokens
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(42)


########### simple EXAMPLE how to mask future token with matrix operation
# build a special triangular matrix (weights matrix)
a = torch.tril(torch.ones(3, 3))    # lower triangular with 1
a = a / torch.sum(a, 1, keepdim=True)

# here is our mtrix to process
b = torch.randint(0, 10, size=(3,2)).float()

# here we process only ast from matrix b
c = a @ b

print(f'a=\n {a}')
print(f'b=\n {b}')
print(f'c=\n {c}')



torch.manual_seed(1337)
############## Example for three dimensions: B, T, C
# how to average only the past preserving batches B

B,T,C = 4,8,2       # batch(number of examples), time (lest sequence), channels (embeddings)
x = torch.randn(B,T,C)

# for every single batch we want  x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]   # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)

# you can see that every next row of xbow is aggregation of previous rows in x


############## Same but vectorized

# our specialized matrix - wei - weights
wei = torch.tril(torch.ones(T, T))  # wei for every batch, then broadcasted
wei = wei / wei.sum(1, keepdim=True)
# NOTE dimensions!: (Broadcast B, T, T) @ (B, T, C) ----> (B, T, C)
xbow2 = wei @ x

torch.allclose(xbow, xbow2) # check for element-wise equal within a certain tolerance


####### same with SOFTMAX
# reason:
# because in elf attention we set initial wei not in zero but calculate then as similarities
# and then by masking them we say that future should not be taken into account
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)    # exp and divide by sum

xbow3 = wei @ x

torch.allclose(xbow, xbow3)


