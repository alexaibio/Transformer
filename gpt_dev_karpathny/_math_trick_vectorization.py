import torch
import torch.nn as nn
from torch.nn import functional as F


########### EXAMPLE how to communicate with only past
# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))    # lower triangular with 1
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, size=(3,2)).float()
c = a @ b

# every nex is an average of previous
# из-за первой треугольной матрицы результирущая матрица С содержит обработку второй матрицы
# - в первом ряду учитываем тоько первые сивмолы
# - во втором первые два
# - в третьем все три
# то есть постепенно продвигаемся вперед - векторизация
print(f'a=\n {a}')
print(f'b=\n {b}')
print(f'c=\n {c}')

############## More complicated example
# consider the following toy example:

torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)

# for every single batch we want
# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)

# you can see that every next row of xbow is aggregation of previous roes in x
print(f' First batch of an initial tensor x[b=0, t, c] \n {x[0]}')
print(f' All batches with past-aggregation xbow[b=0, t, c] \n {xbow[0]}')

############## LETS vectorize it now
# using matrix multiply for a weighted aggregation
# wei - weights
wei = torch.tril(torch.ones(T, T))  # wei for every batch, then broadcasted
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x     # (Broadcast B, T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)

####### ADD SOFTMAX - because
# version 3: use Softmax - it produces the same wei matrix
# reason: лежит в интерпретации весов - в аттеншен это affinity, i.e 0 afinity means token dont communicate
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)    # exp and divide by sum
xbow3 = wei @ x
torch.allclose(xbow, xbow3)


