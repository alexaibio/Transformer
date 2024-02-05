import torch
from bi_gram import BigramLanguageModel
from load_data import chars, text, vocab_size, get_batch
from _encode_decode import encode, decode
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)


### get text, encode it, convert to tensor -> Tensor
data_tensor = torch.tensor(encode(text), dtype=torch.long)
print(data_tensor.shape, data_tensor.dtype)

### split into train / validation
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]


### split text into chunks, train transformer on those chunks in random order
#  BLOCK SIZE - a size of that chunk
block_size = 8  # block_size - length of context in symbols, T dimension
print(train_data[:block_size+1])    # [18, 47, 56, 57, 58,  1, 15, 47, 58]

#### create examples context->target
'''
NOTE
when we have a block of 8 tokens we have MULTIPLE EXAMPLES built into it
18->47
18, 47 -> 56
etc
that is why in all-you-need they have target shifted right, because every next chacted is 
a target by itself
Q: but here we have only BuGram model - why here is long context?
'''
# print out all exmaples from one block
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]  # y is already shifted
    print(f"when input is {context} the target: {target}")

###### create input tensor with batches
batch_size = 4  # B (batch) -  how many independent sequences will we process in parallel?
block_size = 8  # T (time) -  what is the maximum context length for predictions, block_size
# C (channel) - embedding dimension

# xb = [4,8]
xb, yb = get_batch(split='train', train_data=train_data, val_data=val_data, batch_size=batch_size, block_size=block_size)
# NOTE: one context/target rows include 8 training examples, so in this 4x8 tensor we have 32 examples

##########################################################
##### BiGram Language Model
# generate text with untrained model with random weights - generates garbage
model = BigramLanguageModel(vocab_size)
m = model.to(device)
logits, loss = m(xb, yb)    # logits (32,65), loss=single number

# idx is Batch x Time, 4 x 8, a starting character to generate
generated_text = m.generate(
    idx=torch.zeros((1, 1), dtype=torch.long, device=device),  # starting context, just 1 in our case
    max_new_tokens=50
)
generated_text_lst = generated_text[0].tolist()
print(f'\n GENERATED Untrained TEXT: {decode(generated_text_lst)}')

# train this model on data
m.train(train_data, val_data, batch_size, block_size)
generated_text = m.generate(
    idx=torch.zeros((1, 1), dtype=torch.long, device=device),  # starting context, just 1 in our case
    max_new_tokens=100
)
generated_text_lst = generated_text[0].tolist()
print(f'\n GENERATED trained TEXT: {decode(generated_text_lst)}')


