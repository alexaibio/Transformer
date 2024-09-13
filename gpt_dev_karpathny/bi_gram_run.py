import torch
from pathlib import Path
from bi_gram import BigramLanguageModel
from load_data import get_batch
#from _encode_decode import encode_fn, decode_fn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)


### get text, encode it, convert to tensor -> Tensor
######## Load training text
#with open('./data/input.txt', 'r', encoding='utf-8') as f:
#    text = f.read()
text = ""
for file_path in Path('./data/hedh').glob('*.txt'):
    with file_path.open('r', encoding='utf-8') as f:
        text += f.read() + "\n"

print("length of dataset in characters: ", len(text))

# print all unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode_fn = lambda s: [stoi[c] for c in s]             # encoder: take a string, output a list of integers
decode_fn = lambda l: ''.join([itos[i] for i in l])

# encode data and place it into a tensor
data_tensor = torch.tensor(data=encode_fn(text), dtype=torch.long)
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
[18, 47, 56, 57, 58,  1, 15, 47, 58]
18->47
18, 47 -> 56

that is why in all-you-need they have target shifted right, because every next 
character is a target by itself
Q: but here we have only BiGram model - why here is long context? should be only two
'''

# print out all examples from one block
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]  # y is already shifted
    print(f"when input is {context} the target: {target}")

###### test batch: create input tensor with batches
batch_size = 4  # B (batch) -  how many independent sequences will we process in parallel?
block_size = 8  # T (time) -  what is the maximum context length for predictions, block_size
# C (channel) - embedding dimension, missing here yet

# xb = [4,8], yb = [4,8]-shifted,  [batch, block]
# get 4 batches of 8 size blocks at random point in train dara
xb, yb = get_batch(split='train', train_data=train_data, val_data=val_data, batch_size=batch_size, block_size=block_size)
# NOTE: one context/target rows include 8 training examples, so in this 4x8 tensor we have 32 examples



##########################################################
##### BiGram Language Model

### CHECK: generate text with untrained model with random weights - generates garbage
model = BigramLanguageModel(vocab_size)
m = model.to(device)
logits, loss = m(xb, yb)    # prediction for entire batch:  logits (32,65), loss=single number

# idx is Batch x Time, 4 x 8, a starting character to generate
generated_text = m.generate(
    idx=torch.zeros((1, 1), dtype=torch.long, device=device),  # starting context, just 1 in our case
    max_new_tokens=50
)
generated_text_lst = generated_text[0].tolist()
print(f'\n GENERATED Untrained TEXT: {decode_fn(generated_text_lst)}')


### TRAIN: train this model on data
m.train(train_data, val_data, batch_size, block_size)
generated_text = m.generate(
    idx=torch.zeros((1, 1), dtype=torch.long, device=device),  # starting context, just 1 in our case
    max_new_tokens=100
)
generated_text_lst = generated_text[0].tolist()
print(f'\n GENERATED trained TEXT: \n {decode_fn(generated_text_lst)}')

print()

# NOTE: we dont use here multiple conext, only one symbol to one symbol

