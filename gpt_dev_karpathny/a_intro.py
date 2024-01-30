import tiktoken
import torch
from bi_gram import BigramLanguageModel, get_batch

######## Load training text
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
print(''.join(chars))
vocab_size = len(chars)
print(vocab_size)

####### simple tokenization

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]             # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])    # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

## (BPE is SOTA - byte-pair encoding)
# tiktoken is best implementation - OpenAI
enc = tiktoken.get_encoding('gpt2')
print(enc.n_vocab)
print(enc.encode('Hi there'))
print(enc.decode([17250, 612]))


### text -> Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)


### train / validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


### split text into chunks, train transformer on those chunks in random order
#  BLOCK SIZE - a size of that chunk
block_size = 8
print(train_data[:block_size+1])    # [18, 47, 56, 57, 58,  1, 15, 47, 58]

# create examples context->target
'''
NOTE
when we have a block of 8 tokens we have MULTIPLE EXAMPLES built into it
18->47
18, 47 -> 56
etc
that is why in all-you-need they have target shifted right, because every next chacted is 
a target by itself
'''

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]  # y is already shifted
    print(f"when input is {context} the target: {target}")

###### create input tensor with batches
torch.manual_seed(1337)
batch_size = 4  # B (batch) -  how many independent sequences will we process in parallel?
block_size = 8  # T (time) -  what is the maximum context length for predictions?
# C (channel) - embedding dimension

xb, yb = get_batch(split='train', train_data=train_data, val_data=val_data, batch_size=batch_size, block_size=block_size)
# NOTE: one context/target rows include 8 training examples, so in this 4x8 tensor we have 32 examples

##########################################################
##### BiGram Language Model

# generate with untrained model with random weights - generates garbage
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)    # logits (32,65), loss=single number

# idx is Batch x Time, 4 x 8, a starting character to generate
generated_text = m.generate(
    idx=torch.zeros((1, 1), dtype=torch.long),  # starting context, just 1 in our case
    max_new_tokens=50
)
generated_text_lst = generated_text[0].tolist()
print(f'\n GENERATED Untrained TEXT: {decode(generated_text_lst)}')

# train this model on data
m.train(train_data, val_data, batch_size, block_size)
generated_text = m.generate(
    idx=torch.zeros((1, 1), dtype=torch.long),  # starting context, just 1 in our case
    max_new_tokens=100
)
generated_text_lst = generated_text[0].tolist()
print(f'\n GENERATED trained TEXT: {decode(generated_text_lst)}')


