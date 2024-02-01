import tiktoken
from load_data import chars


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