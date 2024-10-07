"""
        Decoder Only transformer
- Decoder does a pure language modeling
- if we need a translation task we need to condition generation on input phrase (add an encoder and cross attend it)
- in self attention we masked future content, that makes it self attention, not cross attention

"""
import torch.nn as nn
from torch.nn import functional as F
from gpt_settings import *
torch.manual_seed(1337)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        # NOTE: head size is a length of context: sentence size (?)
        # he input embedding dimension is split into multiple smaller dimensions, one for each head
        # Each head operates on its own subspace dimension (head_size)
        super().__init__()

        # add weights to each embedding input feature
        # W has shape (head_size, n_embd) = (output_features, input_features)
        # note: head_size split long embeddings into chunks for faster computation
        self.key   = nn.Linear(in_features=n_embd, out_features=head_size, bias=False)
        self.query = nn.Linear(in_features=n_embd, out_features=head_size, bias=False)
        self.value = nn.Linear(in_features=n_embd, out_features=head_size, bias=False)

        # Buffers: self.tril
        # used for tensors that do not change during training, like masks,
        # moved automatically to the same device as the model, making them convenient to use in GPU/CPU
        self.register_buffer(
            name='tril',
            tensor=torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        # key() and quiery() are linear layers, they perform C->head_size
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)

        # initial attention scores with similarity matrix and dimension normalization
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5    # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        # masking the future: place -Inf in TRUE in masks
        wei = wei.masked_fill(
            mask=self.tril[:T, :T] == 0,
            value=float('-inf')
        )    # (B, T, T)

        wei = F.softmax(wei, dim=-1)    # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values (multiply by V)
        v = self.value(x)   # (B,T,hs)
        out = wei @ v       # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # NOTE: projection is a linear layer without non-linear activation
        self.proj = nn.Linear(in_features=head_size * num_heads, out_features=n_embd)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # should it be a n_head instead of 4?
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),      # no RELU here because it is a projection
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)     # Layer Normalization (trainable)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))        # NOTE: we have a skip connection and layer normalization
        x = x + self.ffwd(self.ln2(x))      # this is to avoid vanishing gradients
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)       # ( tokens, C)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)    # (T, C)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)    # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better weigth init
        self.apply(self._init_weights)  # apply to all layers above

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)   # idx(B,T,C), lookup for embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb   # (B,T,C)
        x = self.blocks(x)      # (B,T,C)
        x = self.ln_f(x)        # (B,T,C)
        logits = self.lm_head(x)    # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]           # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)   # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)     # (B, T+1)
        return idx

