import torch
import torch.nn as nn
from torch.nn import functional as F
from load_data import get_batch
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # embedding layer returns embeddings vectors by indexes
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)   # use embedding lenght= num embeding - why?!

    def forward(self, idx, targets=None):
        '''
        Prediction of next character based on previous one
        # idx and targets are both (B=4, T=8), but shifted
        # in each batch B we have T=8 indexes of words and get embeddings for it
        # each index/word now is embedding - that is why we have one more dimension - C-channel
        '''
        # 1 - prediction (forward pass)
        # just extract weight which are embeddings in this case,
        # idx is (B, T) so we add embedding dimension C
        logits = self.token_embedding_table(idx)    # (B,T,C) # raw score of prob for next character

        # 2 - loss calculation
        if targets is None:
            # TODO: note that logit formal is 3dim in that case not 2-dim as in else clause
            loss = None
        else:
            # combine all 4 batches into one line, because loss requires one-dim vector
            B, T, C = logits.shape
            logits = logits.view(B*T, C)    # 32x65: 32 example with prob of 65 tokens
            targets = targets.view(B*T)     # 32 targets
            # logits are embeddings here, targets are prob of tokens,
            # cross_entropy - multi-class classification scenario.
            loss = F.cross_entropy(logits, targets)     # single number

        return logits, loss # logits are learned embeddings (weights)

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context

        # repeat 50 times
        for _ in range(max_new_tokens):
            # get the predictions: B, T, C
            logits, loss = self(idx)    # one forward pass (get embeddings + loss calculation)

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # calculate next character: sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)     # (B, T+1)
        return idx

    def train(self, train_data, val_data, batch_size, block_size):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)

        for steps in range(250000):  # increase number of steps for good results...
            # sample a batch of data
            xb, yb = get_batch('train', train_data, val_data, batch_size=batch_size, block_size=block_size)

            # evaluate the loss
            logits, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if steps % 50000 == 0:
                print(f'   - Step {steps}, Loss: {loss.item()}')

        print(f' final loss= {loss.item()}')


if __name__ == "__main__":
    vocab_size = 65
    m = BigramLanguageModel(vocab_size)
