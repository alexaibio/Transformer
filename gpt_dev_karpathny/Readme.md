# This is nanoGPT created from scratch by Karpathy video
- https://www.youtube.com/watch?v=kCc8FmEb1nY
- https://github.com/karpathy/ng-video-lecture

# Abbreviations
- B T C : batch, time, channel
- Batch - number of parallel examples to process, typically 32 - 64
- block_size: T dimension, how many letters we analyse, the comtext size
- Channel: embedding size, number of compressed features
- head_size: dimensionality of K, V vectors within each attention heads
- number of heads: how many parallel attention layers or "heads" the self-attention mechanism has
- 

# How to run
- bi_gram_run.py: run simple bi-gram language model
- gtp_run: run simple GPT2 model
