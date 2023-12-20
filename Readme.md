# Simple transformer replication of 'Attention is all you need' article
Based on 2-language data train a foundational transformer on generative settings

Training corpus: a numer of 2-language sentences from opus-books

## inspired by
https://github.com/hkproj/pytorch-transformer


# CUDA installation


# TODO
- replicate with FastTransformers
- replicate with FlashAttention

# Train pipeline
- get_ds(config) 
  - ds_raw = load_dataset (from haggingface.datasets) - ('opus_books', 'en-ru'), 17496x2
  - tokenizer_src = get_or_build_tokenizer, create tokenizer_en/ru.json
  - BilingualDataset
  - tokenizer_src.encode
  - return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
- dfdfd