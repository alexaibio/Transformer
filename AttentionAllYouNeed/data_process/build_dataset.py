from pathlib import Path
from torch.utils.data import DataLoader, random_split
from AttentionAllYouNeed.data_process.dataset import BilingualDataset

# Huggingface datasets and tokenizers
from datasets import load_dataset, load_dataset_builder
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def _get_all_sentences(ds, lang):
    '''
    Get all sentences from out dataset
    We need generator here, because...
    '''
    for item in ds:
        yield item['translation'][lang]


def _get_or_build_tokenizer(config, ds, lang):

    # save tokenizer to this file
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        # we are using the simpliest tokenizer which split sentence by words
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # if tokenizer see a word not in vocabulary it substitutes it with UNK
        tokenizer.pre_tokenizer = Whitespace()   # split sentence by white space

        # [SOS] - start of sentence / [EOS] - end of sentence
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(_get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    # explore dataset
    ds_builder = load_dataset_builder(
        path=f"{config['datasource']}",
        name=f"{config['lang_src']}-{config['lang_tgt']}"
    )
    print("--------------- using datasets from Hugging Face:")
    print(ds_builder.info.description)

    # huggingface Datasets is a library for easily accessing and sharing datasets
    ds_raw = load_dataset(
        path=f"{config['datasource']}", # opus_books
        name=f"{config['lang_src']}-{config['lang_tgt']}",  # en-ru
        split='train'
    )

    # Build tokenizers
    tokenizer_src = _get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = _get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # out of HGF DataSets create a complete input/outpu tensors and a label tensor with masks
    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src, tokenizer_tgt,
        config['lang_src'], config['lang_tgt'], config['seq_len']
    )

    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src, tokenizer_tgt,
        config['lang_src'], config['lang_tgt'], config['seq_len']
    )

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

