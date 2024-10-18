from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoTokenizer
from huggingface_hub import login
from multiprocessing import cpu_count
from settings import get_project_root


def _apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example


def _load_raw_ultrachat_dataset() -> DatasetDict:
    # Enter your Hugging Face token when prompted
    login()

    ####### human DATASET
    raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k")

    # ---- remove this when done debugging: limit data to first 100 items
    indices = range(0,100)
    dataset_dict = {
        "train": raw_datasets["train_sft"].select(indices),
        "test" : raw_datasets["test_sft"].select(indices)
    }
    raw_datasets = DatasetDict(dataset_dict)
    # ------------------------------------

    '''
    example = raw_datasets["train"][0]
    messages = example["messages"]
    for message in messages:
      role = message["role"]
      content = message["content"]
      print('{0:20}:  {1}'.format(role, content))
    '''
    return raw_datasets


######## Load tokenizer
def load_model_tokenizer(model_id):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # LlamaTokenizerFast is a BPE-based tokenizer
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")


    # set PADDING for whole context length
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048       # set to maximum context length ( T )

    return tokenizer


#### apply Jinja2 template to convert dictionary to string

def build_raw_sft_dataset_txt(model_id) -> DatasetDict:
    """
    RETURN: a text not tokenized train and test data
    """
    raw_datasets = _load_raw_ultrachat_dataset()
    column_names = list(raw_datasets["train"].features)

    tokenizer = load_model_tokenizer(model_id)

    tokenizer.chat_template = """
    {% for message in messages %}
        {% if message['role'] == 'user' %}
            {{ '<|user|>\n' + message['content'] + eos_token }}
        {% elif message['role'] == 'system' %}
            {{ '<|system|>\n' + message['content'] + eos_token }}
        {% elif message['role'] == 'assistant' %}
            {{ '<|assistant|>\n' + message['content'] + eos_token }}
        {% endif %}
    
        {% if loop.last and add_generation_prompt %}
            {{ '<|assistant|>' }}
        {% endif %}
    {% endfor %}
    """


    # run templating on multiple cores - convert to a plain text
    raw_datasets = raw_datasets.map(
        _apply_chat_template,
        num_proc=cpu_count(),
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=column_names,        # which columns to remove from the dataset after applying the function.
        desc="Applying chat template",
    )
    print(f"------------> one training example: \n {raw_datasets['train'][0]}")
    return raw_datasets



from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


def _load_text_corpus_from_folder():
    text_data = ""
    directory_path = get_project_root() / 'data/hedh/'

    # Iterate over all text files in the directory
    for file_path in directory_path.glob("*.txt"):
        with file_path.open(encoding="utf-8") as file:
            text_data += file.read() + "\n"

    return text_data


def _split_text_into_chunks(txt_data, chunk_size, overlap_size, eos_token_id):
    """Splits  text into chunks with overlap and adds an EOS token to each chunk."""
    chunks = []
    #total_length = tokenized_data.size(1)  # The length of the sequence (number of tokens)
    total_length = len(txt_data)

    start = 0
    while start < total_length:
        # Define the end of the chunk
        end = min(start + chunk_size, total_length)

        # Extract the chunk and add the EOS token at the end
        chunk = txt_data[start:end]
        chunk += eos_token_id

        # Add the chunk to the list of chunks
        chunks.append(chunk)

        # Move the start of the next chunk with overlap
        start += chunk_size - overlap_size

    return chunks


def build_raw_domain_adaptation_dataset(model_id):
    """Loads text files from a folder, splits them into chunks, and returns a train/test dataset dictionary."""
    chunk_size = 2000
    overlap_size = 100

    test_size = 0.2
    random_state = 42

    tokenizer = load_model_tokenizer(model_id)

    # Load all text files from the specified directory
    text_data = _load_text_corpus_from_folder()



    # Tokenize the text data
    #tokenized_data = tokenizer(text_data, return_tensors="pt", truncation=False)["input_ids"]

    # Split the tokenized data into chunks with overlap, and add eos token at the end of each chunk
    chunked_texts = _split_text_into_chunks(text_data, chunk_size, overlap_size, tokenizer.eos_token)



    # Split the data into training and test sets (80% train, 20% test by default)
    train_texts, test_texts = train_test_split(chunked_texts, test_size=test_size, random_state=random_state)

    # Create Dataset objects from the text chunks
    train_dataset = Dataset.from_dict({"text": train_texts})
    test_dataset = Dataset.from_dict({"text": test_texts})

    # Return a dataset dictionary with 'train' and 'test' keys
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    print(f" One example of train data: \n {dataset_dict['train'][0]}")

    return dataset_dict



if __name__ == '__main__':
    from settings import model_id
    #sfo_ds = build_raw_sft_dataset(model_id)
    #print()

    da_ds = build_raw_domain_adaptation_dataset(model_id)
    print()