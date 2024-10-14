from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoTokenizer
from huggingface_hub import login
from multiprocessing import cpu_count


def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example


def load_raw_ultrachat_dataset():
    # Enter your Hugging Face token when prompted
    login()

    ####### human DATASET
    raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k")

    # ---- remove this when done debugging: limit data to first 100 items
    indices = range(0,100)
    dataset_dict = {"train": raw_datasets["train_sft"].select(indices),
                    "test": raw_datasets["test_sft"].select(indices)}
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

def build_raw_sft_dataset(model_id):
    raw_datasets = load_raw_ultrachat_dataset()
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
        apply_chat_template,
        num_proc=cpu_count(),
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=column_names,        # which columns to remove from the dataset after applying the function.
        desc="Applying chat template",
    )
    return raw_datasets



from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


def load_text_files_from_directory(directory: str):
    text_data = []
    directory_path = Path(directory)

    # Iterate over all text files in the directory
    for file_path in directory_path.glob("*.txt"):
        # Read the content of each file
        with file_path.open(encoding="utf-8") as file:
            text_data.append(file.read())

    return text_data


def split_text_into_chunks(text, chunk_size=1500, tokenizer=None):
    """Splits text into smaller chunks of size `chunk_size` based on token length."""
    if tokenizer:
        tokens = tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]
        chunked_text = []
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i+chunk_size]
            chunked_text.append(tokenizer.decode(chunk, skip_special_tokens=True))
        return chunked_text
    else:
        # Fallback to character-based splitting (if tokenizer is not available)
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def build_raw_domain_adaptation_dataset(model_id, directory="/data/hedh/", chunk_size=2048, test_size=0.2, random_state=42):
    """Loads text files from a folder, splits them into chunks, and returns a train/test dataset dictionary."""

    # Load the tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load all text files from the specified directory
    text_data = load_text_files_from_directory(directory)

    # Split each document into smaller chunks based on token length
    chunked_texts = []
    for text in text_data:
        chunks = split_text_into_chunks(text, chunk_size, tokenizer)
        chunked_texts.extend(chunks)

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

    return dataset_dict




if __name__ == '__main__':
    from settings import model_id
    ds = build_raw_sft_dataset(model_id)
    print()

    domain_adaptation_dataset = build_raw_domain_adaptation_dataset(model_id)
    print()