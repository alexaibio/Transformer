"""
SFT
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb

"""
from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoTokenizer


####### human DATASET
raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k")

# remove this when done debugging
indices = range(0,100)

dataset_dict = {"train": raw_datasets["train_sft"].select(indices),
                "test": raw_datasets["test_sft"].select(indices)}

raw_datasets = DatasetDict(dataset_dict)

example = raw_datasets["train"][0]
messages = example["messages"]
for message in messages:
  role = message["role"]
  content = message["content"]
  print('{0:20}:  {1}'.format(role, content))


######## Load tokenizer
model_id = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
    tokenizer.model_max_length = 2048

# Set chat template
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE


######## Apply chat template
