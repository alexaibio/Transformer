from settings import output_dir
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# Let's generate some new texts with our trained model.

# Load the fine-tuned tokenizer and model from your output directory
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForCausalLM.from_pretrained(output_dir, load_in_4bit=True, device_map="auto")



# We use the tokenizer's chat template to format each message -
# see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]

# prepare the messages for the model
input_ids = tokenizer._apply_chat_template(
    messages,
    truncation=True,
    add_generation_prompt=True,
    return_tensors="pt").to("cuda")

# inference
outputs = model.generate(
    input_ids=input_ids,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
