from settings import output_base_dir
from transformers import AutoTokenizer, AutoModelForCausalLM


# Load the fine-tuned tokenizer and model from your output directory
tokenizer = AutoTokenizer.from_pretrained(output_base_dir + '/save_tokenizer')
model = AutoModelForCausalLM.from_pretrained(output_base_dir + '/save_model', load_in_4bit=True, device_map="auto")

# Define the prompt/question
prompt_1 = "You are a scientific advisor in heat and mass transfer field. Enlist how many modes of heat transfer exist."
prompt_2 = "You are a scientific advisor in heat and mass transfer field. Do a short introduction to Foam System."

input_ids = tokenizer(prompt_2, return_tensors="pt").input_ids

# inference
outputs = model.generate(
    input_ids=input_ids,
    max_new_tokens=512,
    do_sample=False,
    temperature=0.1,
    top_k=5,
    top_p=0.9
)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

