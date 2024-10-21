from settings import output_base_dir
from transformers import AutoTokenizer, AutoModelForCausalLM
from settings import device_map, model_id
from load_data import load_model_tokenizer
from transformers import BitsAndBytesConfig


# Load initial pre-trained model
tokenizer = load_model_tokenizer(model_id)
#tokenizer = AutoTokenizer.from_pretrained(output_base_dir + '/save_tokenizer')

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
)
model_kwargs = dict(
    attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
    torch_dtype="auto",
    use_cache=False, # set to False as we're going to use gradient checkpointing
    device_map=device_map,
    quantization_config=quantization_config.to_dict(),  # @ALex to_dict added
)

model_pretrained = AutoModelForCausalLM.from_pretrained(
    model_id,  # your model ID from Hugging Face Hub
    **model_kwargs  # Unpack the model_kwargs dictionary
)


model_finetuned = AutoModelForCausalLM.from_pretrained(
    output_base_dir + '/save_model',
    load_in_4bit=True,
    device_map=device_map
)


############# Define the prompt/question
prompt_1 = """
You are a scientific advisor in heat and mass transfer field. Do not allow repetitions. \n
Question: Briefly explain which heat transfer modes exists. \n
Answer:
"""

prompt_2 = """
You are a scientific advisor in heat and mass transfer field. Do not allow repetitions. \n
Question: Do a short introduction to Foam System. \n
Answer: 
"""


input_ids = tokenizer(prompt_1, return_tensors="pt").input_ids



########### Inference with pre trained model
# inference
outputs_pre = model_pretrained.generate(
    input_ids=input_ids,
    max_new_tokens=256,
    do_sample=False,
    temperature=0.1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
    #top_k=3,
    #top_p=0.9
)

print('-------------------- Pre trained model answer ---------------------')
ans = tokenizer.batch_decode(outputs_pre, skip_special_tokens=True)[0]
print(ans)



########### Inference with fine-tuned model



# inference
outputs = model_finetuned.generate(
    input_ids=input_ids,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
    #top_k=3,
    #top_p=0.9
)

print('-------------------- Fine Tuned answer ---------------------')
ans = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(ans)

