"""
SFT
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb
https://github.com/huggingface/alignment-handbook/blob/main/scripts/run_sft.py

"""

from transformers import AutoModelForCausalLM
from settings import output_dir
from transformers import BitsAndBytesConfig
import torch
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments
from settings import device_map, model_id
from load_data import build_raw_sft_dataset, load_model_tokenizer, build_raw_domain_adaptation_dataset
torch.cuda.empty_cache()

# create the splits
#raw_datasets = build_raw_sft_dataset(model_id)
raw_datasets = build_raw_domain_adaptation_dataset()
tokenizer = load_model_tokenizer(model_id)

train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

#for index in random.sample(range(len(raw_datasets["train"])), 3):
#  print(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")



#################### MODEL

# quantize and LoRa
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


# Load the model explicitly
model = AutoModelForCausalLM.from_pretrained(
    model_id,  # your model ID from Hugging Face Hub
    **model_kwargs  # Unpack the model_kwargs dictionary
)





## Define SFTTrainer from TRL library.
# This class inherits from the Trainer class available in the Transformers library,
# but is specifically optimized for supervised fine-tuning (instruction tuning).


# path where the Trainer will save its checkpoints and logs


# based on config
training_args = TrainingArguments(
    bf16=True,      # fp16=True or specify bf16=True instead when training on GPUs that support bf16
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=128,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=5,     # was 1
    output_dir=output_dir + '/domain_adaptation',
    overwrite_output_dir=True,
    per_device_eval_batch_size=2,   # 1 originally set to 8
    per_device_train_batch_size=2,  # 1 originally set to 8
    # push_to_hub=True,
    # hub_model_id="zephyr-7b-sft-lora",
    # hub_strategy="every_save",
    # report_to="tensorboard",
    save_strategy="no",
    save_total_limit=None,
    seed=42,

)

# based on config
peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

trainer = SFTTrainer(
        model=model,
        #model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True,
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
    )

train_result = trainer.train()
print('---------------------- TRAINING FINISHED! ---------------------')

metrics = train_result.metrics

max_train_samples = training_args.max_train_samples if hasattr(training_args, 'max_train_samples') else len(train_dataset)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))
trainer.log_metrics("train", metrics)

trainer.save_metrics("train", metrics)
trainer.save_state()


# Save the fine-tuned model and tokenizer
trainer.save_model(output_dir)          # this saves the model, including any adapters
tokenizer.save_pretrained(output_dir)   # save the tokenizer



