import torch
from pathlib import Path
from gpt import GPTLanguageModel
from gpt_settings import device


text = ""
for file_path in Path('./data/hedh').glob('*.txt'):
    with file_path.open('r', encoding='utf-8') as f:
        text += f.read() + "\n"

chars = sorted(list(set(text)))
vocab_size = len(chars)

# LOAD model
model_save_path = 'gpt2_hedh_model_15000.pth'  # Specify your desired file name and path
m = GPTLanguageModel(vocab_size)
m.load_state_dict(torch.load(model_save_path, map_location=torch.device(device)))


# generate from the model
itos = { i:ch for i,ch in enumerate(chars) }
decode_fn = lambda l: ''.join([itos[i] for i in l])

torch.manual_seed(2260)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode_fn(m.generate(context, max_new_tokens=4000)[0].tolist()))
