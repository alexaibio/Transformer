import torch
from pathlib import Path
from gpt import GPTLanguageModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'


text = ""
for file_path in Path('./data/hedh').glob('*.txt'):
    with file_path.open('r', encoding='utf-8') as f:
        text += f.read() + "\n"

chars = sorted(list(set(text)))
vocab_size = len(chars)

# save the model
model_save_path = 'gpt2_hedh_model_10000.pth'  # Specify your desired file name and path
m = GPTLanguageModel(vocab_size)
m.load_state_dict(torch.load(model_save_path, map_location=torch.device(device)))




# generate from the model
itos = { i:ch for i,ch in enumerate(chars) }
decode_fn = lambda l: ''.join([itos[i] for i in l])

torch.manual_seed(1260)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode_fn(m.generate(context, max_new_tokens=2000)[0].tolist()))
