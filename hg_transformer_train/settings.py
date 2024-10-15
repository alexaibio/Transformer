import torch
from pathlib import Path

output_dir = 'data/check_points_Mistral-7B-v0.1'

model_id = "mistralai/Mistral-7B-v0.1"

device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None


def get_project_root() -> Path:
    return Path(__file__).parent




