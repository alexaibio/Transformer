import torch
from pathlib import Path
from datetime import datetime

current_run = '_24_10_19'
output_base_dir = f'models/Mistral-7B-v0.1/domain_adaptation_{current_run}'

model_id = "mistralai/Mistral-7B-v0.1"

device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None


def get_project_root() -> Path:
    return Path(__file__).parent




