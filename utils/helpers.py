import torch
import torch.nn as nn

def save_checkpoint(model:nn.Module, filename:str):
    checkpoint = model.state_dict()
    torch.save(checkpoint, filename)

def load_checkpoint(model:nn.Module, filename:str):
    check_file = torch.load(filename)
    checkpoint = model.load_state_dict(check_file)

    return checkpoint
