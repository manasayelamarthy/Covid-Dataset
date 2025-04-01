import torch
import torch.nn as nn

def save_checkpoint(model:nn.Module, filename:str):
    checkpoint = model.state_dict()
    torch.save(checkpoint, filename)

def load_checkpoint(model:nn.Module, checkpoint_path:str):
    check_file = torch.load(checkpoint_path)
    model.load_state_dict(check_file)

    return model
