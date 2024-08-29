"""
Main file for handling model variations
"""

import os
import torch
from argparse import Namespace

from models.model_cnnt import CNNT_enhanced_denoising_runtime
from utils.utils import *

# -------------------------------------------------------------------------------------------------

def model_list_from_dir(model_path_dir):
    
    return sorted(os.listdir(model_path_dir))

# -------------------------------------------------------------------------------------------------

def load_config_from_path(load_path):

    model_dict = torch.load(load_path)
    config = model_dict['config']
    config.load_path = load_path
    return config

def update_config(config, config_update):

    if not isinstance(config, dict): config = vars(config)
    if not isinstance(config_update, dict): config_update = vars(config_update)

    for key in config_update:
        config[key] = config_update[key]

    config = Namespace(**config)

    return config

def load_model_from_config(config):

    model_dict = torch.load(config.load_path)
    model = CNNT_enhanced_denoising_runtime(config=config)
    model.load_state_dict(model_dict["model_state"])

    return model, config

def load_model_from_path(load_path, device):

    config = load_config_from_path(load_path)
    config.device = device
    return load_model_from_config(config)

def load_model_from_files(model_file, device):

    model_dict = torch.load(model_file)
    config = model_dict['config']
    config.device = device
    model = CNNT_enhanced_denoising_runtime(config=config)
    model.load_state_dict(model_dict["model_state"])
    return model, config

# -------------------------------------------------------------------------------------------------
# Wrapper around local vs uploaded models

def load_model(model_path=None, model_file=None, config=None, device=None):

    if model_path is not None:
        return load_model_from_path(model_path, device)
    elif model_file is not None:
        return load_model_from_files(model_file, device)
    elif config is not None:
        return load_model_from_config(config)
    else:
        raise ValueError(f"Need one of path or file or config to load model")
