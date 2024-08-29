"""
Main file for handling model variations
"""

import os
import json
import torch
import pathlib
from argparse import Namespace

from models.model_cnnt import CNNT_enhanced_denoising_runtime
from utils.utils import *

# -------------------------------------------------------------------------------------------------
# Multiple model types

def load_true_model(model_path, config_update_dict):
    # load model where model is created from local files and weights are loaded in
    # Requires a json file with same name to create the model

    json_path = pathlib.Path(model_path).with_suffix('.json')

    if not os.path.exists(json_path):
        raise JsonNotPresent(f"{json_path} not found. Required to create the model")

    config = vars(json.load(open(json_path)))

    for key in config_update_dict:
        config[key] = config_update_dict[key]

    config = Namespace(**config)

    config.load_path = model_path
    model = CNNT_enhanced_denoising_runtime(config=config)

    return model, config

def load_true_model_pth(load_path, config_update_dict, device="cpu"):
    # load model where model is created from local files and weights are loaded in
    # Requires a json file with same name to create the model

    model_dict = torch.load(load_path, map_location=device)

    config = vars(model_dict['config'])
    config["device"] = device

    for key in config_update_dict:
        config[key] = config_update_dict[key]

    config = Namespace(**config)

    # config.load_path = load_path
    model = CNNT_enhanced_denoising_runtime(config=config)
    model.load_state_dict(model_dict["model_state"])

    return model, config

def create_true_model(config):
    # Create a new model given config

    config = Namespace(**config)

    config.load_path = None
    model = CNNT_enhanced_denoising_runtime(config=config)

    return model, config

def model_list_from_dir(model_path_dir):
    
    return sorted(os.listdir(model_path_dir))

# -------------------------------------------------------------------------------------------------

# One funtion to read all different types of models from local paths
def load_model_path(model_path, config_update_dict, device):

    file_ext = pathlib.Path(model_path).suffix.lower()
    if file_ext == ".pt":
        model, config = load_true_model(model_path, config_update_dict)
    if file_ext == ".pth":
        model, config = load_true_model_pth(model_path, config_update_dict, device)
    elif os.path.basename(model_path) == "Empty Model (Train from scratch)":
        model, config = create_true_model(config_update_dict)
    else:
        raise FileTypeNotSupported(f"Model type in not supported:{file_ext}")

    return model, config

# One funtion to read all different types of models from uploaded files
# def load_model_file(model_files, config_update_dict):

#     pt_f, json_f = (model_files[0], model_files[1]) if model_files[0].name.endswith(".pt") \
#         and model_files[1].name.endswith(".json") else (model_files[1], model_files[0])

#     st.write(f"Loading model: {pt_f.name}")

#     config = json.load(json_f)
    
#     for key in config_update_dict:
#         config[key] = config_update_dict[key]

#     config = Namespace(**config)

#     config.load_path = pt_f
#     model = CNNT_enhanced_denoising_runtime(config=config)

#     return model, config

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
