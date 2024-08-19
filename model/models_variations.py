"""
Main file for handling model variations
"""

import os
import json
import torch
import pathlib
from argparse import Namespace

from model.model_cnnt import CNNT_enhanced_denoising_runtime
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

# -------------------------------------------------------------------------------------------------

# Wrapper around local vs uploaded models

def load_model(model_path, model_files, config_update_dict, device):

    if model_path != None:
        return load_model_path(model_path=model_path, config_update_dict=config_update_dict, device=device)
    else:
        raise NotImplementedError(f"Loading from uploaded model not supported yet")
        # return load_model_file(model_files=model_files, config_update_dict=config_update_dict)
