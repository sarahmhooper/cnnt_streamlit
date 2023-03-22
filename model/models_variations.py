"""
Main file for handling model variations

TODO: add support for loading empty model (training from scratch)
"""

import os
import json
import torch
import pathlib
import streamlit as st
from argparse import Namespace

from model.models import CNNT_enhanced_denoising_runtime
from utils.utils import *

###################################################################################################

# Multiple model types

def load_true_model(model_path, config_update_dict):
    # load model where model is created from local files and weights are loaded in
    # Requires a json file with same name to create the model

    
    json_path = pathlib.Path(model_path).with_suffix('.json')

    if not os.path.exists(json_path):
        raise JsonNotPresent(f"{json_path} not found. Required to create the model")

    config = json.load(open(json_path))

    for key in config_update_dict:
        config[key] = config_update_dict[key]
    config["dp"] = torch.cuda.is_available() and torch.cuda.device_count() > 1

    config = Namespace(**config)

    config.load_path = model_path
    model = CNNT_enhanced_denoising_runtime(config=config)

    return model, config

def create_true_model(config):
    # Create a new model given config

    config = Namespace(**config)

    config.load_path = None
    model = CNNT_enhanced_denoising_runtime(config=config)

    return model, config

def filter_f(model_path_dir):
    # filter function for model types
    
    def filter_fx(x):
        """
        Given a file name return True of False if the type is supported for loading: .pt format
        @inputs:
            - x : the filename
        @ returnL
            - True or False depending on whether the model type is supported
        """
        file_ext = pathlib.Path(x).suffix.lower()
        return file_ext in [".pt"]

    return sorted(filter(filter_fx, os.listdir(model_path_dir)))

###################################################################################################

# One funtion to read all different types of models from local paths
def load_model_path(model_path, config_update_dict):

    st.write(f"Loading model: {os.path.basename(model_path)}")

    file_ext = pathlib.Path(model_path).suffix.lower()
    if file_ext == ".pt":
        model, config = load_true_model(model_path, config_update_dict)
    elif os.path.basename(model_path) == "Empty Model (Train from scratch)":
        model, config = create_true_model(config_update_dict)
    else:
        raise FileTypeNotSupported(f"Model type in not supported:{file_ext}")

    return model, config

# One funtion to read all different types of models from uploaded files
def load_model_file(model_files, config_update_dict):

    pt_f, json_f = (model_files[0], model_files[1]) if model_files[0].name.endswith(".pt") \
        and model_files[1].name.endswith(".json") else (model_files[1], model_files[0])

    st.write(f"Loading model: {pt_f.name}")

    config = json.load(json_f)
    
    for key in config_update_dict:
        config[key] = config_update_dict[key]
    config["dp"] = torch.cuda.is_available() and torch.cuda.device_count() > 1

    config = Namespace(**config)

    config.load_path = pt_f
    model = CNNT_enhanced_denoising_runtime(config=config)

    return model, config

###################################################################################################

# Wrapper around local vs uploaded models

def load_model(model_path, model_files, config_update_dict):

    if model_path != None:
        return load_model_path(model_path=model_path, config_update_dict=config_update_dict)
    else:
        return load_model_file(model_files=model_files, config_update_dict=config_update_dict)
