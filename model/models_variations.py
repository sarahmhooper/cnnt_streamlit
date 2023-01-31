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

    return filter(filter_fx, os.listdir(model_path_dir))

###################################################################################################

# One funtion to read all different types of models
def load_model(model_path, config_update_dict):

    st.write(f"Loading model: {os.path.basename(model_path)}")

    file_ext = pathlib.Path(model_path).suffix.lower()
    if file_ext == ".pt":
        model, config = load_true_model(model_path, config_update_dict)
    else:
        raise FileTypeNotSupported(f"Model type in not supported:{file_ext}")

    return model, config
