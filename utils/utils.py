"""
General purpose utilities for the UI
"""

import torch
import argparse
import numpy as np
import streamlit as st

def nextpage(): st.session_state.page_num += 0.5
def prevpage(): st.session_state.page_num -= 1
def restart(): reset_session_state(); st.session_state.page_num = 0

def reset_session_state():

    for key in st.session_state.keys():
        del st.session_state[key]

def arg_parse_model_path_dir():
    """
    Parses the command line arg of model path directory
    """

    parser = argparse.ArgumentParser("Argument parser for CNNT_Streamlit Inference UI")
    parser.add_argument("--model_path_dir", type=str, default=None, help='The folder containing models')
    parser.add_argument("--check_path", type=str, default=None, help='The folder for temporary model weight saves')
    args = parser.parse_args()
    return args.model_path_dir, args.check_path

def flatten(l):
    return [item for sublist in l for item in sublist]

def normalize_image(image, percentiles=None, values=None, clip=True):
    """
    Normalizes image locally.
    @inputs:
        image: nd numpy array or torch tensor
        percentiles: pair of percentiles ro normalize with
        values: pair of values normalize with
        NOTE: only one of percentiles and values is required
    @return:
        n_img: the image normalized wrt given params.
    """

    assert (percentiles==None and values!=None) or (percentiles!=None and values==None)

    if type(image)==torch.Tensor:
        image_c = image.cpu().detach().numpy()
    else:
        image_c = image

    if percentiles != None:
        i_min = np.percentile(image_c, percentiles[0])
        i_max = np.percentile(image_c, percentiles[1])
    if values != None:
        i_min = values[0]
        i_max = values[1]

    n_img = (image - i_min)/(i_max - i_min)

    if clip:
        return torch.clip(n_img, 0, 1) if type(n_img)==torch.Tensor else np.clip(n_img, 0, 1)

    return n_img

###################################################################################################
# Custom exceptions

class FileTypeNotSupported(Exception):
    """Raise when uploaded file type is not supported"""

class JsonNotPresent(Exception):
    """Raise when .json corresponding to the model is not present"""