"""
General purpose utilities for the UI
"""

import os
import torch
import argparse
import numpy as np
import streamlit as st

from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

def reset_session_state():

    for key in st.session_state.keys():
        del st.session_state[key]

def arg_parse():
    """
    Parses the command line arg of model path directory
    """

    parser = argparse.ArgumentParser("Argument parser for CNNT_Streamlit Inference UI")
    parser.add_argument("--debug", "-D", action="store_true", help='Option to run in debug mode')
    parser.add_argument("--model_path_dir", type=str, default=None, help='The folder containing models')

    parser.add_argument("--cutout", nargs="+", type=int, default=[8,128,128], help='cutout for inference')
    parser.add_argument("--overlap", nargs="+", type=int, default=[2,32,32], help='overlap for inference')
    parser.add_argument("--cuda_devices", type=str, default=None, help='devices for cuda training')
    parser.add_argument("--num_workers", type=int, default=2, help='worker for dataloader')
    parser.add_argument("--prefetch_factor", type=int, default=4, help='prefetching for dataloader')

    args = parser.parse_args()

    args = check_args(args)

    return args

def check_args(args):
    #TODO: check args properly

    if args.cuda_devices == "cuda":
        args.device = "cuda"
    elif args.cuda_devices not in [None,"None"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        args.device = "cuda"
    else:
        args.device = "cpu"

    return args

def is_inf_mode():
    return st.session_state.run_type == "Inference"

def is_dbg_mode():
    return st.session_state.args.debug

def flatten(l):
    return [item for sublist in l for item in sublist]

def normalize_image(image, percentiles=None, values=None, clip=True):
    """
    Normalizes image locally.
    @args:
        - image: nd numpy array or torch tensor
        - percentiles: pair of percentiles ro normalize with
        - values: pair of values normalize with
        NOTE: only one of percentiles and values is required
    @rets:
        - n_img: the image normalized wrt given params.
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

def register_translation_3D(noisy, clean):
    # Register noisy clean pair using translation

    (z_off, y_off, x_off), _, _ = phase_cross_correlation(clean, noisy)

    return shift(noisy, shift=(z_off, y_off, x_off), mode="reflect")

def infer_scale(image):

    if image.dtype == np.uint16: return [0.0,4096.0]
    if image.dtype == np.uint8: return [0.0,256.0]

    return [0.0,65536.0]

# -------------------------------------------------------------------------------------------------
# Custom exceptions

class FileTypeNotSupported(Exception):
    """Raise when uploaded file type is not supported"""

class JsonNotPresent(Exception):
    """Raise when .json corresponding to the model is not present"""
