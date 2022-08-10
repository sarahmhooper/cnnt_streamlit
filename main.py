import streamlit as st
import numpy as np
import tifffile
import io
from io import BytesIO
import os
import cv2
import argparse

import torch

from plotting_utils import *
from inference import *
from utils import *

parser = argparse.ArgumentParser("Argument parser for CNNT_Streamlit")
parser.add_argument("--model_path", type=str, default=None, help='The model to run inference on')

args = parser.parse_args()
model, config = load_model(args.model_path)

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = BytesIO(uploaded_file.read()) 
    noisy_im = np.array(tifffile.imread(bytes_data))

    st.write(f"Given image shape : {noisy_im.shape}")

    cutout_shape = get_cutout()

    noisy_cut, clean_pred = run_model(model, config, args.model_path, noisy_im, cutout_shape)

    download_pair(noisy_cut, clean_pred)

    plot_and_show(noisy_cut, clean_pred)
