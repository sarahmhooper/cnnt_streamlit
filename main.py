import streamlit as st
import numpy as np
import tifffile
from io import BytesIO
import os

import argparse

import torch

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plotting_utils import *
from inference import *
from utils import *

parser = argparse.ArgumentParser("Argument parser for CNNT_Streamlit")
parser.add_argument("--model_path", type=str, default=None, help='The model to run inference on')

args = parser.parse_args()

# st.write(f"Loading model from {args.model_path}")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = BytesIO(uploaded_file.read()) 
    noisy_im = np.array(tifffile.imread(bytes_data))

    cutout_shape = get_cutout()

    clean_pred = run_model(noisy_im)

    plot_2_images([noisy_im, clean_pred], path="./tmp/", id="temp_vid", show=False)

    st.video("./tmp/temp_vid")