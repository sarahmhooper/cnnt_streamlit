import os
import time
import json
import torch
import numpy as np
import streamlit as st
from pathlib import Path

from utils import *
from plotting_utils import *


def device_run(model, noisy_im, device):

    model = model.to(device)
    noisy_im = noisy_im.to(device)

    return model(noisy_im).cpu().detach().numpy()

def load_model(model_path):

    st.write(f"Loading model from {model_path}")

    model = torch.jit.load(model_path)
    model.eval()

    json_path = Path(model_path).with_suffix('.json')

    if not os.path.exists(json_path):
        print("Require a .json file with same name as model to work properly.")
        exit(-1)

    config = json.load(open(json_path))

    return model, config


def run_model(model, config, model_path, noisy_im, cutout_shape=(0,32,0,128)):


    if "Microscopy" in model_path:
        st.write("Using limits from json")
        noisy_max = config["limits"]["noisy_max"]
        noisy_min = config["limits"]["noisy_min"]
    else:
        st.write("Inferring limits from the provided image")
        noisy_max = np.min(noisy_im)
        noisy_min = np.max(noisy_im)

    noisy_im = (noisy_im - noisy_min) / (noisy_max - noisy_min)

    noisy_im = noisy_im[cutout_shape[0]:cutout_shape[1], cutout_shape[2]:cutout_shape[3], cutout_shape[2]:cutout_shape[3]]

    T, H, W = noisy_im.shape
    noisy_im = noisy_im.reshape(1, T, 1, H, W)
    noisy_im = torch.from_numpy(noisy_im.astype(np.float32))

    try:
        start = time.time()
        st.write("Running on GPU")
        clean_pred = device_run(model, noisy_im, 'cuda').reshape(T, H, W)
    except:
        start = time.time()
        st.write("Failed on GPU, Running on CPU")
        clean_pred = device_run(model, noisy_im, 'cpu').reshape(T, H, W)

    st.write(f"Prediction took {time.time() - start : .3f} seconds")

    return torch_2_numpy(noisy_im), torch_2_numpy(clean_pred)



def plot_and_show(noisy_cut, clean_pred):

    clip_1, clip_2 = get_clip()

    st.write("Preparing plots")

    plot_2_images([noisy_cut, np.clip(clean_pred, clip_1, clip_2)], path="./tmp/", id="temp_vid", show=False)

    st.video("./tmp/temp_vid.mp4")
    