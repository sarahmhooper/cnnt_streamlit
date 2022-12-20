import argparse
import numpy as np
import streamlit as st
import io
import os
import tifffile

from inputs.tiff_single import Tiff_Single

def nextpage(): st.session_state.page_num += 1
def prevpage(): st.session_state.page_num -= 1
def restart(): st.session_state.page_num = 0

def get_cutout():
    T, H, W = st.session_state.input_class.get_noisy_shape()

    cut_type = st.selectbox(
                "Select the shape for inference",
                ("Full Image", "Random Patch", "Custom Patch"))

    if cut_type == "Full Image":
        cutout_shape = (0, T, 0, H, 0, W)
    elif cut_type == "Random Patch":
        cutout_shape = (0, T//2, 0, H//2, 0, W//2)
    else:

        time_cutout_1 = st.selectbox(
                    "Time cutout start",
                    (0, 16, 32, 64, "custom"))

        if time_cutout_1 == "custom":
            time_cutout_1 = st.number_input("Time cutout start", min_value=0, format="%d")

        time_cutout_2 = st.selectbox(
                    "Time cutout end",
                    (16, 30, 64, 128, "custom"))

        if time_cutout_2 == "custom":
            time_cutout_2 = st.number_input("Time cutout end", min_value=0, format="%d")

        HW_cutout_1 = st.selectbox(
                        "Height and Width cutout start (square patch)",
                        (256, 64, 128, 192, "custom"))

        if HW_cutout_1 == "custom":
            HW_cutout_1 = st.number_input("Height and Width cutout start (square patch)", min_value=0, format="%d")

        HW_cutout_2 = st.selectbox(
                        "Height and Width cutout end (square patch)",
                        (768, 64, 128, 192, 256, "custom"))

        if HW_cutout_2 == "custom":
            HW_cutout_2 = st.number_input("Height and Width cutout end (square patch)", min_value=0, format="%d")

        cutout_shape = (time_cutout_1, time_cutout_2, HW_cutout_1, HW_cutout_2, HW_cutout_1, HW_cutout_2)

    st.write("Cutout shape is")
    st.write(f"[{cutout_shape[0]}:{cutout_shape[1]}, {cutout_shape[2]}:{cutout_shape[3]}, {cutout_shape[4]}:{cutout_shape[5]}]")

    
    # agree = st.checkbox('Run inference on the above cutout shape?')

    # if agree:

    # if time_cutout_2 <= time_cutout_1:
    #     st.write("Please make sure time end point is greater than time start point")
    # elif HW_cutout_2 <= HW_cutout_1:
    #     st.write("Please make sure HW end point is greater than HW start point")
    # else:
    return cutout_shape

    st.stop()


def get_clip():

    st.write("Clipping only changes the following displayed image, not the actual data")

    clip_1 = st.selectbox(
                "Clip lower end",
                (0, 0.25, 0.5, "custom"))

    if clip_1 == "custom":
        clip_1 = st.number_input("Clip lower end", format="%f")

    clip_2 = st.selectbox(
                "Clip upper end",
                (1, 1.5, 2, "custom"))

    if clip_2 == "custom":
        clip_2 = st.number_input("Clip upper end", format="%f")

    if clip_2 > clip_1:
        return clip_1, clip_2
    else:
        st.write("Please make sure the lower end point is smaller than the upper end point")

    st.stop()


def download_pair(noisy_cut, clean_pred):

    with io.BytesIO() as buffer:
        # Write array to buffer
        np.save(buffer, noisy_cut)
        btn = st.download_button(
            label="Download noisy_cut(.npy)",
            data = buffer, # Download buffer
            file_name = 'noisy_cut.npy'
        )

    with io.BytesIO() as buffer:
        # Write array to buffer
        tifffile.imwrite(buffer, noisy_cut)
        btn = st.download_button(
            label="Download noisy_cut(.tiff)",
            data = buffer, # Download buffer
            file_name = 'noisy_cut.tiff'
        )

    with io.BytesIO() as buffer:
        # Write array to buffer
        np.save(buffer, clean_pred)
        btn = st.download_button(
            label="Download clean_pred(.npy)",
            data = buffer, # Download buffer
            file_name = 'clean_pred.npy'
        )

    with io.BytesIO() as buffer:
        # Write array to buffer
        tifffile.imwrite(buffer, clean_pred)
        btn = st.download_button(
            label="Download clean_pred(.tiff)",
            data = buffer, # Download buffer
            file_name = 'clean_pred.tiff'
        )


def set_model_path():
    """
    Sets the model path from cli args to session state
    @returns:
        - list of all .pts file in the directory
    """

    parser = argparse.ArgumentParser("Argument parser for CNNT_Streamlit")
    parser.add_argument("--model_path_dir", type=str, default=None, help='The model to run inference on')
    args = parser.parse_args()
    st.session_state.model_path_dir = args.model_path_dir

    return filter(lambda x: x[-4:] == ".pts", os.listdir(st.session_state.model_path_dir))

def input_type(some_str):

    return Tiff_Single()