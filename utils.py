import numpy as np
import streamlit as st
import io
import tifffile

def get_cutout():

    time_cutout_1 = st.selectbox(
                "Time cutout start",
                (0, 16, 32, 64, "custom"))

    if time_cutout_1 == "custom":
        time_cutout_1 = st.number_input("Time cutout start", min_value=0, format="%d")

    time_cutout_2 = st.selectbox(
                "Time cutout end",
                (20, 30, 64, 128, "custom"))

    if time_cutout_2 == "custom":
        time_cutout_2 = st.number_input("Time cutout end", min_value=0, format="%d")

    HW_cutout_1 = st.selectbox(
                    "Height and Width cutout start (square patch)",
                    (0, 64, 128, 192, "custom"))

    if HW_cutout_1 == "custom":
        HW_cutout_1 = st.number_input("Height and Width cutout start (square patch)", min_value=0, format="%d")

    HW_cutout_2 = st.selectbox(
                    "Height and Width cutout end (square patch)",
                    (64, 128, 192, 256, "custom"))

    if HW_cutout_2 == "custom":
        HW_cutout_2 = st.number_input("Height and Width cutout end (square patch)", min_value=0, format="%d")

    cutout_shape = (time_cutout_1, time_cutout_2, HW_cutout_1, HW_cutout_2)

    st.write("Cutout shape is")
    st.write(f"[{time_cutout_1}:{time_cutout_2}, {HW_cutout_1}:{HW_cutout_2}, {HW_cutout_1}:{HW_cutout_2}]")

    
    agree = st.checkbox('Run inference on the above cutout shape?')

    if agree:

        if time_cutout_2 <= time_cutout_1:
            st.write("Please make sure time end point is greater than time start point")
        elif HW_cutout_2 <= HW_cutout_1:
            st.write("Please make sure HW end point is greater than HW start point")
        else:
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
