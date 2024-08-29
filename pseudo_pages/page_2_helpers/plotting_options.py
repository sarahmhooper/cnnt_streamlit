"""
Interface for plotting images for visual inspection
"""

import numpy as np
import streamlit as st

def scale_and_clip(image, min, max):
    # Scales the image to given min max and then clips it to [0,1]

    image = (image-min)/(max-min+0.00000001)

    return np.clip(image, 0, 1)


def plot_image(image, col, key):
    # Given image and column plot the image and give the ability to change clip values

    with col:

        min_l : float = np.min(image).item()
        max_l : float = np.max(image).item()

        min_l, max_l = st.slider("Image Clip Values", min_value=min_l, max_value=max_l, value=(min_l, max_l), step=0.0001, key=f"{key}_slider")

        st.image(scale_and_clip(image, min_l, max_l), caption=key)


def plot_three(name, noisy, cpred, clean=None):
    # plot the given image pair/triple

    st.write(f"Plotting Image {name}")
    image_frame = st.slider("Frame to Display", min_value=0, max_value=noisy.shape[0]-1 if not noisy.shape[0]==1 else 1, disabled=noisy.shape[0]==1)

    if clean is None:
        col1, col2 = st.columns(2)

        plot_image(noisy[image_frame], col1, "Noisy")
        plot_image(cpred[image_frame], col2, "Predicted")
    else:
        col1, col2, col3 = st.columns(3)

        plot_image(noisy[image_frame], col1, "Noisy")
        plot_image(cpred[image_frame], col2, "Predicted")
        plot_image(clean[image_frame], col3, "Clean")
