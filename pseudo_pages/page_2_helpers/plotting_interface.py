

import streamlit as st

from utils.utils import *
from .plotting_options import *

from inputs.input_class import Input_Class
from model.model_class import Model_Class
from outputs.output_class import Output_Class

ic : Input_Class = st.session_state.input_class
mc : Model_Class = st.session_state.model_class
oc : Output_Class = st.session_state.output_class

def plotting_st():

    st.write("Image Plots")

    plot_ind = index_slider(ic.get_num_images())

    plot_image(plot_ind)

def index_slider(num_images):

    max_value = num_images-1 if not num_images == 1 else 1
    return st.slider("Index of Image to plot", min_value=0, max_value=max_value, disabled=num_images==1)


def plot_image(index):
    # Given index, plot the pair of noisy, pred, and clean images

    name = ic.noisy_im_names[index]

    noisy_im = ic.noisy_im_list[index]
    predi_image = ic.predi_im_list[index]
    clean_im = ic.clean_im_list[index] if ic.clean_im_list is not None else None

    plot_three(name, noisy_im, predi_image, clean_im)
