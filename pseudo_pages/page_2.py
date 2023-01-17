"""
Third and final page of the UI

Prepare the download option and plots images with interactive controls
"""


import streamlit as st

from utils.utils import *

from inputs.inputs_class import Inputs_Class
from model.model_class import Model_Class
from outputs.outputs_class import Outputs_Class

ic : Inputs_Class = st.session_state.inputs_class
mc : Model_Class = st.session_state.model_class
oc : Outputs_Class = st.session_state.outputs_class


def page_2(placeholder):
    placeholder.empty()

    with placeholder.container():

        format_a, format_d = get_formats()

        d_typ, d_ind, d_for = get_download_type_and_index(ic.get_num_images())

        oc.set_download_params(format_a, format_d, d_typ, d_ind, d_for, *(ic.get_format()))

        if prepare_download():
            oc.prepare_download()

        st.markdown("""---""")

        st.write("Image Plots")

        plot_ind = index_slider(ic.get_num_images())

        oc.plot_image(plot_ind)


def get_formats():

    col1, col2 = st.columns(2)
    
    with col1:
        format_a = st.radio(
            "Download Image Format (Label of axis)",
            ["Same as input", "THW", "HWT"]
        )

    with col2:
        format_d = st.radio(
            "Download Data Format (Size of data)",
            ["Same as input", "8-bit", "16-bit", "Float-32 (Normalized to [0,1])"]
        )

    return format_a, format_d

def get_download_type_and_index(num_images):

    col1, col2 = st.columns(2)

    with col1:
        d_typ = st.radio(
            "Download Type", 
            ["Download all as zip", "Download invidual"],
            key="download_all"
        )

        d_ind = st.number_input(
            "Image Index to Download",
            min_value=0,
            max_value=num_images-1,
            format="%d",
            disabled=st.session_state.download_all=="Download all as zip",
        )

    with col2:
        d_for = st.radio(
            "Download Image Format",
            [".tiff"]
        )

    return d_typ, d_ind, d_for

def prepare_download():

    return st.checkbox("Prepare Download? (uncheck unless downloading)")

def index_slider(num_images):

    return st.slider("Index of Image", min_value=0, max_value=num_images-1 if not num_images == 1 else 1, disabled=num_images==1)
