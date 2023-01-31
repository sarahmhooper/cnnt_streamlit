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


def page_2():

    download_function()

    st.markdown("""---""")

    st.write("Image Plots")

    if infer_images():
        plot_ind = index_slider(ic.get_num_images())

        oc.plot_image(plot_ind)

def download_function():

    col1, col2 = st.columns(2)

    with col2:
        d_typ = st.radio(
            "Download Model Type", 
            [".pt + .json config as zip"],
        )

    with col1:
        if prepare_download():
            oc.prepare_download(d_typ)

def prepare_download():

    return st.checkbox("Prepare Download? (uncheck, unless downloading)")

def index_slider(num_images):

    return st.slider("Index of Image", min_value=0, max_value=num_images-1 if not num_images == 1 else 1, disabled=num_images==1)

def infer_images():

    return st.checkbox("Run inference on train images?")
