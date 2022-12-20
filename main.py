import streamlit as st
import numpy as np
import tifffile
import io
from io import BytesIO
import os
import argparse
import torch

from utils.plotting_utils import *
from utils.inference import *
from utils.utils import *

if "page_num" not in st.session_state:
    st.session_state.page_num = 0

placeholder = st.empty()
model_list = set_model_path()

if st.session_state.page_num == 0:
    placeholder.empty()

    with placeholder.container():

        model_name = st.selectbox("Select the model to use for inference", model_list)

        model_path = os.path.join(st.session_state.model_path_dir, model_name)
        st.session_state.model_path = model_path

        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = BytesIO(uploaded_file.read()) 

            input_class = input_type("asd")
            input_class.read_noisy_image(bytes_data=bytes_data)

            st.session_state.input_class = input_class

            st.write(f"Given image shape : {input_class.get_noisy_shape()}")

            cutout_shape = get_cutout()

            st.session_state.cutout_shape = cutout_shape

if st.session_state.page_num == 1:
    placeholder.empty()

    with placeholder.container():
        
        model, config = load_model(st.session_state.model_path)
        st.session_state.model = model
        st.session_state.config = config

        noisy_cut, clean_pred = run_model(st.session_state.model, [], st.session_state.input_class.get_noisy_image(), st.session_state.cutout_shape)

        st.session_state.noisy_cut = noisy_cut
        st.session_state.clean_pred = clean_pred

    st.session_state.no_page_change = False

if st.session_state.page_num == 2:
    placeholder.empty()

    with placeholder.container():

        download_pair(st.session_state.noisy_cut, st.session_state.clean_pred)

        plot_and_show(st.session_state.noisy_cut, st.session_state.clean_pred)
    
    st.session_state.no_page_change = False


st.button("Next",on_click=nextpage,disabled=(st.session_state.page_num >= 2))
st.button("Restart",on_click=restart)