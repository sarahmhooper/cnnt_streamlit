

import streamlit as st

from utils.utils import *

from inputs.input_class import Input_Class
from models.model_class import Model_Class
from outputs.output_class import Output_Class

ic : Input_Class = st.session_state.input_class
mc : Model_Class = st.session_state.model_class
oc : Output_Class = st.session_state.output_class


def download_st():

    is_inf = is_inf_mode()

    with st.spinner("Preparing Download"):
        if is_inf:
            oc.prepare_download_image_all(ic.predi_im_list, ic.noisy_im_names)
        else:
            oc.prepare_download_model(mc.model, mc.config)

    if is_inf:
        download_inference_interface(ic.get_num_images())
    else:
        download_finetuning_interface()

def download_inference_interface(num_images):

    col1, col2, col3 = st.columns(3)

    with col1:
        options = ["Download all as zip", "Download invidual"]
        d_typ = st.radio(
            "Download Type", 
            options,
            key="download_all"
        )

    with col2:
        d_idx = st.number_input(
            "Image Index to Download Individual",
            min_value=0,
            max_value=num_images-1,
            format="%d",
            disabled=st.session_state.download_all=="Download all as zip",
        )

    d_typ = options.index(d_typ)

    with col3:
        if d_typ:
            st.download_button(
                label=f"Download Predicted Clean Image {d_idx}",
                data = oc.image_idx_buffer[d_idx], # Download buffer
                file_name = oc.image_names_list[d_idx]
            )
        else:
            st.download_button(
                label="Download Predicted Clean Images",
                data = oc.image_all_buffer, # Download buffer
                file_name = 'All_predicted_clean_images.zip' 
            )

def download_finetuning_interface():

    st.download_button(
        label=f"Download Model",
        data = oc.model_buffer, # Download buffer
        file_name = f"{mc.config.model_file_name}.pth"
    )
