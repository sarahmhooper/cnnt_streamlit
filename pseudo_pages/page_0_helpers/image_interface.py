
import math
import streamlit as st

from utils.utils import *

from inputs.input_class import Input_Class
from model.model_class import Model_Class
from outputs.output_class import Output_Class

ic : Input_Class = st.session_state.input_class
mc : Model_Class = st.session_state.model_class
oc : Output_Class = st.session_state.output_class

def image_reader_st():

    is_inf = is_inf_mode()
    
    noisy_images = st.file_uploader("Noisy Image(s)", accept_multiple_files=True)

    clean_img_str = "Clean Image(s) (optional for comparison)" if is_inf else "Clean Image(s)"
    clean_images = st.file_uploader(clean_img_str, accept_multiple_files=True)

    if len(noisy_images):
        with st.spinner("Reading inputs"):
            ic.read_input_files(noisy_images, clean_images)
        display_image_info()

def display_image_info():

    def build_info_table():

        noisy_name_dict = {}
        clean_name_dict = {}
        shape_dict = {}

        for i in range(ic.get_num_images()):
            noisy_name_dict[f"{i}"] = ic.noisy_im_names[i]
            clean_name_dict[f"{i}"] = ic.clean_im_names[i] if ic.clean_im_names is not None else "None"
            shape_dict[f"{i}"] = ic.noisy_im_list[i].shape

        final_dict = {
            "Noisy":noisy_name_dict,
            "Clean":clean_name_dict,
            "Shape":shape_dict,
        }

        return final_dict

    st.dataframe(build_info_table(), use_container_width=True)
