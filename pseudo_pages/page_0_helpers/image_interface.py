
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

        if is_inf:
            set_scale_inf()
        else:
            set_scale_and_register_fnt()

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

def set_scale_inf():

    if not mc.is_model_loaded(): return
    im_value_scale = mc.config.im_value_scale

    if isinstance(im_value_scale, int): im_value_scale = [0,im_value_scale]
    ic.im_value_scale = im_value_scale

def set_scale_and_register_fnt():

    im_value_scale = infer_scale(ic.noisy_im_list[0])
    print(ic.noisy_im_list[0].dtype)

    st.write(f"Enter values to scale images with. The default values for dtype {ic.noisy_im_list[0].dtype} are:")
    col1, col2 = st.columns(2)
    with col1:
        im_value_min = st.number_input(f"Min value", min_value=0.0, max_value=65535.0, value=im_value_scale[0])
    with col2:
        im_value_max = st.number_input(f"Max value", min_value=0.0, max_value=65536.0, value=im_value_scale[1])
    ic.im_value_scale = [im_value_min, im_value_max]

    ic.register_image_check = st.checkbox("Register images with translation in 3D before training?")
