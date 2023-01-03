"""
First page of the UI

Used the gather the inputs and information:
- model name
- noisy images
- axis order
- data type
- cutouts
"""

import math
import streamlit as st

from utils.utils import *

from inputs.inputs_class import Inputs_Class
from model.model_class import Model_Class
from outputs.outputs_class import Outputs_Class

ic : Inputs_Class = st.session_state.inputs_class
mc : Model_Class = st.session_state.model_class
oc : Outputs_Class = st.session_state.outputs_class

def page_0(placeholder):

    placeholder.empty()
    model_list = mc.get_model_list()

    with placeholder.container():

        model_name = st.selectbox("Select the model to use for inference", model_list)
        mc.set_model_path(model_name=model_name)

        uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)

        if uploaded_files == []:
            st.stop()

        ic.read_inputs_files(uploaded_files)

        display_image_info()

        format_a = get_format_a()
        format_d = get_format_d()
        ic.set_format(format_a=format_a, format_d=format_d)

        cutout_shape = get_cutout()
        ic.set_cutouts(cutout_shape=cutout_shape)

def display_image_info():

    def build_info_table(start, end):

        name_dict = {}
        shape_dict = {}

        for i in range(start, end):
            name_dict[f"{i}"] = ic.get_noisy_im_name(i)
            shape_dict[f"{i}"] = ic.get_noisy_im_shape(i)

        final_dict = {
            "Name":name_dict,
            "Shape":shape_dict,
        }

        return final_dict

    # 5 images per tab
    num_images = ic.get_num_images()
    images_per_tab = 5
    num_tabs = math.ceil(num_images/images_per_tab)

    tab_list = st.tabs([f"Images {i*5}-{i*images_per_tab+4}" for i in range(num_tabs)])

    for i, tab in enumerate(tab_list):

        start = i*5
        end = min(num_images, i*5+5)

        with tab:
            st.table(build_info_table(start, end))

    return

def get_cutout():
    
    cut_type = st.selectbox(
                "Select the shape for inference",
                ("Complete Image", "Random Patch", "Custom Patch"))

    if cut_type == "Complete Image":
        cutout_shape = "complete"
    elif cut_type == "Random Patch":
        cutout_shape = "random"
    elif cut_type == "Custom Patch":

        st.write("Out of bounds will be clipped to max values")

        col1, col2 = st.columns(2)

        with col1:

            t_cutout_1 = st.selectbox(
                        "Time cutout start",
                        (0, 16, 32, 64, "custom"))

            if t_cutout_1 == "custom":
                t_cutout_1 = st.number_input("Time cutout start (custom)", min_value=0, format="%d")

            h_cutout_1 = st.selectbox(
                        "Height cutout start",
                        (0, 64, 128, 256, 512, "custom"))

            if h_cutout_1 == "custom":
                h_cutout_1 = st.number_input("Height cutout start (custom)", min_value=0, format="%d")

            w_cutout_1 = st.selectbox(
                        "Width cutout start",
                        (0, 64, 128, 256, 512, "custom"))

            if w_cutout_1 == "custom":
                w_cutout_1 = st.number_input("Width cutout start (custom)", min_value=0, format="%d")

        with col2:

            t_cutout_2 = st.selectbox(
                        "Time cutout end",
                        (8, 16, 32, 64, "custom"))

            if t_cutout_2 == "custom":
                t_cutout_2 = st.number_input("Time cutout end (custom)", min_value=0, format="%d")

            h_cutout_2 = st.selectbox(
                        "Height cutout end",
                        (64, 128, 256, 512, 1024, "custom"))

            if h_cutout_2 == "custom":
                h_cutout_2 = st.number_input("Height cutout end (custom)", min_value=0, format="%d")

            w_cutout_2 = st.selectbox(
                        "Width cutout end",
                        (64, 128, 256, 512, 1024, "custom"))

            if w_cutout_2 == "custom":
                w_cutout_2 = st.number_input("Width cutout end (custom)", min_value=0, format="%d")

        cutout_shape = (t_cutout_1, t_cutout_2, h_cutout_1, h_cutout_2, w_cutout_1, w_cutout_2)

        st.write(f"Cutout shape is :[{cutout_shape[0]}:{cutout_shape[1]}, {cutout_shape[2]}: \
                                     {cutout_shape[3]}, {cutout_shape[4]}:{cutout_shape[5]}]")

    return cutout_shape

def get_format_a():

    return st.selectbox(
                "Format of the images (Label of axis)",
                ("THW", "HWT"))

def get_format_d():

    return st.selectbox(
                "Format of the data (Size of data)",
                ("8-bit", "16-bit"))
