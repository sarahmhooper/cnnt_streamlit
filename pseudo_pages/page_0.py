"""
First page of the UI

Used to set up model configuration and get inputs
- model name
- noisy images
- axis order
- data type
- config
"""

import math
import datetime
import streamlit as st

from utils.utils import *

from inputs.inputs_class import Inputs_Class
from model.model_class import Model_Class
from outputs.outputs_class import Outputs_Class

ic : Inputs_Class = st.session_state.inputs_class
mc : Model_Class = st.session_state.model_class
oc : Outputs_Class = st.session_state.outputs_class

def page_0():

    model_list = mc.get_model_list()

    model_name = st.selectbox("Select the model to use for fine tuning", model_list)
    mc.set_model_path(model_name=model_name)

    config_update_dict = get_config_update()

    noisy_images = st.file_uploader("Noisy Images", accept_multiple_files=True)
    clean_images = st.file_uploader("Clean Images", accept_multiple_files=True)

    if noisy_images == [] or clean_images == []:
        st.stop()

    format_a, format_d = ic.read_inputs_files(noisy_images, clean_images)

    display_image_info()

    format_a, format_d = get_formats(format_a, format_d)
    ic.set_format(format_a=format_a, format_d=format_d)

    mc.set_config_update_dict(config_update_dict)

def display_image_info():

    def build_info_table(start, end):

        noisy_name_dict = {}
        clean_name_dict = {}
        shape_dict = {}

        for i in range(start, end):
            noisy_name_dict[f"{i}"] = ic.get_noisy_im_name(i)
            clean_name_dict[f"{i}"] = ic.get_clean_im_name(i)
            shape_dict[f"{i}"] = ic.get_noisy_im_shape(i)

        final_dict = {
            "Noisy":noisy_name_dict,
            "Clean":clean_name_dict,
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

def get_config_update():
    #TODO: caliburate on machine setup

    config_update_dict = {}

    def get_name():

        now = datetime.datetime.now()
        now = now.strftime("%m-%d-%Y_T%H-%M-%S")
        config_update_dict["date"] = now

        return st.text_input("Model Name", value=f"Model_{now}")

    def get_epoch():

        return st.number_input("Number of Epochs", value=30, min_value=0, format="%d")

    def get_cutout():

        col1, col2, col3 = st.columns(3)

        with col1:
            time_c = st.number_input("Time Cutout", min_value=2, format="%d", value=16)
        with col2:
            height = st.multiselect("Height Cutout(s)", [64, 128, 160, 256], default=128)
        with col3:
            widthc = st.multiselect("Width Cutout(s)", [64, 128, 160, 256], default=128)

        return time_c, height, widthc

    def get_lr():

        return st.number_input("Learning Rate", min_value=0.0, format="%f", value=0.000025)

    def get_batch_size():

        return st.number_input("Batch Size", min_value=1, format="%d", value=8)

    def get_loss():

        return st.multiselect("Loss(es)", ["ssim", "ssim3D", "l1", "mse", "sobel"], default="ssim")

    def get_loss_weights(n):

        col_list = st.columns(n)
        loss_weights = [0 for _ in range(n)]

        for i in range(n):

            with col_list[i]:
                loss_weights[i] = st.number_input(f"Loss {i} weight", min_value=0.0, format="%f", value=1.0)

        return loss_weights

    def get_save_cycle():

        return st.number_input("Save Cycle (Epochs between each model save and image show)", min_value=1, format="%d", value=5)
    
    config_update_dict["model_file_name"] = get_name()

    col1, col2, col3 = st.columns(3)
    with col1: config_update_dict["num_epochs"] = get_epoch()
    with col2: config_update_dict["global_lr"] = get_lr()
    with col3: config_update_dict["batch_size"] = get_batch_size()

    config_update_dict["time"], config_update_dict["height"], config_update_dict["width"] = get_cutout()
    config_update_dict["loss"] = get_loss()
    n_loss = len(config_update_dict["loss"])
    if n_loss > 1 : config_update_dict["loss_weights"] = get_loss_weights(n_loss)
    config_update_dict["save_cycle"] = get_save_cycle()

    return config_update_dict

def get_format_a(col, format_a):
    
    options = ("THW", "HWT")

    if format_a == "THW":
        index = 0

    if format_a == "HWT":
        index = 1

    with col:
        return st.radio(
                "Format of the images (Label of axis \
                inferred from image. Correct if wrong.)",
                options, index=index)

def get_format_d(col, format_d):

    options = ("8-bit", "16-bit")

    if format_d == "8-bit":
        index = 0

    if format_d == "16-bit":
        index = 1

    with col:
        return st.radio(
                "Format of the data (Size of data \
                inferred from image. Correct if wrong.)",
                options, index=index)

def get_formats(format_a, format_d):

    col1, col2 = st.columns(2)
    return get_format_a(col1, format_a), get_format_d(col2, format_d)
    