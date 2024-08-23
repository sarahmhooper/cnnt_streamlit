

import streamlit as st
from datetime import datetime

from utils.utils import *

from inputs.input_class import Input_Class
from model.model_class import Model_Class
from outputs.output_class import Output_Class

ic : Input_Class = st.session_state.input_class
mc : Model_Class = st.session_state.model_class
oc : Output_Class = st.session_state.output_class


def config_update_st():

    sst = st.session_state

    if (not is_inf_mode()) and mc.is_model_loaded() and ic.get_num_images():

        config_update = {
            "model_path_dir" : mc.model_path_dir,
            "model_file_name" : f"test_run_{sst.datetime}",
            "num_epochs" : 30 if not sst.args.debug else 2,
            "im_value_scale" : infer_scale(ic.noisy_im_list[0]),
            "global_lr" : 2.5e-5,
            "device" : sst.args.device,
            "num_workers" : sst.args.num_workers,
            "prefetch_factor" : sst.args.prefetch_factor,
        }

        mc.update_config(config_update)
        mc.reload_model()

    # current_config = mc.config
    # current_config_dict = vars(current_config)

    # st.data_editor(current_config_dict)

    # print(current_config_dict)
    # st.write(current_config_dict)


def get_config_update():
    #TODO: caliburate on machine setup

    config_update_dict = {}

    def get_name():

        return st.text_input("Model Name", value=f"Model_{st.session_state.datetime}")

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

    def get_samples_per_image():

        return st.number_input("Number of train samples per image", min_value=1, format="%d", value=8)
    
    config_update_dict["model_file_name"] = get_name()

    col1, col2, col3 = st.columns(3)
    with col1: config_update_dict["num_epochs"] = get_epoch()
    with col2: config_update_dict["global_lr"] = get_lr()
    with col3: config_update_dict["batch_size"] = get_batch_size()

    config_update_dict["time"], config_update_dict["height"], config_update_dict["width"] = get_cutout()
    config_update_dict["loss"] = get_loss()
    n_loss = len(config_update_dict["loss"])
    config_update_dict["loss_weights"] = [1.0]
    if n_loss > 1 : config_update_dict["loss_weights"] = get_loss_weights(n_loss)
    config_update_dict["save_cycle"] = get_save_cycle()
    config_update_dict["num_samples_per_file"] = get_samples_per_image()

    return config_update_dict

def make_complete_config(config):

    def get_channels():

        blocks = [16, 32, 64]

        col1, col2, col3 = st.columns(3)
        with col1: blocks[0] = st.number_input("Num channels for first layer", min_value=1, format="%d", value=16)
        with col2: blocks[1] = st.number_input("Num channels for second layer", min_value=1, format="%d", value=32)
        with col3: blocks[2] = st.number_input("Num channels for third layer", min_value=1, format="%d", value=64)

        return blocks

    def blocks_and_heads():

        col1, col2 = st.columns(2)
        with col1: b = st.number_input("Number of CNNT blocks per layer", min_value=1, format="%d", value=4)
        with col2: h = st.number_input("Number of Transformer heads per block", min_value=1, format="%d", value=8)

        return b, h

    def conv_params():

        col1, col2, col3 = st.columns(3)
        with col1: k = st.number_input("Kernel size for each Conv", min_value=1, format="%d", value=3)
        with col2: s = st.number_input("Stride for each Conv", min_value=1, format="%d", value=1)
        with col3: p = st.number_input("Padding for each Conv", min_value=1, format="%d", value=1)

        return k, p, s

    def mixer_and_conv3D():

        col1, col2 = st.columns(2)
        with col1: m = 1 if st.checkbox("Use mixer (Conv after attention)?", value=True) else 0
        with col2: c = st.checkbox("Use Conv3D instead of Conv2D?", value=False)

        return m, c

    def optim():

        col1, col2, col3 = st.columns(3)
        with col1: o = st.radio("Optimizer", ["adamw", "nadam", "sgd"])
        with col2: s = st.radio("Scheduler", ["ReduceLROnPlateau", "StepLR"])
        with col3: n = st.radio("Norm mode", ["instance", "layer", "batch"])

        return o, s, n

    def optim_params():

        col1, col2 = st.columns(2)
        with col1:
            l = st.number_input("Global learning rate (Overwrites previous learning rate)", min_value=0.0, format="%f", value=0.0001)
            d = st.number_input("Dropout probability", min_value=0.0, format="%f", value=0.1)
            b = st.number_input("Optimizer beta 1", min_value=0.0, format="%f", value=0.90)
        with col2:
            w = st.number_input("Optimizer weight decay", min_value=0.0, format="%f", value=0.1)
            g = st.number_input("Gradient norm clip", min_value=0.0, format="%f", value=1.0)
            c = st.number_input("Optimizer beta 2", min_value=0.0, format="%f", value=0.95)
            

        return l, d, b, w, g, c
        
    def misc_1():

        col1, col2, col3 = st.columns(3)
        with col1: b = st.checkbox("Bias for Convs", value=False)
        with col2: r = st.checkbox("No residual connection", value=False)
        with col3: w = st.checkbox("No weight decay", value=False)

        return b, r, w

    config["blocks"] = get_channels()
    config["blocks_per_set"], config["n_head"] = blocks_and_heads()
    config["kernel_size"], config["stride"], config["padding"] = conv_params()
    config["optim"], config["scheduler"], config["norm_mode"] = optim()
    config["global_lr"], config["dropout_p"], config["beta1"], \
    config["weight_decay"], config["clip_grad_norm"], config["beta2"] = optim_params()
    config["with_mixer"], config["use_conv_3D"] = mixer_and_conv3D()
    config["bias"], config["no_residual"], config["no_w_decay"] = misc_1()

    return config

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
