"""
Second page of the UI

Runs the fine tuning cycle and shows the progress
Requires a click to get to next page
"""

import streamlit as st

from utils.utils import *

from inputs.input_class import Input_Class
from model.model_class import Model_Class
from outputs.output_class import Output_Class

ic : Input_Class = st.session_state.input_class
mc : Model_Class = st.session_state.model_class
oc : Output_Class = st.session_state.output_class


def page_1():

    is_inf = is_inf_mode()

    # with st.spinner("Preparing data for training"):
    #     train_set, val_set = mc.prepare_train_n_val(noisy_im_list=ic.get_noisy_ims(), clean_im_list=ic.get_clean_ims(), scale=ic.get_scale())
    # model_tuned, config = mc.run_finetuning(train_set, val_set)

    # oc.set_model(model=model_tuned, config=config, infer_func=mc.run_inference, scale=ic.get_scale())
    # oc.set_lists(noisy_im_list=ic.get_noisy_ims(), noisy_im_names=ic.get_noisy_im_names(), clean_im_list=ic.get_clean_ims())

    oc.set_model(mc.model, mc.config, mc.run_inference, ic.get_scale())


    pred_im_list = mc.run_inference(ic.get_noisy_ims())

    oc.set_lists(noisy_im_list=ic.get_noisy_ims(), noisy_im_names=ic.get_noisy_im_names(), cpred_im_list=pred_im_list)
