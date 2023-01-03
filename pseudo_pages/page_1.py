"""
Second page of the UI

Just runs the inference cycle and shows the progress
Requires a click to get to next page
"""

import streamlit as st

from utils.utils import *

from inputs.inputs_class import Inputs_Class
from model.model_class import Model_Class
from outputs.outputs_class import Outputs_Class

ic : Inputs_Class = st.session_state.inputs_class
mc : Model_Class = st.session_state.model_class
oc : Outputs_Class = st.session_state.outputs_class


def page_1(placeholder):

    placeholder.empty()

    with placeholder.container():

        mc.load_model()

        cut_noisy_im_list, cut_cpred_im_list = mc.run_inference(cut_np_images=ic.get_cut_np_images())

        oc.set_lists(noisy_im_list=cut_noisy_im_list, noisy_im_names=ic.get_noisy_im_names(), cpred_im_list=cut_cpred_im_list)
