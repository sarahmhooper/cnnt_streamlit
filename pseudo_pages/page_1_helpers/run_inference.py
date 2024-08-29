"""
Interface for running inference
Shows the total progress and progress per image
Does not allow user interaction but does not stop it either
If user interacts with dead widgets on the screen it can error out
"""

import streamlit as st

from utils.utils import *
from .running_inference import running_inference

from inputs.input_class import Input_Class
from models.model_class import Model_Class
from outputs.output_class import Output_Class

ic : Input_Class = st.session_state.input_class
mc : Model_Class = st.session_state.model_class
oc : Output_Class = st.session_state.output_class

sst = st.session_state

def run_inference_st():

    with st.spinner("Preparing Images"):
        ic.scale_images()

    predi_im_list = running_inference(mc.model,
                                        ic.noisy_im_list,
                                        sst.args.cutout,
                                        sst.args.overlap,
                                        sst.args.device)

    ic.set_predi_im_list(predi_im_list)
