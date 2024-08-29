"""
Interface for model setup

Lets user choose a model from the list
Or upload a model that they have finetuned beforehand
"""

import streamlit as st

from utils.utils import *

from inputs.input_class import Input_Class
from models.model_class import Model_Class
from outputs.output_class import Output_Class

ic : Input_Class = st.session_state.input_class
mc : Model_Class = st.session_state.model_class
oc : Output_Class = st.session_state.output_class

def model_setup_st():

    placeholder1 = st.empty()
    placeholder2 = st.empty()

    model_list = ["Upload a Model", *mc.get_model_list()]
    model_file = None

    col1, col2 = placeholder1.columns([3, 1])

    # TODO: Let user train an empty model:
    # model_name = st.selectbox("Select the model to use for fine tuning", ["Select a Model", "Empty Model (Train from scratch)", "Upload a Model", *model_list])
    model_name = col1.selectbox("Select the model", model_list, index=None,\
                                placeholder="Select the model by clicking here", label_visibility="collapsed")

    if model_name == "Upload a Model":
        model_file = get_model()

    def load_model_wrapper():
        placeholder2.text(f"Model loading in progress: {model_name}")
        mc.load_model(model_name, model_file)

    disable_load = model_name is None or (model_name == "Upload a Model" and model_file is None)
    col2.button("Load Model", on_click=load_model_wrapper, disabled=disable_load, use_container_width=True)

    placeholder2.text(f"Model loaded: {mc.model_name}")

def get_model():

    return st.file_uploader("Upload Model", accept_multiple_files=False)
