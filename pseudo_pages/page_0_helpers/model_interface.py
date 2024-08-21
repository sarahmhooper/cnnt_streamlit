

import streamlit as st

from utils.utils import *

from inputs.input_class import Input_Class
from model.model_class import Model_Class
from outputs.output_class import Output_Class

ic : Input_Class = st.session_state.input_class
mc : Model_Class = st.session_state.model_class
oc : Output_Class = st.session_state.output_class

def model_setup_st():

    placeholder1 = st.empty()
    # placeholder2 = st.empty()
    placeholder3 = st.empty()

    is_inf = is_inf_mode()

    model_list = mc.get_model_list()

    col1, col2 = placeholder1.columns([3, 1])

    # TODO: Empty and Upload:
    # model_name = st.selectbox("Select the model to use for fine tuning", ["Select a Model", "Empty Model (Train from scratch)", "Upload a Model", *model_list])
    model_name = col1.selectbox("Select the model", model_list, index=None,\
                                placeholder="Select the model by clicking here", label_visibility="collapsed")

    # if model_name == "Select a Model":
    #     st.stop()
    # if model_name == "Upload a Model":
    #     model_files = get_model()
    def load_model_wrapper():
        placeholder3.text(f"Model loading in progress: {model_name}")
        model_files = []
        mc.load_model(model_name, model_files, is_inf)

    col2.button("Load Model", on_click=load_model_wrapper, disabled=model_name is None, use_container_width=True)

    placeholder3.text(f"Model loaded: {mc.model_name}")

# def get_model():

#     return st.file_uploader("Upload Model (.pt+.json)", accept_multiple_files=True)
