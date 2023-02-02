"""
Main file for the UI

Keeps track of the pages and shares data in between
"""

import datetime
import streamlit as st

from utils.utils import *
from inputs.inputs_class import Inputs_Class
from model.model_class import Model_Class
from outputs.outputs_class import Outputs_Class

def init_session_state():
    # Initializes the classes and page number

    model_path_dir, check_path = arg_parse_model_path_dir()

    if "page_num" not in st.session_state:
        st.session_state.page_num = 0

    if "inputs_class" not in st.session_state:
        st.session_state.inputs_class = Inputs_Class()

    if "model_class" not in st.session_state:
        st.session_state.model_class = Model_Class(model_path_dir=model_path_dir, check_path=check_path)

    if "outputs_class" not in st.session_state:
        st.session_state.outputs_class = Outputs_Class()

    now = datetime.datetime.now()
    now = now.strftime("%m-%d-%Y_T%H-%M-%S")
    if "datetime" not in st.session_state:
        st.session_state.datetime = now

init_session_state()

from pseudo_pages.page_0 import page_0
from pseudo_pages.page_1 import page_1
from pseudo_pages.page_2 import page_2

sst = st.session_state

if sst.page_num == 0:
    page_0()

if sst.page_num == 0.5: # Extra page to flush out the screen
    st.write("Config setup complete")
    st.write("Click \"Next\" to begin training")

if sst.page_num == 1:
    page_1()

if sst.page_num == 2 or sst.page_num == 1.5: # 1.5 because incrementing with 0.5
    page_2()

# Render buttons at the bottom of the page to prevent early render
st.button("Next",on_click=nextpage,disabled=(sst.page_num >= 1.5))
st.button("Restart",on_click=restart,disabled=(sst.page_num == 0))