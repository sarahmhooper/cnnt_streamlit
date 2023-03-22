"""
Main file for the UI

Keeps track of the pages and shares data in between
"""

import streamlit as st

from utils.utils import *
from inputs.inputs_class import Inputs_Class
from model.model_class import Model_Class
from outputs.outputs_class import Outputs_Class

def init_session_state():
    # Initializes the classes and page number

    args = arg_parse()

    if "page_num" not in st.session_state:
        st.session_state.page_num = 0

    if "inputs_class" not in st.session_state:
        st.session_state.inputs_class = Inputs_Class()

    if "model_class" not in st.session_state:
        st.session_state.model_class = Model_Class(args=args)

    if "outputs_class" not in st.session_state:
        st.session_state.outputs_class = Outputs_Class()

init_session_state()

from pseudo_pages.page_0 import page_0
from pseudo_pages.page_1 import page_1
from pseudo_pages.page_2 import page_2

placeholder = st.empty()
sst = st.session_state

if sst.page_num == 0:

    page_0(placeholder)

if sst.page_num == 1:
   
    page_1(placeholder)

if sst.page_num == 2:
    
    page_2(placeholder)

# Render buttons at the bottom of the page to prevent early render
st.button("Next",on_click=nextpage,disabled=(sst.page_num >= 2))
st.button("Restart",on_click=restart,disabled=(sst.page_num == 0))