"""
Third and final page of the UI

Prepare the download option and plots images with interactive controls
"""


import streamlit as st

from utils.utils import *
from pseudo_pages.page_2_helpers import *

from inputs.input_class import Input_Class
from model.model_class import Model_Class
from outputs.output_class import Output_Class

ic : Input_Class = st.session_state.input_class
mc : Model_Class = st.session_state.model_class
oc : Output_Class = st.session_state.output_class

def page_2():

    download_st()

    st.divider()

    plotting_st()
