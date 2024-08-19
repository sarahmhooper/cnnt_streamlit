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
import streamlit as st

from utils.utils import *

from pseudo_pages.page_0_helpers import *

from inputs.input_class import Input_Class
from model.model_class import Model_Class
from outputs.output_class import Output_Class

ic : Input_Class = st.session_state.input_class
mc : Model_Class = st.session_state.model_class
oc : Output_Class = st.session_state.output_class

def page_0():

    model_setup_st()

    image_reader_st()
