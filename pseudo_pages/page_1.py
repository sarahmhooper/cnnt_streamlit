"""
Second page of the UI

Runs the inference/finetuning cycle and shows the progress
Requires a click to get to next page

TODO: bug/annoyance that dead widgets/button from previous page can stay under alive under the progress bar
clicking those button can make the program error out
"""

import streamlit as st

from utils.utils import *
from pseudo_pages.page_1_helpers import *

sst = st.session_state

def page_1():

    if is_inf_mode():
        run_inference_st()
    else:
        run_training_st()
