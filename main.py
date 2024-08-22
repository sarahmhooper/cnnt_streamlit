"""
Main file for the UI

Keeps track of the pages and shares data in between
"""

import datetime
import streamlit as st

from utils.utils import *
from inputs.input_class import Input_Class
from model.model_class import Model_Class
from outputs.output_class import Output_Class

def init_session_state():
    # Initializes the classes and page number

    args = arg_parse()

    if "args" not in st.session_state:
        st.session_state.args = args

    if "page_num" not in st.session_state:
        st.session_state.page_num = 0

    if "input_class" not in st.session_state:
        st.session_state.input_class = Input_Class()

    if "model_class" not in st.session_state:
        st.session_state.model_class = Model_Class(args=args)

    if "output_class" not in st.session_state:
        st.session_state.output_class = Output_Class()

    now = datetime.datetime.now()
    now = now.strftime("%m-%d-%Y_T%H-%M-%S")
    if "datetime" not in st.session_state:
        st.session_state.datetime = now

def nextpage(): st.session_state.page_num += 1
def prevpage(): st.session_state.page_num -= 1
def restart(): reset_session_state(); init_session_state()

init_session_state()

from pseudo_pages.page_0 import page_0
from pseudo_pages.page_1 import page_1
from pseudo_pages.page_2 import page_2

sst = st.session_state
is_inf = is_inf_mode()
title_str = "Inference" if is_inf else "Finetuning"
disable_next = (sst.page_num >= 2)

if sst.page_num == 0:
    st.title(f"{title_str} Session: Setup")
    page_0()
    disable_next |= (not sst.model_class.is_model_loaded())
    disable_next |= (not sst.input_class.get_num_images())

elif sst.page_num == 1:
    st.title(f"{title_str} Session: Running")
    page_1()

elif sst.page_num == 2:
    st.title(f"{title_str} Session: Download Results")
    page_2()

if sst.args.debug:
    st.write(f"disable: {disable_next}, model_loaded: {sst.model_class.is_model_loaded()}, num_images: {sst.input_class.get_num_images()}")
    disable_next = False

# Render buttons at the bottom of the page to prevent early render
st.button("Next",on_click=nextpage,disabled=disable_next)
st.button("Restart",on_click=restart,disabled=(sst.page_num == 0))
