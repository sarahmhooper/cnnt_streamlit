"""
General purpose utilities for the UI
"""

import argparse
import streamlit as st

def nextpage(): st.session_state.page_num += 1
def prevpage(): st.session_state.page_num -= 1
def restart(): reset_session_state(); st.session_state.page_num = 0

def reset_session_state():

    for key in st.session_state.keys():
        del st.session_state[key]

def arg_parse_model_path_dir():
    """
    Parses the command line arg of model path directory
    """

    parser = argparse.ArgumentParser("Argument parser for CNNT_Streamlit Inference UI")
    parser.add_argument("--model_path_dir", type=str, default=None, help='The folder containing models')
    args = parser.parse_args()
    return args.model_path_dir

def flatten(l):
    return [item for sublist in l for item in sublist]

###################################################################################################
# Custom exceptions

class FileTypeNotSupported(Exception):
    """Raise when uploaded file type is not supported"""

class JsonNotPresent(Exception):
    """Raise when .json corresponding to the model is not present"""