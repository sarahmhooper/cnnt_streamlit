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

def arg_parse():
    """
    Parses the command line arg of model path directory
    """

    parser = argparse.ArgumentParser("Argument parser for CNNT_Streamlit Inference UI")
    parser.add_argument("--model_path_dir", type=str, default=None, help='The folder containing models')
    parser.add_argument("--cutout", nargs="+", type=int, default=[8,64,64], help='cutout for inference')
    parser.add_argument("--overlap", nargs="+", type=int, default=[2,16,16], help='overlap for inference')
    parser.add_argument("--device", type=str, default="cuda", help='the device to run on')
    args = parser.parse_args()

    return args

def flatten(l):
    return [item for sublist in l for item in sublist]

###################################################################################################
# Custom exceptions

class FileTypeNotSupported(Exception):
    """Raise when uploaded file type is not supported"""

class JsonNotPresent(Exception):
    """Raise when .json corresponding to the model is not present"""