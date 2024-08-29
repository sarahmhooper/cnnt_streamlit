"""
Third and final page of the UI

Prepare the download option and plots images with interactive controls
"""

import streamlit as st

from utils.utils import *
from pseudo_pages.page_2_helpers import *

def page_2():

    download_st()

    st.divider()

    plotting_st()
