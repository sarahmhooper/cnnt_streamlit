"""
First page of the UI

Used to setup the run
Sets up the model
Takes the input images
Updates the config for finetuning
"""

from utils.utils import *
from pseudo_pages.page_0_helpers import *

def page_0():

    model_setup_st()

    st.divider()

    image_reader_st()

    config_update_st()
