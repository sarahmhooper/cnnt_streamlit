"""
First page of the UI

Used to set up model configuration and get inputs
- model name
- noisy images
- axis order
- data type
- config
"""

from utils.utils import *
from pseudo_pages.page_0_helpers import *

def page_0():

    model_setup_st()

    image_reader_st()

    config_update_st()
