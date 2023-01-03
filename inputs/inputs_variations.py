"""
Main file for reading all different types of input images and
handling input variations like axis order and data types

Currently supported types:
    - .tif/.tiff

(Ideally) Only need to update this file for new input variations
TODO: more input types: .lif, .ometiff
"""

import pathlib
import tifffile
import numpy as np

from io import BytesIO
from utils.utils import *

###################################################################################################

# Multiple input types

def read_tiffs(input_list):
    """
    read the tiff file input
    @args:
        - input_list: a list of byte_data objects
    @requirements:
        - byte_data objects should be in tiff format
    @returns:
        - noisy_im_list: list of noisy image, eachone as a 3D-array (H, W, T)
    """

    image_list = [np.array(tifffile.imread(bytes_data)) for bytes_data in input_list]

    for image in image_list:
        assert image.ndim == 3

    return image_list


###################################################################################################

# One function to read all different input types

def read_inputs(input_list_raw):
    """
    Main function for reading the input.
    It will try reading input of every form until it finds the correct one or fails.
    @args:
        - input_list_raw: a list of streamlit UploadedFile objects
    @requirements:
        - input_list_raw != []
    @returns:
        - noisy_im_list: list of noisy image, eachone as a 3D-array (H, W, T)
    """
    # save names for later
    noisy_im_names = [x.name for x in input_list_raw]
    # get extension to select how to read. Assumes all files have same extension. 
    # TODO: add this check
    file_ext = pathlib.Path(noisy_im_names[0]).suffix.lower()
    # convert raw data to bytes data
    input_list = [BytesIO(x.read()) for x in input_list_raw]

    if file_ext == ".tif" or file_ext == ".tiff":
        noisy_im_list = read_tiffs(input_list)
    else:
        raise FileTypeNotSupported(f"File type input not supported:{file_ext}")

    return noisy_im_names, noisy_im_list

###################################################################################################

# Other input variations

def set_dim(image, format_a):
    # Variation in axis order. (THW, HWT, etc)

    if format_a == "THW":
        return image
    
    return image.transpose(2, 0, 1)

def set_scale(image, format_d):
    # Variation in data type. (8-bit, 16-bit, etc)

    scale_max = 4096 if format_d == "16-bit" else 256
            
    return image/scale_max

def set_image(image, format_a, format_d):
    # Goes through all(2) variation funtions

    return set_dim(set_scale(image, format_d), format_a)
