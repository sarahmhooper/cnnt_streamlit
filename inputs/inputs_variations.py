"""
Main file for reading all different types of input images and
handling input variations like axis order and data types

Currently supported types:
    - .tif/.tiff

(Ideally) Only need to update this file for new input variations
TODO: more input types: .ometiff
"""

import pathlib
import tifffile
import numpy as np

from io import BytesIO
from utils.utils import *
from readlif.reader import LifFile

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

# lif reading helpers start

def iter_t(image, name, c, m, z):

    return [np.array([np.array(im_x) for im_x in image.get_iter_t(z=z, c=c, m=m)])], [name]

def iter_z(image, name, c, m):
    
    if image.dims.z == 1:
        return iter_t(image, name, c, m, 0)

    if image.dims.t == 1:
        return [np.array([np.array(im_x) for im_x in image.get_iter_z(t=0, c=c, m=m)])], [name]

    tuple_list = [iter_t(image, f"{name}_Z_{z}", c, m, z) for z in range(image.dims.z)]
    image_list = flatten([x[0] for x in tuple_list])
    names_list = flatten([x[1] for x in tuple_list])

    return image_list, names_list

def iter_m(image, name, c):
    
    if image.dims.m == 1:
        return iter_z(image, name, c, 0)

    tuple_list = [iter_z(image, f"{name}Tile_{nm}", c, nm) for nm in range(image.dims.m)]
    image_list = flatten([x[0] for x in tuple_list])
    names_list = flatten([x[1] for x in tuple_list])

    return image_list, names_list

def iter_c(image, name):

    if image.channels == 1:
        return iter_m(image, name, 0)

    tuple_list = [iter_m(image, f"{name}Channel_{nc}/", nc) for nc in range(image.channels)]
    image_list = flatten([x[0] for x in tuple_list])
    names_list = flatten([x[1] for x in tuple_list])

    return image_list, names_list

# lif reading helpers end

def read_lifs(input_list, lif_names):
    """
    read the lif file input consisting of nD images
    @args:
        - file_list: a list of LifFile objects
        - lif_names: names of the lif files
    @requirements:
        - byte_data objects should be in lif format
        - all images in the file should have the same format and dimensions
    @returns:
        - noisy_im_list: list of noisy image, eachone as a 3D-array (H, W, T)
        - noisy_im_names: names of the noisy images, taken from the lif file
    """
    file_list = [LifFile(x) for x in input_list]

    for i, file in enumerate(file_list):
        
        tuple_list = [iter_c(image, f"{lif_names[i]}/{image.name}_") for image in file.get_iter_image()]
        noisy_im_list = flatten([x[0] for x in tuple_list])
        noisy_im_names = flatten([x[1] for x in tuple_list])

    return noisy_im_list, noisy_im_names

###################################################################################################

# Infer input formats

def infer_format_a(image):
    """
    Infer the axis format of the given image. THW, vs HWT
    @args:
        - image: n-D array to infer dim from 
    @requirements:
        - only supports 3D arrays for now
    @returns:
        - format_a: inferred format of the axis
    """

    dims = image.shape

    assert image.ndim == 3

    format_a = "THW"
    if np.argmin(dims) == 0 : format_a = "THW"
    if np.argmin(dims) == 2 : format_a = "HWT"

    return format_a

def infer_format_d(image):
    """
    Infer the data format of the given image. 8-bit, vs 16-bit
    @args:
        - image: n-D array to infer from 
    @requirements:
        - only supports 8-bit or 16-bit data
    @returns:
        - format_d: inferred format of the data
    """

    format_d = "16-bit"
    if image.dtype == np.uint16 : format_d = "16-bit"
    if image.dtype == np.uint8 : format_d = "8-bit"

    return format_d

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
    elif file_ext == ".lif":
        noisy_im_list, noisy_im_names = read_lifs(input_list, noisy_im_names)
    else:
        raise FileTypeNotSupported(f"File type input not supported:{file_ext}")

    return noisy_im_names, noisy_im_list, infer_format_a(noisy_im_list[0]), infer_format_d(noisy_im_list[0])

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
