"""
Main file for reading all different types of input images

Currently supported types:
    - .tif/.tiff
    - .lif
"""

import pathlib
import tifffile
import numpy as np

from io import BytesIO
from utils.utils import *
from readlif.reader import LifFile

# -------------------------------------------------------------------------------------------------
# Multiple input types

def read_tiffs(input_list):
    """
    read the tiff file input
    @args:
        - input_list: a list of byte_data objects
    @rets:
        - noisy_im_list: list of noisy image, eachone as a 3D-array
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
    @rets:
        - im_list: list of noisy image, eachone as a 3D-array (H, W, T)
        - im_names: names of the noisy images, taken from the lif file
    """
    file_list = [LifFile(x) for x in input_list]

    im_list = []
    im_names = []

    for i, file in enumerate(file_list):
        
        tuple_list = [iter_c(image, f"{lif_names[i]}/{image.name}_") for image in file.get_iter_image()]
        im_list.extend(flatten([x[0] for x in tuple_list]))
        im_names.extend(flatten([x[1] for x in tuple_list]))

    return im_list, im_names

# -------------------------------------------------------------------------------------------------
# Sort lists so one to one mapping

def sort_list(images, names):
    """
    Sorting images by name. zips, sorts, unzips
    @args:
        - images: the list of images
        - names: the list of names
    @rets:
        - sorted_images: the sorted list of images
        - sorted_names: the sorted list of names
    """
    sorted_l = sorted(zip(names, images), key=lambda x:x[0])

    sorted_images = [x[1] for x in sorted_l]
    sorted_names = [x[0] for x in sorted_l]

    return sorted_images, sorted_names

# -------------------------------------------------------------------------------------------------

# One function to read all different input types

def read_inputs(input_list_raw):
    """
    Main function for reading the input.
    @args:
        - input_list_raw: a list of streamlit UploadedFile objects
    @rets:
        - noisy_im_list: list of noisy image, eachone as a 3D-array (T, H, W)
    """
    # save names for later
    im_names = [x.name for x in input_list_raw]
    # get extension to select how to read. Assumes all files have same extension. 
    file_ext = pathlib.Path(im_names[0]).suffix.lower()

    all_file_exts = [pathlib.Path(im_name).suffix.lower() for im_name in im_names]
    assert len(set(all_file_exts))==1, "All files must have same filetype."
    
    # convert raw data to bytes data
    input_list = [BytesIO(x.read()) for x in input_list_raw]

    if file_ext == ".lif":
        im_list, im_names = read_lifs(input_list, im_names)
    else:
        try:
            im_list = read_tiffs(input_list)
        except:
            raise FileTypeNotSupported(f"File type input not supported:{file_ext}", file_ext)
    
    return im_names, im_list
