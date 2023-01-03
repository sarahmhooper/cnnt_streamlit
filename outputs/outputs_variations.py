"""
Main file for downloading options and
variations like axis order and data type

Currently support downloading individual/all(.zip) as:
- .tiff

(Ideally) Only need to update this file for new input variations
#TODO: Support more types: .ometiff
"""

import io
import zipfile
import tifffile
import numpy as np

import streamlit as st

###################################################################################################

# Multiple output types

def write_tiff(image):
    # individual tiff

    final_buffer = io.BytesIO()
    tifffile.imwrite(final_buffer, image)
    return final_buffer

def write_tiff_zip(image_list, names_list):
    # all as tiffs zipped

    final_buffer = io.BytesIO()

    with zipfile.ZipFile(final_buffer, "w") as myzip:
        
        for i, image in enumerate(image_list):
            temp_buff = io.BytesIO()
            tifffile.imwrite(temp_buff, image)
            myzip.writestr(names_list[i], temp_buff.getvalue())

    return final_buffer

###################################################################################################

def download_files(file_list, file_names, format):

    d_one = len(file_list)==1

    if format == ".tiff":
        final_buffer = write_tiff(file_list[0]) if d_one else write_tiff_zip(file_list, file_names)
    else:
        raise NotImplementedError

    st.download_button(
        label="Download Predicted Clean Image(s)",
        data = final_buffer, # Download buffer
        file_name = f'{file_names[0]}' if d_one else 'Pred_Clean.zip' 
    )

###################################################################################################

# Other output variations

def set_dim(image, format_a):
    # Variation in axis order. (THW, HWT, etc)

    if format_a == "THW":
        return image
    
    return image.transpose(2, 0, 1)

def set_scale(image, format_d):
    # Variation in data type. (8-bit, 16-bit, etc)

    if format_d == "8-bit":
        return (np.clip(image, 0, 1)*256).astype(np.uint8)

    if format_d == "16-bit":
        return (np.clip(image, 0, 1)*4096).astype(np.uint16)

    return np.clip(image, 0, 1)

def set_image(image, format_a, format_d):
    # Goes through all(2) variation funtions

    return set_dim(set_scale(image, format_d), format_a)
