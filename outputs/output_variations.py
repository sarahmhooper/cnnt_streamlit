"""
Main file for downloading options and
variations like model type

Currently support downloading as:
- .pts
- .pt + .json

(Ideally) Only need to update this file for new input variations
"""

import io
import json
import torch
import zipfile
import tifffile
import numpy as np

import streamlit as st

# -------------------------------------------------------------------------------------------------
# Multiple output types

# def write_pts(model, config):
#     # as .pts
#     # TODO: fix this

#     try:
#         model = model.cpu().module
#     except:
#         model = model.cpu()

#     model.eval()

#     C = 1
#     model_input = torch.randn(1, config.time, C, config.height[0], config.width[0], requires_grad=False)
#     model_input = model_input.to('cpu')

#     model_scripted = torch.jit.trace(model, model_input, strict=False)
#     final_buffer = io.BytesIO()
#     torch.jit.save(model_scripted, final_buffer)

#     return final_buffer, f"{config.model_file_name}.pts"

# def write_pt_and_json(model, config):
#     # all as tiffs zipped

#     final_buffer = io.BytesIO()

#     with zipfile.ZipFile(final_buffer, "w") as myzip:
#         for i, image in enumerate(image_list):
#             temp_buff = io.BytesIO()
#             tifffile.imwrite(temp_buff, image)
#             myzip.writestr(f"{names_list[i]}_predicted.tiff", temp_buff.getvalue())

#         temp_buff = io.BytesIO()
#         torch.save(model.state_dict(), temp_buff)
#         myzip.writestr(f"{config.model_file_name}.pt", temp_buff.getvalue())

#         temp_buff = io.StringIO()
#         json.dump(vars(config), temp_buff)
#         myzip.writestr(f"{config.model_file_name}.json", temp_buff.getvalue())

#     return final_buffer, f"{config.model_file_name}.zip"

# -------------------------------------------------------------------------------------------------

# def download_files(model, config, format):

#     if format == ".pts":
#         final_buffer, file_name = write_pts(model, config)
#     elif format == ".pt + .json config as zip":
#         final_buffer, file_name = write_pt_and_json(model, config)
#     else:
#         raise NotImplementedError

#     st.download_button(
#         label="Download Model",
#         data = final_buffer, # Download buffer
#         file_name = f'{file_names[0]}_predicted.tiff' if d_one else 'Pred_Clean.zip' 
#     )

# -------------------------------------------------------------------------------------------------
# Image outputs

def write_tiff(image):
    # individual tiff

    final_buffer = io.BytesIO()
    tifffile.imwrite(final_buffer, image, metadata={'axes': 'TYX'})
    return final_buffer

def write_tiff_zip(image_list, names_list):
    # all as tiffs zipped

    final_buffer = io.BytesIO()
    buffer_list = []

    with zipfile.ZipFile(final_buffer, "w") as myzip:
        
        for i, image in enumerate(image_list):
            temp_buff = write_tiff(image)
            buffer_list.append(temp_buff)
            myzip.writestr(f"{names_list[i]}_predicted.tiff", temp_buff.getvalue())

    return final_buffer, buffer_list

# -------------------------------------------------------------------------------------------------

def write_model(model, config):

    final_buffer = io.BytesIO()

    torch.save({
        "epoch":config.num_epochs,
        "model_state": model.state_dict(),
        "config": config,
    }, final_buffer)

    return final_buffer

# -------------------------------------------------------------------------------------------------

def create_download_buffer(file_list, file_names, format):

    d_one = len(file_list)==1

    if format == ".tiff":
        final_buffer = write_tiff(file_list[0]) if d_one else write_tiff_zip(file_list, file_names)
    else:
        raise NotImplementedError

    return final_buffer

# -------------------------------------------------------------------------------------------------

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
