"""
Main file for downloading options and
variations like model type

Currently support downloading as:
- .pts
- .pt + .json

(Ideally) Only need to update this file for new input variations
"""

import io
import torch
import zipfile
import tifffile

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

            name = names_list[i]
            if name[-1]=='/': name = name[:-1]
            name = name.replace('.lif','').replace('.tiff','').replace('.tif','')
            
            myzip.writestr(f"{name}_predicted.tiff", temp_buff.getvalue())

    return final_buffer, buffer_list

# -------------------------------------------------------------------------------------------------
# Model outputs

def write_model(model, config):

    try:
        model = model.cpu().module
    except:
        model = model.cpu()

    final_buffer = io.BytesIO()

    torch.save({
        "epoch":config.num_epochs,
        "model_state": model.state_dict(),
        "config": config,
    }, final_buffer)

    return final_buffer
