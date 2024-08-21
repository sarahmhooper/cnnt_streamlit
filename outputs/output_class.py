"""
File for output class

Supports 3D image plotting
"""

from outputs.output_variations import *

class Output_Class():
    """
    Class for holding finetuned model and noisy and clean images

    Provides options to download model and plot images for visual testing
    """
    def __init__(self):
        # Images are of format THW
        
        self.image_all_buffer = None
        self.image_idx_buffer = []
        self.image_names_list = []
        self.model_buffer = None

    def prepare_download_image_all(self, image_list, names_list):
        
        if self.image_all_buffer is None:
            self.image_all_buffer, self.image_idx_buffer = write_tiff_zip(image_list, names_list)
            self.image_names_list = [f"{name}_predicted.tiff" for name in names_list]
        return self.image_all_buffer

    def get_download_image_idx(self, idx):

        return self.image_all_buffer[idx]
