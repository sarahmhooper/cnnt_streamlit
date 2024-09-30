"""
File for output class

Holds buffers of the downloadable items
Avoid creating buffer everytime the page renders
"""

from outputs.output_variations import *

class Output_Class():
    """
    Class for holding output buffers
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
            self.image_names_list = []
            for name in names_list:
                if name[-1]=='/': name = name[:-1]
                name = name.replace('.lif','').replace('.tiff','').replace('.tif','')
                self.image_names_list += [f"{name}_predicted.tiff"]
                
        return self.image_all_buffer
    
    def prepare_download_model(self, model, config):

        if self.model_buffer is None:
            self.model_buffer = write_model(model, config)

        return self.model_buffer

    def get_download_image_idx(self, idx):

        return self.image_all_buffer[idx]
