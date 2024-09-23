"""
File for the input class

Holds the given noisy and clean images

Supports loading multiple dimensions (See inputs_variations.py for detail on loading)
Stores all as 3D images after loading
"""

from inputs.input_variations import read_inputs
from utils.utils import normalize_image, register_translation_3D
import numpy as np

class Input_Class():
    """
    Class for holding the inputs given and provides access to them

    Allows access to indvidual/all images and names.
    """

    def __init__(self):
        # Init with empty

        self.noisy_im_names = None
        self.noisy_im_list = None
        self.clean_im_names = None
        self.clean_im_list = None
        self.predi_im_list = None

        self.im_value_scale = None
        self.register_image_check = False

        # set datatype to keep track of what images are loaded to avoid loading again
        self.read_noisy_im_names = {*{}}
        self.read_clean_im_names = {*{}}

    def read_input_files(self, noisy_list_raw, clean_list_raw=None):
        # Read noisy and clean using the one function
        # can maybe return non 0 for error handling

        if len(noisy_list_raw) == 0: 0

        # filter read images
        noisy_list_filtered = [x for x in noisy_list_raw if x.name not in self.read_noisy_im_names]
        if len(noisy_list_filtered):
            self.noisy_im_names, self.noisy_im_list = read_inputs(noisy_list_filtered)
            self.read_noisy_im_names.update([x.name for x in noisy_list_filtered])
            self.predi_im_list = [None for _ in self.noisy_im_list]

        if len(clean_list_raw):
            clean_list_filtered = [x for x in clean_list_raw if x.name not in self.read_clean_im_names]
            if len(clean_list_filtered):
                self.clean_im_names, self.clean_im_list = read_inputs(clean_list_filtered)
                self.read_clean_im_names.update([x.name for x in clean_list_filtered])

                assert np.all([clean_im.shape==noisy_im.shape for clean_im,noisy_im in zip(self.noisy_im_list,self.clean_im_list)]), "All paired noisy and clean images do not have the same shape. Please start a new session and ensure each noisy/clean pair has the same image shape."

        return 0

    def get_num_images(self):
        # Total number of images
        return len(self.noisy_im_list) if self.noisy_im_list is not None else 0

    def set_predi_im_idx(self, predi_im, idx):

        self.predi_im_list[idx] = predi_im

    def set_predi_im_list(self, predi_im_list):

        self.predi_im_list = predi_im_list

    def scale_images(self):

        # TODO: add parellism for faster execution
        self.noisy_im_list = [normalize_image(noisy_im, values=self.im_value_scale, clip=True) for noisy_im in self.noisy_im_list]
        if self.clean_im_list is not None:
            self.clean_im_list = [normalize_image(clean_im, values=self.im_value_scale, clip=True) for clean_im in self.clean_im_list]

    def register_images(self):

        # TODO: add parellism for faster execution
        if self.register_image_check:
            self.noisy_im_list = [register_translation_3D(noisy_im, clean_im) for noisy_im, clean_im in zip(self.noisy_im_list, self.clean_im_list)]
