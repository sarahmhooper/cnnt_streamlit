"""
File for the input class

Holds the given noisy and clean images and attributes related to it:
- name
- scale

Supports loading multiple dimensions (See inputs_variations.py for detail on loading)
Stores all as 3D images after loading
"""

from inputs.input_variations import read_inputs

class Input_Class():
    """
    Class for holding the inputs given and provides access to them

    Hold the following variables:
        - noisy_im_names: the names of the noisy images given
        - noisy_im_list: the given noisy images. each noisy image is a 3D numpy array
        - clean_im_names: the names of the clean images given
        - clean_im_list: the given clean images. each clean image is a 3D numpy array
        - scale: that value to scale images with

    Allows access to indvidual/all images and names.
    """

    def __init__(self):
        # Init with empty

        self.noisy_im_names = None
        self.noisy_im_list = None
        self.clean_im_names = None
        self.clean_im_list = None
        self.predi_im_list = None

        self.read_noisy_im_names = {*{}}
        self.read_clean_im_names = {*{}}

    def read_input_files(self, noisy_list_raw, clean_list_raw=None):
        # Read noisy and clean using the one function

        if len(noisy_list_raw) == 0: 0

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

        return 0

    def get_num_images(self):
        # Total number of images
        return len(self.noisy_im_list) if self.noisy_im_list is not None else 0

    def set_predi_im_idx(self, predi_im, idx):

        self.predi_im_list[idx] = predi_im

    def set_predi_im_list(self, predi_im_list):

        self.predi_im_list = predi_im_list
