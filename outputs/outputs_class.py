"""
File for output class

Supports only 3D images for now

TODO: Support nD images (n=3,4,5)
"""

from outputs.outputs_variations import set_image
from outputs.outputs_variations import download_files
from outputs.plotting_options import plot_pair


class Outputs_Class():
    """
    Class for holding outputs and ways to access

    Provides options to download and plot images for visual testing
    """


    def __init__(self):
        # Images are of format THW
        
        self.noisy_im_list = []
        self.noisy_im_names = []
        self.cpred_im_list = []

    def set_lists(self, noisy_im_list, noisy_im_names, cpred_im_list):
        # set given lists
        # TODO: some checks to make sure the images correspond correctly

        self.noisy_im_list = noisy_im_list
        self.noisy_im_names = noisy_im_names
        self.cpred_im_list = cpred_im_list

    def set_download_params(self, format_a, format_d, d_typ, d_ind, d_for, format_a_org, format_d_org):
        # set up download params for the download button

        self.format_a = format_a_org if format_a=="Same as input" else format_a
        self.format_d = format_d_org if format_d=="Same as input" else format_d
        self.d_typ = d_typ
        self.d_ind = d_ind
        self.d_for = d_for

    def prepare_download(self):
        # once params are set, prepare the download button

        image_list = [self.cpred_im_list[self.d_ind]] if self.d_typ!="Download all as zip" else self.cpred_im_list
        image_list = [set_image(x, self.format_a, self.format_d) for x in image_list]

        download_files(image_list, self.noisy_im_names, self.d_for)

    def plot_image(self, index):
        # Given index, plot the pair of noisy and clean images

        plot_pair(self.noisy_im_names[index], self.noisy_im_list[index], self.cpred_im_list[index])
