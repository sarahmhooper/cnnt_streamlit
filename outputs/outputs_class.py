"""
File for output class

Supports 3D image plotting
"""

from utils.utils import normalize_image
from outputs.plotting_options import plot_three
from outputs.outputs_variations import download_files

class Outputs_Class():
    """
    Class for holding finetuned model and noisy and clean images

    Provides options to download model and plot images for visual testing
    """


    def __init__(self):
        # Images are of format THW
        
        self.noisy_im_list = []
        self.noisy_im_names = []
        self.clean_im_list = []
        self.cpred_im_list = []

    def set_model(self, model, config, infer_func, scale):

        self.model = model
        self.config = config
        self.infer_func = infer_func
        self.scale = scale

    def set_lists(self, noisy_im_list, noisy_im_names, clean_im_list):
        # set given lists
        # TODO: some checks to make sure the images correspond correctly

        self.noisy_im_list = noisy_im_list
        self.noisy_im_names = noisy_im_names
        self.clean_im_list = clean_im_list
        self.cpred_im_list = [None for _ in range(len(noisy_im_list))]

    # Not needed for finetuning
    # def set_download_params(self, format_a, format_d, d_typ, d_ind, d_for, format_a_org, format_d_org):
    #     # set up download params for the download button

    #     self.format_a = format_a_org if format_a=="Same as input" else format_a
    #     self.format_d = format_d_org if format_d=="Same as input" else format_d
    #     self.d_typ = d_typ
    #     self.d_ind = d_ind
    #     self.d_for = d_for

    def prepare_download(self, d_typ):
        # prepare the download button
        download_files(self.model, self.config, d_typ)

    def plot_image(self, index):
        # Given index, plot the pair of noisy, pred, and clean images

        name = self.noisy_im_names[index]

        noisy_im = normalize_image(self.noisy_im_list[index], values=(0, self.scale), clip=True)
        clean_im = normalize_image(self.clean_im_list[index], values=(0, self.scale), clip=True)

        if self.cpred_im_list[index] is not None:
            cpred_im = self.cpred_im_list[index]
        else:
            cpred_im = (self.infer_func([noisy_im]))[0]
            self.cpred_im_list[index]=cpred_im

        plot_three(name, noisy_im, cpred_im, clean_im)
