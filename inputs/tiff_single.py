
from io import BytesIO
import numpy as np
import tifffile



class Tiff_Single():
    """
    Class for when inference is to be done on a single tiff file
    """

    def __init__(self):

        self.noisy_im = np.array([])

    def read_noisy_image(self, bytes_data):
        
        noisy_im = np.array(tifffile.imread(bytes_data))

        self.noisy_im = noisy_im

    def get_noisy_image(self):

        return self.noisy_im

    def get_noisy_shape(self):

        return self.noisy_im.shape