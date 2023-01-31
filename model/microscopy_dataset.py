"""
File for microscopy dataset used during fine tuning
"""

import torch
import numpy as np

from utils.utils import *
from torch.utils.data import Dataset

from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

class MicroscopyDataset(Dataset):
    """
    Dataset for Microscopy.
    """

    def __init__(self, noisy_im_list, clean_im_list,
                scale, cutout_shape=[30, 128, 128],
                num_samples_per_file=8
                ):
        """
        Initialize the dataset
        @args:
            - noisy_im_list: list of noisy images
            - clean_im_list: list of GT images corresponding to the noisy ones
            - scale: scale to scale down the images with
            - cutout_shape: time, height, width cutout of train samples
            - num_samples_per_file: number of patches to cut from each file per epoch
        """

        self.cutout_shape = cutout_shape
        self.num_samples_per_file = num_samples_per_file

        self.N_images = len(noisy_im_list)

        self.start_samples = np.zeros(self.N_images)
        self.end_samples = np.zeros(self.N_images)

        self.start_samples[0] = 0
        self.end_samples[0] = num_samples_per_file

        for i in range(1, self.N_images):
            self.start_samples[i] = self.end_samples[i-1]
            self.end_samples[i] = self.start_samples[i] + num_samples_per_file

        self.noisy_im_list = [normalize_image(noisy_im, values=(0, scale), clip=True) for noisy_im in noisy_im_list]
        self.clean_im_list = [normalize_image(clean_im, values=(0, scale), clip=True) for clean_im in clean_im_list]

        # Register images to align up better
        self.noisy_im_list = [self.register_translation_3D(self.noisy_im_list[i], self.clean_im_list[i]) for i in range(len(self.noisy_im_list))]

    def load_one_sample(self, index):
        """
        Load one sample given the index
        @args:
            - index: the index to load from
        @returns:
            - noisy_im: list of noisy image
            - clean_im: list of GT images corresponding to the noisy one
        """
        noisy_im = []
        clean_im = []

        # get the image
        noisy_data = self.noisy_im_list[index]
        clean_data = self.clean_im_list[index]

        # pad symmetrically if not enough images in the time dimension
        if noisy_data.shape[0] < self.cutout_shape[0]:
            noisy_data = np.pad(noisy_data, ((0,self.cutout_shape[0] - noisy_data.shape[0]),(0,0),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0,self.cutout_shape[0] - clean_data.shape[0]),(0,0),(0,0)), 'symmetric')

        # random flip
        noisy_data, clean_data = self.random_flip(noisy_data, clean_data)

        if noisy_data.shape[1] < self.cutout_shape[1]:
            noisy_data = np.pad(noisy_data, ((0,0),(0,self.cutout_shape[1] - noisy_data.shape[1]),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0,0),(0,self.cutout_shape[1] - clean_data.shape[1]),(0,0)), 'symmetric')

        if noisy_data.shape[2] < self.cutout_shape[2]:
            noisy_data = np.pad(noisy_data, ((0,0),(0,0),(0,self.cutout_shape[2] - noisy_data.shape[2])), 'symmetric')
            clean_data = np.pad(clean_data, ((0,0),(0,0),(0,self.cutout_shape[2] - clean_data.shape[2])), 'symmetric')

        T, RO, E1 = noisy_data.shape

        if(RO>=self.cutout_shape[0] and E1>=self.cutout_shape[1]):

            noisy_cutout, clean_cutout = self.do_cutout(noisy_data, clean_data)

            noisy_im.append(torch.from_numpy(noisy_cutout[:,np.newaxis,:,:].astype(np.float32)))
            clean_im.append(torch.from_numpy(clean_cutout[:,np.newaxis,:,:].astype(np.float32)))

        return noisy_im, clean_im

    def do_cutout(self, noisy, clean):
        # Randomly cutout each iteration

        t, x, y = noisy.shape
        ct, cx, cy = self.cutout_shape

        st = np.random.randint(0, t - ct + 1) if t>ct else 0
        sx = np.random.randint(0, x - cx + 1) if x>cx else 0
        sy = np.random.randint(0, y - cy + 1) if y>cy else 0

        return noisy[st:st+ct, sx:sx+cx, sy:sy+cy], clean[st:st+ct, sx:sx+cx, sy:sy+cy]

    def random_flip(self, noisy, clean):
        # Randomly flip each iteration

        flip1 = np.random.randint(0,2) > 0
        flip2 = np.random.randint(0,2) > 0

        def flip(image):
            if flip1:
                image = image[:,::-1,:].copy()

            if flip2:
                image = image[:,:,::-1].copy()

            return image

        return flip(noisy), flip(clean)

    def find_sample(self, ind):
        # find the image index given dataset index
        ind_file = 0

        for i in range(self.N_images):
            if(ind>=self.start_samples[i] and ind<self.end_samples[i]):
                ind_file = i

        return ind_file

    def register_translation_3D(self, noisy, clean):
        # Register noisy clean pair using translation

        z_off, y_off, x_off = phase_cross_correlation(clean, noisy, return_error=False)

        return shift(noisy, shift=(z_off, y_off, x_off), mode="reflect")

    def __len__(self):

        return self.N_images * self.num_samples_per_file

    def __getitem__(self, ind):
        # 10 tries to find the best image patch

        sample_list = []
        count_list = []
        found = False

        for _ in range(10):
            ind_file = self.find_sample(ind)
            noisy_im, clean_im = self.load_one_sample(ind_file)

            sample = (noisy_im[0], clean_im[0])

            area_threshold = 0.5 * torch.prod(torch.as_tensor(sample[1].shape))

            if (torch.count_nonzero(sample[1] > 0.02) >= area_threshold):
                found = True
                break

            sample_list.append(sample)
            count_list.append(torch.count_nonzero(sample[1] > 0.02))

        if not found:
            sample = sample_list[count_list.index(max(count_list))]

        return sample
