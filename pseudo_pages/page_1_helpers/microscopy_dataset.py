"""
Microscopy dataloader

Modified to take in lists of images on setup
"""

import torch
import numpy as np
from torch.utils.data import Dataset

from utils.utils import *

class MicroscopyDataset(Dataset):
    """
    Dataset for loading microscopy data from lists.
    """

    def __init__(self, noisy_im_list, clean_im_list,
                    time_cutout=30, cutout_shape=[64, 64],
                    num_samples_per_file=1, test=False,
                    per_scaling=False, im_value_scale=[0,4096],
                    valu_thres=0.002, area_thres=0.25):
        """
        Initilize the denoising dataset

        @args:
            - noisy_im_list (npy list): list of noisy images
            - clean_im_list (npy list): list of clean images
            - time_cutout (int): time cutout for the first dim
            - cutout_shape (2 tuple int): patch size. Defaults to [64, 64]
            - num_samples_per_file (int): number of patches to cut out per image
            - test (bool): whether this dataset is for evaluating test set. Does not make a cutout if True
            - per_scaling (bool): whether to use percentile scaling or scale with static values
            - im_value_scale (2 tuple int): the value to scale with
            - valu_thres (float): threshold of pixel value between background and foreground
            - area_thres (float): percentage threshold of area that needs to be foreground
        """
        self.noisy_im_list = noisy_im_list
        self.clean_im_list = clean_im_list

        self.time_cutout = time_cutout
        self.cutout_shape = cutout_shape

        self.num_samples_per_file = num_samples_per_file

        self.test = test
        self.per_scaling = per_scaling
        self.im_value_scale = im_value_scale
        self.valu_thres = valu_thres
        self.area_thres = area_thres

        if per_scaling:
            self.noisy_im_list = [normalize_image(noisy_data, percentiles=(1.5, 99.5), clip=True) for noisy_data in self.noisy_im_list]
            self.clean_im_list = [normalize_image(clean_data, percentiles=(1.5, 99.5), clip=True) for clean_data in self.clean_im_list]
        else:
            self.noisy_im_list = [normalize_image(noisy_data, values=im_value_scale, clip=True) for noisy_data in self.noisy_im_list]
            self.clean_im_list = [normalize_image(clean_data, values=im_value_scale, clip=True) for clean_data in self.clean_im_list]

    def load_one_sample(self, idx):
        """
        Loads one sample from the h5file and key pair for regular paired image
        """
        # get the image
        noisy_data = self.noisy_im_list[idx]
        clean_data = self.clean_im_list[idx]

        if self.test:
            noisy_cutout = noisy_data[:,np.newaxis,:,:]
            clean_cutout = clean_data[:,np.newaxis,:,:]

            noisy_im = torch.from_numpy(noisy_cutout.astype(np.float32))
            clean_im = torch.from_numpy(clean_cutout.astype(np.float32))

            return noisy_im, clean_im

        # pad symmetrically if not enough images in the time dimension
        if noisy_data.shape[0] < self.time_cutout:
            noisy_data = np.pad(noisy_data, ((0,self.time_cutout - noisy_data.shape[0]),(0,0),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0,self.time_cutout - clean_data.shape[0]),(0,0),(0,0)), 'symmetric')

        # random flip
        noisy_data, clean_data = self.random_flip(noisy_data, clean_data)

        if noisy_data.shape[1] < self.cutout_shape[0]:
            noisy_data = np.pad(noisy_data, ((0, 0), (0,self.cutout_shape[0] - noisy_data.shape[1]),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0, 0), (0,self.cutout_shape[0] - clean_data.shape[1]),(0,0)), 'symmetric')

        if noisy_data.shape[2] < self.cutout_shape[1]:
            noisy_data = np.pad(noisy_data, ((0,0), (0,0), (0,self.cutout_shape[1] - noisy_data.shape[2])), 'symmetric')
            clean_data = np.pad(clean_data, ((0,0), (0,0), (0,self.cutout_shape[1] - clean_data.shape[2])), 'symmetric')

        T, RO, E1 = noisy_data.shape

        if(RO>=self.cutout_shape[0] and E1>=self.cutout_shape[1]):
            # define a set of cut range
            s_x, s_y, s_t = self.get_cutout_range(noisy_data)

            noisy_cutout = self.do_cutout(noisy_data, s_x, s_y, s_t)[:,np.newaxis,:,:]
            clean_cutout = self.do_cutout(clean_data, s_x, s_y, s_t)[:,np.newaxis,:,:]

            noisy_im = torch.from_numpy(noisy_cutout.astype(np.float32))
            clean_im = torch.from_numpy(clean_cutout.astype(np.float32))

        return noisy_im, clean_im

    def get_cutout_range(self, data):
        """
        get the starting positions of cutouts
        """
        t, x, y = data.shape
        ct = self.time_cutout
        cx, cy = self.cutout_shape

        s_t = np.random.randint(0, t - ct + 1)
        s_x = np.random.randint(0, x - cx + 1)
        s_y = np.random.randint(0, y - cy + 1)

        return s_x, s_y, s_t

    def do_cutout(self, data, s_x, s_y, s_t):
        """
        Cuts out the patches
        """
        T, x, y = data.shape
        ct = self.time_cutout
        cx, cy = self.cutout_shape

        if T < ct or y < cy or x < cx:
            raise RuntimeError(f"File is borken because {T} is less than {ct} or {x} is less than {cx} or {y} is less than {cy}")

        return data[s_t:s_t+ct, s_x:s_x+cx, s_y:s_y+cy]

    def random_flip(self, noisy, clean):

        flip1 = np.random.randint(0,2) > 0
        flip2 = np.random.randint(0,2) > 0

        def flip(image):
            if image.ndim == 2:
                if flip1:
                    image = image[::-1,:].copy()

                if flip2:
                    image = image[:,::-1].copy()
            else:
                if flip1:
                    image = image[:,::-1,:].copy()

                if flip2:
                    image = image[:,:,::-1].copy()

            return image

        return flip(noisy), flip(clean)

    def find_sample(self, index):
        ind_file = 0

        for i in range(self.N_files):
            if(index>= self.start_samples[i] and index<self.end_samples[i]):
                ind_file = i
                ind_in_file = int(index - self.start_samples[i])//self.num_samples_per_file
                break

        return ind_file, ind_in_file

    def __len__(self):

        return len(self.noisy_im_list) * self.num_samples_per_file

    def __getitem__(self, idx):

        #print(f"{idx}")
        sample_list = []
        count_list = []
        found = False

        # iterate 10 times to find the best sample
        for i in range(10):

            sample = self.load_one_sample(idx//self.num_samples_per_file)

            # The foreground content check
            valu_score = torch.count_nonzero(sample[1] > self.valu_thres)
            area_score = self.area_thres * sample[1].numel()
            if (valu_score >= area_score):
                found = True
                break

            sample_list.append(sample)
            count_list.append(valu_score)

        if not found:
            sample = sample_list[count_list.index(max(count_list))]

        return sample
