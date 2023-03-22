"""
File for the model class

Hold items related to the model:
- name and directory
- model itself
- config
"""
import os
import torch
import random
import numpy as np

from model.models_variations import load_model
from model.models_variations import filter_f
from model.running_inference import running_inference
from model.microscopy_dataset import MicroscopyDataset

from utils.utils import *
from model.trainer import train


class Model_Class():
    """
    Model class for background model finetuning

    Loads the model
    Provides ways to setup and load model
    Provides method to finetune the model
    """

    def __init__(self, args):
        # save path_dir given when starting the streamlit server
        # save check_path for saving checkpoints

        self.model = None
        self.config = None
        self.model_path = None

        self.device = args.device
        self.cutout = args.cutout
        self.overlap = args.overlap
        self.model_path_dir = args.model_path_dir
        self.check_path = args.check_path
        self.num_workers = args.num_workers
        self.prefetch_factor = args.prefetch_factor

    def get_model_list(self):
        # Retrieve the possible models from given model path directory

        return filter_f(self.model_path_dir)

    def set_model_path(self, model_name, model_files):
        # Given model name, set the model path. Load later
        if model_files == []:
            model_path = os.path.join(self.model_path_dir, model_name)
            self.model_files = []
            self.model_path = model_path
        else:
            self.model_files = model_files
            self.model_path = None

    def set_config_update_dict(self, config_update_dict):
        # Save the config update to setup the model later

        config_update_dict["model_path_dir"] = self.model_path_dir
        config_update_dict["check_path"] = self.check_path
        config_update_dict["dp"] = self.device == "cuda" and torch.cuda.device_count() > 1
        config_update_dict["device"] = self.device
        self.config_update_dict = config_update_dict

    def load_model(self):
        # Load model before inference

        self.model, self.config = load_model(model_path=self.model_path, model_files=self.model_files, config_update_dict=self.config_update_dict)

    def prepare_train_n_val(self, noisy_im_list, clean_im_list, scale):
        # Prepare train and val sets

        train_set = []

        for h, w in zip(self.config.height, self.config.width):
            train_set.append(MicroscopyDataset(noisy_im_list=noisy_im_list, clean_im_list=clean_im_list,
                                                scale=scale, cutout_shape=(self.config.time, h, w),
                                                num_samples_per_file=self.config.num_samples_per_file)
            )

        indices = list(range(len(noisy_im_list)))
        random.shuffle(indices)
        def prepare_image(image):

            return torch.from_numpy(
                        (normalize_image(image, values=(0, scale), clip=True).astype(np.float32))[np.newaxis,:,np.newaxis]
                    )

        val_set = (prepare_image(noisy_im_list[indices[0]]),
                   prepare_image(clean_im_list[indices[0]]),)

        return train_set, val_set

    def run_inference(self, noisy_image):
        # Run inference on loaded model and given images
        # cut_np_images: 3D numpy images of axis order: THW

        return running_inference(self.model, noisy_image, self.cutout, self.overlap, self.device)

    def run_finetuning(self, train_set, val_set):
        # Run the finetuning cycle and update the model and config
        
        model, config = train(self.model, self.config, train_set, val_set, self.device, self.num_workers, self.prefetch_factor)
        self.model = model
        self.config = config

        return model, config
