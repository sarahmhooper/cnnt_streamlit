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

from model.model_variations import load_model
from model.model_variations import model_list_from_dir
from model.running_inference import running_inference
from model.microscopy_dataset import MicroscopyDataset

from utils.utils import *
from model.trainer import train

# -------------------------------------------------------------------------------------------------

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
        self.model_name = None
        self.model_path = None

        self.args = args
        self.device = args.device
        self.model_path_dir = args.model_path_dir

    def get_model_list(self):
        # Retrieve the possible models from given model path directory

        return model_list_from_dir(self.model_path_dir)

    def set_config_update_dict(self, config_update_dict):
        # Save the config update to setup the model later

        config_update_dict["model_path_dir"] = self.model_path_dir
        config_update_dict["check_path"] = self.check_path
        config_update_dict["dp"] = self.device == "cuda" and torch.cuda.device_count() > 1
        config_update_dict["device"] = self.device
        self.config_update_dict = config_update_dict

    def load_model(self, model_name, model_files, is_inf=True):

        model_path = os.path.join(self.model_path_dir, model_name)
        self.model, self.config = load_model(model_path=model_path, model_files=model_files, config_update_dict={}, device=self.device)
        self.model_name = model_name

    def is_model_loaded(self):

        return self.model is not None

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

        return running_inference(self.model, noisy_image, self.args.cutout, self.args.overlap, self.device)

    def run_finetuning(self, train_set, val_set):
        # Run the finetuning cycle and update the model and config
        
        model, config = train(self.model, self.config, train_set, val_set, self.device, self.args.num_workers, self.args.prefetch_factor)
        self.model = model
        self.config = config

        return model, config
