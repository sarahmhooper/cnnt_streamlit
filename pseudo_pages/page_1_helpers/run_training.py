

import streamlit as st

from utils.utils import *
from .microscopy_dataset import MicroscopyDataset
from .trainer import train

from inputs.input_class import Input_Class
from model.model_class import Model_Class
from outputs.output_class import Output_Class

ic : Input_Class = st.session_state.input_class
mc : Model_Class = st.session_state.model_class
oc : Output_Class = st.session_state.output_class

sst = st.session_state

def run_training_st():

    train_set, val_set = data_setup(mc.config, ic.noisy_im_list, ic.clean_im_list)

    model = train(mc.model, mc.config, train_set, val_set)
    mc.model = model

def data_setup(config, noisy_im_list, clean_im_list):
    # Prepare train and val sets

    train_set = []

    for h, w in zip(config.height, config.width):
        train_set.append(MicroscopyDataset(noisy_im_list=noisy_im_list, clean_im_list=clean_im_list,
                                            time_cutout=config.time, cutout_shape=(h, w),
                                            num_samples_per_file=8,
                                            im_value_scale=config.im_value_scale)
        )

    val_set = MicroscopyDataset(noisy_im_list=[noisy_im_list[0]], clean_im_list=[clean_im_list[0]],
                                time_cutout=config.time, cutout_shape=(h, w),
                                num_samples_per_file=1, test=True,
                                im_value_scale=config.im_value_scale)

    return train_set, val_set
