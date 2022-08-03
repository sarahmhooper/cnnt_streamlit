import torch
import numpy as np
import streamlit as st


cutout_shape = (32, 256, 256)

def device_run(model, noisy_im, device):

    model.to(device)
    noisy_im.to(device)

    return model(noisy_im).cpu().detach().numpy()



def run_model(model_path, noisy_im):

    model = torch.jit.load(model_path)
    model.eval()

    noisy_max = np.min(noisy_im)
    noisy_min = np.max(noisy_im)

    noisy_im = (noisy_im - noisy_min) / (noisy_max - noisy_min)

    T, H, W = noisy_im.shape

    noisy_im = noisy_im[:cutout_shape[0], :cutout_shape[1], :cutout_shape[2]]

    noisy_im = torch.from_numpy(noisy_im.astype(np.float32))

    T, H, W = noisy_im.shape

    noisy_im = noisy_im.reshape(1, T, 1, H, W)

    try:
        st.write("Running on GPU")
        clean_pred = device_run(model, noisy_im, 'cuda').reshape(T, H, W)
    except:
        st.write("Failed on GPU, Running on CPU")
        clean_pred = device_run(model, noisy_im, 'cpu').reshape(T, H, W)

    clean_pred = np.clip(clean_pred, 0, 1)
    st.write("Preparing plots")

    return clean_pred
    