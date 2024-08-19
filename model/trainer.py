"""
File for finetuning cycle
Adjusted for display during streamlit session
"""
import json
import time
import torch
import numpy as np

from utils.utils import *
from model.model_cnnt import *
from torch.utils.data.dataloader import DataLoader
from model.running_inference import running_inference_per_image

import streamlit as st

# -------------------------------------------------------------------------------------------------

def train(model, config, train_set, val_set, device, num_workers, prefetch_factor):
    """
    Main function for training/finetuning loop
    @args:
        - model: a pytorch model
        - config: a namespace holding the configuration
        - train_set: pytorch dataset
        - val_set: list of np images similar to noisy images
        - device: device to train on
        - num_workers: worker for dataloader
        - prefetch_factor: prefetching for dataloader
    @returns:
        - model: the trained model
        - config: updated?
    """

    st.write("Setting up training cycle")

    # setup
    train_loader = []
    for idx, h in enumerate(config.height):
        train_loader.append(
            DataLoader(train_set[idx], shuffle=True, pin_memory=False, drop_last=False,
                        batch_size=config.batch_size, num_workers=num_workers,
                        prefetch_factor=prefetch_factor, persistent_workers=num_workers>0)
            )

    if config.dp and device=="cuda":
        model = torch.nn.DataParallel(model)

    model.to(device)

    if config.dp:
        scheduler = model.module.scheduler
        scheduler_on_batch = model.module.scheduler_on_batch
    else:
        scheduler = model.scheduler
        scheduler_on_batch = model.scheduler_on_batch

    st.write("Begin Training")
    
    # Place holders for proper visuals
    placeholder_1 = st.empty()
    placeholder_2 = st.empty()
    placeholder_3 = st.empty()

    with placeholder_1.container():
        st.write(f"Epoch:{0}/{config.num_epochs}, train_loss:--, val_loss:--, val_ssim_3D_loss:--, val_psnr:--, learning_rate:{config.global_lr:0.8f}")
        p_bar = st.progress(0)

    for epoch in range(config.num_epochs):
        model.train()

        train_loader_iter = []
        for h in range(len(config.height)):
            train_loader_iter.append(iter(train_loader[h]))

        total_num_batches = len(config.height) * len(train_loader[0])
        indices = np.arange(total_num_batches)
        np.random.shuffle(indices)

        with placeholder_2.container():
            st.write("Current epoch train loss:--")
            p_bar_2 = st.progress(0)

        for i, idx in enumerate(indices):
            loader_ind = idx % len(config.height)
            x, y = next(train_loader_iter[loader_ind])

            x = x.to(device)
            y = y.to(device)
            output = model(x)
            weights = None

            if config.dp:
                loss = model.module.compute_loss(output, y, weights, epoch, config)
            else:
                loss = model.compute_loss(output, y, weights, epoch, config)

            model.zero_grad()
            loss.backward()

            if(config.clip_grad_norm>0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

            model.module.optim.step() if config.dp else model.optim.step()

            if (scheduler is not None) and scheduler_on_batch:
                scheduler.step()
                curr_lr = scheduler.get_last_lr()[0]
            else:
                curr_lr = scheduler.optimizer.param_groups[0]['lr']

            with placeholder_2.container():
                st.write(f"Current epoch train loss:{loss.item():0.6f}")
            p_bar_2.progress((i+1)/len(indices))

        if (scheduler is not None) and (scheduler_on_batch == False):
            if(isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                scheduler.step(loss.item())
            else:
                scheduler.step()

        if epoch % config.save_cycle == 0:
            torch.cuda.empty_cache()
            val_loss, val_ssim3D_loss, val_psnr_val = run_val(model, config, val_set, device, placeholder_2, placeholder_3)
            torch.cuda.empty_cache()
            model.module.save(epoch) if config.dp else model.save(epoch)

        with placeholder_1.container():
            st.write(f"Epoch:{epoch+1}/{config.num_epochs}, train_loss:{loss.item():0.4f}, val_loss:{val_loss:0.4f}, val_ssim_3D_loss:{val_ssim3D_loss:0.4f}, val_psnr:{val_psnr_val:0.4f}, learning_rate:{curr_lr:0.6f}")
        p_bar.progress((epoch+1)/config.num_epochs)

    with placeholder_2.container():
        st.write("Saving Model")
        save_cnnt_model(model, config, height=config.height[-1], width=config.width[-1])
    with placeholder_2.container():
        st.write("Model Saved")

    return model, config

# -------------------------------------------------------------------------------------------------
# Run inference for val image
def run_val(model, config, val_set, device, placeholder_2, placeholder_3):

    #TODO: calibrate on machine setup
    cutout = (16, 128, 128)
    overlap = (4, 32, 32)
    batch_size = 4

    model.eval()

    noisy_image, clean_image = val_set[0].to(device), val_set[1].to(device)

    ssim3D_loss_func = Weighted_SSIM3D_Complex_Loss(device=device)
    psnr_func = PSNR()

    with placeholder_2.container():
        st.write("Running Validation Inference")
        placeholder_temp = st.empty()

    clean_pred_numpy, clean_pred_torch = running_inference_per_image(model, noisy_image, cutout, overlap, batch_size, device=device, placeholder=placeholder_temp)

    placeholder_temp.empty()
    placeholder_2.empty()

    clean_pred_torch = normalize_image(clean_pred_torch, values=(0,1), clip=True)
    loss = model.module.compute_loss(output=clean_pred_torch, targets=clean_image, weights=None).item() if config.dp else \
           model.compute_loss(output=clean_pred_torch, targets=clean_image, weights=None).item()
    ssim3D_loss = ssim3D_loss_func(clean_pred_torch, clean_image, weights=None).item()
    psnr_val = psnr_func(clean_pred_torch, clean_image).item()

    noisy_image_numpy = noisy_image.cpu().detach().numpy()[0,:,0]
    clean_image_numpy = clean_image.cpu().detach().numpy()[0,:,0]
    show_image_frames(noisy=noisy_image_numpy, predi=clean_pred_numpy, clean=clean_image_numpy, placeholder=placeholder_3)

    return loss, ssim3D_loss, psnr_val

# -------------------------------------------------------------------------------------------------
# Save the model as .pt as .pts in save dir set at start

def save_cnnt_model(model, config, last="", height=64, width=64):
    
    try:
        model = model.cpu().module
    except:
        model = model.cpu()

    model.eval()

    C = 1
    model_input = torch.randn(1, config.time, C, height, width, requires_grad=False)
    model_input = model_input.to('cpu')

    model_file_name = os.path.join(config.model_path_dir, config.model_file_name)
    model_file_name += last

    torch.save(model.state_dict(), f"{model_file_name}.pt")

    model_scripted = torch.jit.trace(model, model_input, strict=False) 
    model_scripted.save(f"{model_file_name}.pts")

    with open(f"{model_file_name}.json", "w") as file:
        json.dump(vars(config), file)

    return model_file_name

# -------------------------------------------------------------------------------------------------
# Show middle frames during train cycle

def show_image_frames(noisy, predi, clean, placeholder):

    frame_to_show = noisy.shape[0]//2
    
    noisy = normalize_image(noisy, percentiles=(1,99), clip=True)
    predi = normalize_image(predi, percentiles=(1,99), clip=True)
    clean = normalize_image(clean, percentiles=(1,99), clip=True)

    with placeholder.container():

        st.write("Result of inference during training (Noisy/Prediction/Clean)")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image=noisy[frame_to_show], caption="Noisy Image")

        with col2:
            st.image(image=predi[frame_to_show], caption="Predicted Image")

        with col3:
            st.image(image=clean[frame_to_show], caption="Clean Image")
