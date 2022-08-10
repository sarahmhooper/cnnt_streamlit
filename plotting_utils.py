"""
Plotting utils for CNNT_streamlit

Seperate file for plotting images
"""
import os
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def torch_2_numpy(image):

    try:
        return image[0,:,0,:,:].cpu().detach().numpy()
    except:
        try:
            return image[0,:,0,:,:]
        except:
            return image

#------------------------------------------------
# Plotting functions
#------------------------------------------------

def plot_1_image(image, cmap='gray', path=None, id=None, show=False):
    """
    Plot given 2D+T image as an animation
    @inputs:
        image: numpy array of shape [T, H, W], dtype = float
        cmap: color pellete to use
        path: if not None, will save on that path
        id: optional identifier to save the image with
        show: if True, will show the animated image (will pause the execution)
    """

    image = torch_2_numpy(image)

    fig = plt.figure()
    
    def init():
        return plt.imshow(image[0],cmap=cmap,animated=True) # start with 0th frame

    def update(i):
        return plt.imshow(image[i],cmap=cmap,animated=True) # ith frame for ith timestamp

    ani = animation.FuncAnimation(fig, update, init_func = init, frames = image.shape[0],
                                    interval = 200, repeat_delay=2000)
    
    if show:
        plt.show()
    if path != None:
        print(f"Saving plot at {path}")
        save_path = os.path.join(path, f'{id}.mp4')
        print(f"Saving video at {save_path}")
        ani.save(save_path, writer = 'ffmpeg')

    plt.close()

def plot_images(images, titles, shape, cmap='gray', path=None, id=None, show=False):
    """
    General function to plot a given number of images
    @inputs:
        images: numpy/torch list of images to plot
        titles: string list, the title of each image
        shape: 2-tuple, rows and column of plotting shape (len(images) == shape[0]*shape[1])
        cmap: color pellete to use
        path: if not None, will save on that path
        id: optional identifier to save the image with
        show: if True, will show the animated image (will pause the execution)
    """
    # convert to 2D+T numpy object
    images = [torch_2_numpy(image) for image in images]

    fig, axs = plt.subplots(shape[0],shape[1], figsize = (shape[1]*4,shape[0]*4))

    for i in range(shape[0]):
        for j in range(shape[1]):

            if shape[0]>1:
                axs[i,j].set_title(titles[i*shape[1]+j])
            else:
                axs[j].set_title(titles[i*shape[1]+j])

    def init():

        final = []

        for i in range(shape[0]):
            for j in range(shape[1]):
                # start with 0th frame

                if shape[0]>1:
                    axs[i,j].imshow(images[i*shape[1]+j][0],cmap=cmap,animated=True)

                    final.append(axs[i,j])
                else:
                    axs[j].imshow(images[i*shape[1]+j][0],cmap=cmap,animated=True)

                    final.append(axs[j])

        return tuple(final)

    def update(n):
        # nth frame for nth timestamp
        final = []

        for i in range(shape[0]):
            for j in range(shape[1]):
                
                if shape[0]>1:
                    axs[i,j].imshow(images[i*shape[1]+j][n],cmap=cmap,animated=True)

                    final.append(axs[i,j])
                else:
                    axs[j].imshow(images[i*shape[1]+j][n],cmap=cmap,animated=True)

                    final.append(axs[j])

        return tuple(final)

    ani = animation.FuncAnimation(fig, update, init_func = init, frames = images[0].shape[0],
                                    interval = 200, repeat_delay=2000)

    if show:
        plt.show()
    if path != None:
        print(f"Saving plot at {path}")
        save_path = os.path.join(path, f'{id}.mp4')
        ani.save(save_path)
    
    plt.close()


#------------------------------------------------
# Wrappers for easy access
#------------------------------------------------

def plot_2_images(images, titles=["input", "output"], path=None, id=None, show=True):

    plot_images(images, titles, (1,2), path=path, id=id, show=show)

def plot_3_images(images, titles=["noisy", "pred", "clean"], path=None, id=None, show=True):

    plot_images(images, titles, (1,3), path=path, id=id, show=show)

def plot_6_images(images, titles=["noisy", "pred", "clean", "added_noise", "pred_noise", "gmap"], path=None, id=None, show=True):

    plot_images(images, titles, (2,3), path=path, id=id, show=show)