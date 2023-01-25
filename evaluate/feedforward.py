import hydra
import torch
import omegaconf
from skimage.filters import gabor_kernel
import numpy as np
import logging
import matplotlib.pyplot as plt
import sys
import pandas as pd
from PIL import Image

import eval_utils

logger = logging.getLogger(__name__)

def pad_kernel(kernel, output_size=12):
    pad_width = output_size - kernel.shape[0]
    before = int(np.floor(pad_width/2))
    after = pad_width-before
    kernel_padded = np.pad(kernel, 
        (before, after), 
        mode='constant'
        )
    return kernel_padded

def trim_kernel(kernel, output_size=12, left=True):
    trim_width = kernel.shape[0] - output_size
    if trim_width > 1:
        # print(trim_width)
        if left:
            before = int(np.floor(trim_width/2))
        else:
            before = int(np.ceil(trim_width/2))
        after = trim_width-before
        # print(before, after)
        out = kernel[before:-after, before:-after]
        # print(out.shape)
        return out
    else:
        before = int(np.floor(trim_width/2))
        after = trim_width-before
        # print(before, after)
        out = kernel[before:-after, before:-after]
        # print(out.shape)
        return out



@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: omegaconf.OmegaConf) -> None:

    def savefig(figname):
        plt.savefig(cfg.paths.user_home_dir+f'/evaluate/figures/{figname}.png')
        return

    # Load model
    model = eval_utils.load_model(
        model_class='SVAE',
        model_path='outputs/SavedModels/SVAE/LAPLACE_e64_ll-2.0_lr1e-04/',
        device=torch.device('cpu'),
    )
    model_path = hydra.utils.to_absolute_path(cfg.eval.evaluate_ll.mdl_path)
    logger.info(f'Loaded: {model_path}')

    angles = np.linspace(0,180,20)
    frequencies = np.linspace(0.1,0.5,20)
    amplitudes = np.linspace(0.5,1.5,7)

    # Model neurons are the intersection of between the neurons with highest 
    # response norm for varying angle and freq
    model_neurons = [72, 38, 42, 82, 158] 
    
    fig, axs =plt.subplots(ncols=len(model_neurons))
    for i, ind in enumerate(model_neurons):
        axs[i].imshow(
            model.output_weights()[:,ind].reshape(12,12), 
            # vmin=torch.min(model.output_weights()), 
            # vmax=torch.max(model.output_weights())
            )
    savefig('features')
    plt.show()
    
    # Varying angle

    freq = 0.4

    vals = torch.zeros((len(angles), len(model_neurons)))
    for i, angle in enumerate(angles):
        theta = angle / 180 * np.pi

        kernel = gabor_kernel(freq, theta).real
        kernel_padded = pad_kernel(kernel)

        X = torch.as_tensor(kernel_padded, dtype=torch.float32).flatten()
        z = model._encode(X)

        vals[i] = torch.tensor([z[0].detach()[n_id] for n_id in model_neurons])

    fig, ax = plt.subplots()
    ax.plot(angles, torch.abs(vals))
    savefig('z_vals_angles')
    plt.show()
    del vals

    # Varying frequency

    theta = 30 / 180 * np.pi

    vals = torch.zeros((len(angles), len(frequencies), len(model_neurons)))
    for i, angle in enumerate(angles):
        theta = angle / 180 * np.pi
        for j, freq in enumerate(frequencies):

            kernel = gabor_kernel(freq, theta).real
            if kernel.shape[0] > 12:
                kernel = trim_kernel(kernel)
            elif kernel.shape[0] < 12:
                kernel = pad_kernel(kernel)

            X = torch.as_tensor(kernel, dtype=torch.float32).flatten()
            z = model._encode(X)

            vals[i,j] = torch.tensor([z[0].detach()[n_id] for n_id in model_neurons])

    break_points = [12,12,10,12,16]
    fig, ax = plt.subplots()

    optim_freq_angles = []
    for i, neuron_id in enumerate(model_neurons):
        optimal_angle_index = torch.argmax(torch.linalg.norm(vals[:,:,i], axis=1))
        optim_freq_angles.append([
            neuron_id, angles[optimal_angle_index], frequencies[torch.argmax(torch.abs(vals)[optimal_angle_index,:,i])]
        ])
        ax.plot(frequencies[:break_points[i]], torch.abs(vals)[optimal_angle_index,:break_points[i],i])
    savefig('z_vals_freqs')
    plt.show()
    del vals

    print(optim_freq_angles)

    # Vary amplitude

    freq = 0.4
    vals = torch.zeros(len(angles), len(amplitudes))

    for i, angle in enumerate(angles):
        for j, amplitude in enumerate(amplitudes):
            theta = angle / 180 * np.pi

            kernel = amplitude * gabor_kernel(freq, theta).real
            kernel_padded = pad_kernel(kernel)

            X = torch.as_tensor(kernel_padded, dtype=torch.float32).flatten()
            z = model._encode(X)

            vals[i,j] = z[0].detach()[model_neurons[0]]

    cmap = plt.get_cmap('RdBu')
    fig, ax = plt.subplots()
    for i in range(len(amplitudes)):
        ax.plot(angles, torch.abs(vals[:,i]), c=cmap(i/len(amplitudes)))
    savefig('variable_constrast')
    plt.show()

    # Vary image size 

    # kernel = gabor_kernel(0.22631578947368422, 9.473684210526315/180 * np.pi).real
    # if kernel.shape[0] < 12:
    #     kernel = pad_kernel(kernel)
    # elif kernel.shape[0] > 12:
    #     kernel = trim_kernel(kernel)
    # kernel = trim_kernel(kernel, output_size=8)

    # kernel_normalized = (kernel - np.amin(kernel))/np.amax((kernel - np.amin(kernel)))
    # im = Image.fromarray(255*kernel_normalized)
    
    # fig, axs = plt.subplots(ncols=2);
    # axs[0].imshow(im)
    # axs[1].imshow(im.resize((12,12)))
    # savefig('im')
    # plt.show()


    plt.figure();

    for i in range(len(model_neurons)):
        kernel = gabor_kernel(optim_freq_angles[i][2], optim_freq_angles[i][1]/180 * np.pi).real
        if kernel.shape[0] < 12:
            kernel = pad_kernel(kernel)
        elif kernel.shape[0] > 12:
            kernel = trim_kernel(kernel)

        vals = np.zeros(12)

        X = torch.as_tensor(kernel, dtype=torch.float32).flatten()
        z = model._encode(X)
        vals[0] = z[0].detach()[model_neurons[i]]
        for j in range(1,12):
            temp = []
            for left in [True, False]:
                sub_kernel = trim_kernel(kernel, output_size=12-j, left=True)
                # sub_kernel_normalized = (sub_kernel - np.amin(sub_kernel))/np.amax((sub_kernel - np.amin(sub_kernel)))
                im = Image.fromarray(sub_kernel)

                sized_kernel = np.asarray(im.resize((12,12)))
                # if sub_kernel.shape[0] < 12:
                #     sized_kernel = pad_kernel(sub_kernel)
                # elif sub_kernel.shape[0] > 12:
                #     sized_kernel = trim_kernel(sub_kernel)
                # else:
                #     sized_kernel = sub_kernel

                X = torch.as_tensor(sized_kernel, dtype=torch.float32).flatten()
                z = model._encode(X)
                temp.append(z[0].detach()[model_neurons[i]])

            vals[j] = np.mean(temp)

        plt.plot(np.abs(vals)[::-1])
    savefig('size')
    plt.show()
    
    # fig, axs = plt.subplots(ncols=2)
    # axs[0].imshow(kernel)
    # axs[1].imshow(kernel_padded)
    # savefig('kernel')
    # plt.show();



if __name__=='__main__':
    main()
