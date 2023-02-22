import hydra
import torch
import omegaconf
from skimage.filters import gabor_kernel
import numpy as np
import logging
import matplotlib
import matplotlib.pyplot as plt
import sys
import pandas as pd
from PIL import Image
import scipy as sp 
import pathlib
from matplotlib.gridspec import GridSpec

import eval_utils

logger = logging.getLogger(__name__)

def pad_kernel(kernel, output_size=12, left=True):
    pad_width = output_size - kernel.shape[0]
    if left or pad_width == 1:
        before = int(np.floor(pad_width/2))
    else:
        before = int(np.ceil(pad_width/2))
    after = pad_width-before
    kernel_padded = np.pad(kernel, 
        (before, after), 
        mode='constant'
        )
    return kernel_padded

def trim_kernel(kernel, output_size=12, left=True):
    trim_width = kernel.shape[0] - output_size
    if left or trim_width == 1:
        before = int(np.floor(trim_width/2))
    else:
        before = int(np.ceil(trim_width/2))
    # if trim_width > 1:
    #     if left:
    #         before = int(np.floor(trim_width/2))
    #     else:
    #         before = int(np.ceil(trim_width/2))
    #     after = trim_width-before
    #     out = kernel[before:-after, before:-after]
    #     return out
    # else:
    #     before = int(np.floor(trim_width/2))
    after = trim_width-before
    # print(before, after)
    out = kernel[before:-after, before:-after]
    # print(out.shape)
    return out

def get_kernel(freq, theta, component='imag', left=True):
    if component == 'imag':
        kernel = gabor_kernel(freq, theta).imag
    else:
        kernel = gabor_kernel(freq, theta).real
    
    # Format kernel to 12x12
    if kernel.shape[0] > 12:
        kernel = trim_kernel(kernel, left=left)
    elif kernel.shape[0] < 12:
        kernel = pad_kernel(kernel, left=left)
    return kernel



@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: omegaconf.OmegaConf) -> None:

    def savefig(figname):
        plt.savefig(cfg.paths.user_home_dir+f'/evaluate/figures/{figname}.png', dpi=300)
        return

    # Load model
    model = eval_utils.load_model(
        model_class='SVAE',
        model_path='outputs/SavedModels/SVAE/LAPLACE_e64_ll-2.0_lr1e-04/',
        # model_path='outputs/SavedModels/SVAE/CAUCHY_e64_ll-2.0_lr1e-04/',
        device=torch.device('cpu'),
    )
    model_path = hydra.utils.to_absolute_path(cfg.eval.evaluate_ll.mdl_path)
    logger.info(f'Loaded: {model_path}')

    def evaluate_response(neuron_id, freq, theta, amplitude=1.):
        temp2 = []
        for left in [True, False]:
            temp = []
            for component in ['real', 'imag']:
                
                kernel = get_kernel(freq, theta, component=component, left=left)
                X = torch.as_tensor(amplitude * kernel, dtype=torch.float32).flatten()
                z = model._encode(X)
                temp.append(z[0].detach()[neuron_id])
            temp2.append(torch.linalg.norm(torch.tensor(temp), axis=0).item())
        return torch.mean(torch.tensor(temp2)).item()
    
    angles = np.linspace(0,180,20)
    frequencies = np.linspace(0.1,0.5,7)
    amplitudes = np.linspace(0.25,1.75,7)

    # Model neurons are the intersection of between the neurons with highest 
    # response norm for varying angle and freq
    high_norm_features = np.argsort(
        [np.linalg.norm(model.output_weights()[:,ind].reshape(12,12)) for ind in range(169)]
    )[-10:][::-1]

    vals = np.zeros((len(angles), len(frequencies), len(high_norm_features)))
    for i, angle in enumerate(angles):
        for j, freq in enumerate(frequencies):
            theta = angle/180 * np.pi

            # kernel = get_kernel(freq, theta)

            # X = torch.as_tensor(kernel, dtype=torch.float32).flatten()
            # z = model._encode(X)

            vals[i] = torch.tensor([
                evaluate_response(neuron_id=n_id, freq=freq, theta=theta,) 
                for n_id in high_norm_features
            ])


            # vals[i,j] = [z[0].detach()[ind] for ind in high_norm_features]
    
    reponse_per_feature = [np.linalg.norm(vals[:,:,i], ord=1) for i in range(len(high_norm_features))]
    sorted_neurons = high_norm_features[np.argsort(reponse_per_feature)[::-1]]
    model_neurons = sorted_neurons[:5]

    # highresp_neurons = np.asarray(highresp_neurons).flatten()
    # u, c = np.unique(highresp_neurons, return_counts=True)
    # print(np.sort(c))
    # model_neurons = u[np.argsort(c)][-10:][::-1]
    # model_neurons = [72, 38, 42, 82, 158] 

    
    fig, axs =plt.subplots(ncols=len(model_neurons))
    for i, ind in enumerate(model_neurons):
        feature = model.output_weights()[:,ind].reshape(12,12)
        axs[i].imshow(feature)
    savefig('features')
    plt.show()
    
    # fig, axs = plt.subplots(ncols=2, nrows=2, constrained_layout=True)

    fig = plt.figure()
    gs0 = GridSpec(2, 2, figure=fig, width_ratios=[3,1], height_ratios=[3,1])
    gs00 = gs0[0,0].subgridspec(2, 2)

    ax_size = fig.add_subplot(gs00[0, 0])
    ax_freq = fig.add_subplot(gs00[0, 1])
    ax_angle = fig.add_subplot(gs00[1, 0])
    ax_amp = fig.add_subplot(gs00[1, 1])

    for ax, label in zip([ax_size, ax_freq, ax_angle, ax_amp], ['A','B','C','D']):
        ax.text(-0.2, 1.1, label, fontsize='x-large', fontweight='bold',
                transform=ax.transAxes)
    
    # Varying angle
    logger.info(f'Plotting response to varying angle.')

    freq = 0.4

    vals = torch.zeros((len(angles), len(model_neurons)))
    for i, angle in enumerate(angles):
        theta = angle / 180 * np.pi

        vals[i] = torch.tensor([
            evaluate_response(neuron_id=n_id, freq=freq, theta=theta,) 
            for n_id in model_neurons
        ])

        # kernel = gabor_kernel(freq, theta).real
        # kernel_padded = pad_kernel(kernel)

        # X = torch.as_tensor(kernel_padded, dtype=torch.float32).flatten()
        # z = model._encode(X)

        # vals[i] = torch.tensor([z[0].detach()[n_id] for n_id in model_neurons])

    ax_angle.plot(angles, torch.abs(vals))
    ax_angle.set_xlabel('Orientation (deg.)')
    # savefig('z_vals_angles')
    # plt.show()
    del vals



    # Varying frequency
    logger.info(f'Plotting response to varying frequency.')
    
    vals = torch.zeros((len(angles), len(frequencies), len(model_neurons)))
    for i, angle in enumerate(angles):
        theta = angle / 180 * np.pi
        for j, freq in enumerate(frequencies):

            vals[i, j] = torch.tensor([
                evaluate_response(neuron_id=n_id, freq=freq, theta=theta,) 
                for n_id in model_neurons
            ])

            # temp2 = []
            # for left in [True, False]:
            #     temp = []
            #     for component in ['real', 'imag']:

            #         kernel = get_kernel(freq, theta, component=component, left=left)
            #         X = torch.as_tensor(kernel, dtype=torch.float32).flatten()
            #         z = model._encode(X)
            #         temp.append([z[0].detach()[n_id] for n_id in model_neurons])
            #     temp2.append(torch.linalg.norm(torch.tensor(temp), axis=0).detach().numpy())

            # vals[i,j] = torch.tensor(temp2).mean(axis=0)


    # break_points = [12,12,10,12,16]
    # fig, ax = plt.subplots()

    optim_freq_angles = []
    for i, neuron_id in enumerate(model_neurons):
        optimal_angle_index = torch.argmax(torch.linalg.norm(vals[:,:,i], axis=1))
        optim_freq_angles.append([
            neuron_id, angles[optimal_angle_index], frequencies[torch.argmax(torch.abs(vals)[optimal_angle_index,:,i])]
        ])
        # ax_freq.plot(frequencies[:break_points[i]], torch.abs(vals)[optimal_angle_index,:break_points[i],i])

        responses = vals[optimal_angle_index,:,i]
        # new_responses = np.copy(responses)
        # diffs = np.abs(np.diff(responses))
        # jumps = []
        # jump_id = None
        # update_id = 0
        # for j in range(len(diffs)):
        #     if j<5:
        #         jump = (diffs[j] > np.mean(diffs[:j]) + 3*np.std(diffs[:j]))
        #     else:
        #         jump = (diffs[j] > np.mean(diffs[j-5:j]) + 3*np.std(diffs[j-5:j]))
        #     jumps.append(jump)
        #     if jump_id is None and jump:
        #         jump_id = j
        #     elif (jump_id is not None) and jump:
        #         break

        #     if jump_id is not None:
        #         new_responses[j+1] = responses[j+1]-np.diff(responses)[jump_id]
        #     update_id += 1
        # jump_ids = np.arange(len(responses)-1)[jumps]
        # print(jump_ids)

        # # Correct responses
        # for i in range(len(jump_ids)-1):
        #     diffs =  np.diff(responses)
        #     new_responses[jump_ids[i]+1:jump_ids[i+1]] = responses[jump_ids[i]+1:jump_ids[i+1]] - diffs[jump_ids[i]] #* np.sign(np.diff(responses)[jump_ids[i]])

        ax_freq.plot(frequencies, responses)
        # ax_freq.plot(frequencies, new_responses, ls='--')

    ax_freq.set_xlabel('Frequency (1/px.)')
    # savefig('z_vals_freqs')
    # plt.show()
    del vals

    # Vary amplitude
    freq = 0.4
    logger.info(f'Plotting response to varying amplitude. Fixed freq: {freq}')
    vals = torch.zeros(len(angles), len(amplitudes))

    for i, angle in enumerate(angles):
        for j, amplitude in enumerate(amplitudes):
            theta = angle / 180 * np.pi

            vals[i,j] = evaluate_response(
                neuron_id=model_neurons[3],
                freq=freq, 
                theta=theta,
                amplitude=amplitude
            )
            # temp2 = []
            # for left in [True, False]:
            #     temp = []
            #     for component in ['real', 'imag']:
                    
            #         kernel = get_kernel(freq, theta, component=component, left=left)
            #         X = torch.as_tensor(amplitude * kernel, dtype=torch.float32).flatten()
            #         z = model._encode(X)
            #         temp.append(z[0].detach()[model_neurons[0]])
            #     temp2.append(torch.linalg.norm(torch.tensor(temp), axis=0).item())
            # vals[i,j] = torch.mean(torch.tensor(temp2))

            # kernel = amplitude * gabor_kernel(freq, theta).real
            # kernel_padded = pad_kernel(kernel)

            # X = torch.as_tensor(kernel_padded, dtype=torch.float32).flatten()
            # z = model._encode(X)

            # vals[i,j] = z[0].detach()[model_neurons[0]]


            

    cmap = plt.get_cmap('RdBu')
    norm = matplotlib.colors.Normalize(vmin=np.amin(amplitudes), vmax=np.amax(amplitudes))

    # fig, ax = plt.subplots()
    for i in range(len(amplitudes)):
        ax_amp.plot(angles, torch.abs(vals[:,i]), c=cmap(i/len(amplitudes)))
    # savefig('variable_constrast')
    # plt.show()

    ax_amp.set_xlabel('Orientation (deg.)')
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_amp)
    cb.set_label('Contrast')

    axins = ax_amp.inset_axes([0.7, 0.7, 0.25, 0.25])
    feature = model.output_weights()[:,model_neurons[3]].reshape(12,12)
    axins.imshow(feature, cmap='Greys_r')
    axins.set_axis_off()

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


    # plt.figure();

    for i in range(len(model_neurons)):
        freq_i = optim_freq_angles[i][2]
        theta_i = optim_freq_angles[i][1]/180 * np.pi

        vals = np.zeros(11)

        # X = torch.as_tensor(kernel, dtype=torch.float32).flatten()
        # z = model._encode(X)
        # vals[0] = z[0].detach()[model_neurons[i]]
        for j in range(1,12):

            temp2 = []
            for left in [True, False]:
                temp = []
                for component in ['real', 'imag']:

                    # if component=='real':
                    #     kernel = gabor_kernel(freq_i, theta=theta_i).real
                    # else:
                    #     kernel = gabor_kernel(freq_i, theta=theta_i).imag

                    # if kernel.shape[0] > 12:
                    #     kernel = trim_kernel(kernel)
                    # elif kernel.shape[0] < 12:
                    
                    kernel = get_kernel(freq_i, theta_i, component=component, left=left)
                    sub_kernel = trim_kernel(kernel, output_size=12-j, left=left)

                    im = Image.fromarray(sub_kernel)
                    sized_kernel = np.copy(np.asarray(im.resize((12,12)))) # copy to make writable

                    X = torch.as_tensor(sized_kernel, dtype=torch.float32).flatten()
                    z = model._encode(X)
                    temp.append(z[0].detach()[model_neurons[i]])
                temp2.append(torch.linalg.norm(torch.tensor(temp), axis=0).item())

            vals[j-1] = torch.mean(torch.tensor(temp2))

            # temp = []
            # for left in [True, False]:
            #     sub_kernel = trim_kernel(kernel, output_size=12-j, left=True)
            #     # sub_kernel_normalized = (sub_kernel - np.amin(sub_kernel))/np.amax((sub_kernel - np.amin(sub_kernel)))
            #     im = Image.fromarray(sub_kernel)

            #     sized_kernel = np.asarray(im.resize((12,12)))
            #     # if sub_kernel.shape[0] < 12:
            #     #     sized_kernel = pad_kernel(sub_kernel)
            #     # elif sub_kernel.shape[0] > 12:
            #     #     sized_kernel = trim_kernel(sub_kernel)
            #     # else:
            #     #     sized_kernel = sub_kernel

            #     X = torch.as_tensor(sized_kernel, dtype=torch.float32).flatten()
            #     z = model._encode(X)
            #     temp.append(z[0].detach()[model_neurons[i]])

            # vals[j-1] = np.mean(temp)

        ax_size.plot(vals[::-1])

    ax_size.set_xlabel('Size (px.)')

    # ----------------------------------------
    for ax in np.array(axs).flatten():
        ax.set_ylabel('Response (a.u.)')

    # savefig('size')
    # savefig('panel')
    # plt.show()
    
    # fig, axs = plt.subplots(ncols=2)
    # axs[0].imshow(kernel)
    # axs[1].imshow(kernel_padded)
    # savefig('kernel')
    # plt.show();


    # Reverse correlation experiment -------------------

    # Fast estimate
    z_feature_loc, z_feature_scale = model._encode(feature.flatten())
    reconstructed = (model.output_weights() @ z_feature_loc)/(z_feature_loc.sum())

    N = 10000
    X = torch.randn(N, 144)

    # # Load data
    
    # logger.info("Loading data")
    # device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logger.info("Using device: %s", device)
    
    # patches = torch.load(pathlib.Path(cfg.paths.user_home_dir) / pathlib.Path(cfg.paths.scratch) / "patches.pt")
    # X = patches['test'].to(device).reshape(-1, 144)    # size `(N_images, image_size)`
    # N = X.shape[0]
    # # noise = 0.1*torch.randn(N, 144)
    # # X = X + noise
    # # print(N)

    z_loc, z_logscale = model._encode(X)
    R = model._reparameterize(z_loc, z_logscale, family='GAUSSIAN')

    scale_estimate = lambda estimate: torch.divide(estimate, torch.sum(torch.abs(R), axis=0))
    
    # Estimate
    STA = X.T @ R # Standard STA
    STA = scale_estimate(STA)

    STA_w = N * torch.tensor(sp.linalg.pinv(X.T @ X)) @ X.T @ R          # Whitened STA
    STA_w = scale_estimate(STA_w)

    lambd = 0.001
    STA_ridge = N * (torch.tensor(sp.linalg.pinv(X.T @ X + lambd * torch.eye(144))) @ X.T @ R)   # Ridge
    STA_ridge = scale_estimate(STA_ridge)

    # for estimate in [STA, STA_w, STA_ridge]:
    #     print('-')
    #     print(STA.mean(), STA.max(), STA.min())
    #     print(scale_estimate(estimate).mean(), scale_estimate(estimate).max(), scale_estimate(estimate).min())




    # fig, axs = plt.subplots(ncols=3)
    # axs[0].imshow(feature)
    # axs[1].plot(z_feature_loc.detach().numpy())
    # axs[2].imshow(reconstructed.detach().reshape(12,12))
    # savefig('z hist')
    # plt.show()

    # fig, axs = plt.subplots(ncols=10, nrows=4, constrained_layout=True)
    gs01 = gs0[1,0].subgridspec(2, 10)
    print(dir(gs01))

    ax_feature0 = fig.add_subplot(gs01[0,0])
    ax_feature0.imshow(model.output_weights()[:, sorted_neurons[0]].reshape(12,12).detach())

    ax_estimate0 = fig.add_subplot(gs01[1,0])
    ax_estimate0.imshow(STA[:, sorted_neurons[0]].reshape(12,12).detach())

    for i in range(1,10):
        ax_feature = fig.add_subplot(gs01[0,i])
        ax_feature.imshow(model.output_weights()[:, sorted_neurons[i]].reshape(12,12).detach())

        ax_estimate = fig.add_subplot(gs01[1,i])
        estimate = scale_estimate(STA)
        ax_estimate.imshow(estimate[:, sorted_neurons[i]].reshape(12,12).detach())
        # for j, estimate in enumerate([STA, STA_w, STA_ridge]):
        #     ax_estimate = fig.add_subplot(gs01[j+1,i])
    # for ax in axs.flatten():
    #     ax.set_axis_off()

    ax_feature0.text(-0.6,0.5, '$\Phi_i$', fontsize='x-large', transform=ax_feature0.transAxes)
    ax_estimate0.text(-0.6,0.5, 'RF', fontsize='x-large', transform=ax_estimate0.transAxes)
    savefig('panel2')
    plt.show()




if __name__=='__main__':
    main()
