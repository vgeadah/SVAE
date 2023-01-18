import numpy as np
import sys
import itertools
import scipy
from tqdm import tqdm
import pandas as pd
from scipy.stats import multinomial
from sklearn.mixture import GaussianMixture
from scipy.optimize import curve_fit
from typing import Callable, Tuple

# ------------------------------------------------------------------
def GMM2d(array, n_samples=1000, n_components=2, gmm_kwargs={'covariance_type':'tied'}):
    r'''2 dim array proportional to the density you want to estimate with a GMM'''
    N = array.shape[0]

    array = np.where(np.isnan(array), 0.0, array)

    # Get samples
    array_normalized_flat = array.flatten()/np.linalg.norm(array.flatten(), 1)
    # array_normalized_flat = scipy.special.softmax(array.flatten())
    assert np.abs(np.sum(array_normalized_flat) - 1.) < 1e-05
    
    categorical = multinomial(1, array_normalized_flat)
    indices = np.array(list(itertools.product(range(N),range(N))))
    assert indices.shape == (N*N, 2)

    samples = categorical.rvs(n_samples)

    X = []
    for sample in samples:
        for index, j in enumerate(sample):
            if j:
                X.append(indices[index])
                break
    X = np.array(X) 
    assert X.shape == (n_samples,2)

    # GMM fit
    gm = GaussianMixture(n_components=n_components, **gmm_kwargs).fit(X)
    # print('Score:', np.mean(np.exp(gm.score_samples(X))))
    return gm

def get_frequencies(
        image: np.ndarray,
        PLOT: bool=False
        ) -> np.ndarray:
    r'''Compute oriented 2D frequency and 
    '''
    N = image.shape[0]

    n_components = 2
    if np.linalg.norm(image) < 0.05:
        n_components = 4
    
    if np.linalg.norm(image) < 0.1:
        _amplify = lambda x: np.square(x)
    else:
        _amplify = lambda x: scipy.special.softmax(x)
    fourier = np.fft.fft2(image)
    fourier_norm = np.sqrt(np.square(fourier.real) + np.square(fourier.imag))

    try:
        signal = _amplify(fourier_norm)
        gm = GMM2d(array=signal, n_components=n_components, n_samples=10000)
        
        freq_domain = np.fft.fftfreq(N)
        frequencies = []
        for mean in gm.means_:
            frequencies.append([freq_domain[round(mean[1])], freq_domain[round(mean[0])]])

        assert np.linalg.norm(frequencies[0] + frequencies[1]) < 1e-08, \
                'Frequencies not opposite. Using frequency shift.'
    except AssertionError:
        # print('Shifting.')
        # counter = 0
        # n_samples = 1000
        # while np.linalg.norm(frequencies[0] + frequencies[1]) > 1e-08 and counter < 4:
        signal = _amplify(np.fft.fftshift(fourier_norm))
        gm = GMM2d(array=signal, n_components=n_components, n_samples=10000)
        
        freq_domain = np.fft.fftshift(np.fft.fftfreq(N))
        frequencies = []
        for mean in gm.means_:
            frequencies.append([freq_domain[round(mean[1])], freq_domain[round(mean[0])]])

        # n_samples *= 10
        # counter += 1
    
    # print(gm.means_)
    # print(gm.converged_, gm.lower_bound_)
    # print(frequencies)
    # assert np.linalg.norm(frequencies[0] + frequencies[1]) < 1e-08, 'Frequencies still not opposite.'

    # # Start full density fit
    # masked = np.zeros(shape=[N,N])
    # for i, j in itertools.product(range(12),range(12)):
    #     if (i==0 or j==0 or i+j > N):
    #         masked[i,j] = np.nan
    #     else:
    #         masked[i,j] = fourier_norm[i,j]
    # fourier_norm = masked# np.array([[fourier_norm[i,j] if i+j<N-1 else np.nan for i in range(N)] for j in range(N)])
    # gm_full = GMM2d(
    #     array=np.square(fourier_norm),
    #     gmm_kwargs={'covariance_type':'tied', 'means_init': frequencies_init}
    # )

    if PLOT:
        _, axs = plt.subplots(ncols=2)
        axs[0].imshow(image)
        axs[1].imshow(signal)
        axs[1].scatter(gm.means_[:,1], gm.means_[:,0], c='r')
        plt.show()

    return np.array(frequencies)

def orient_frequencies(frequencies: np.ndarray):
    def single_orientedfrequency(frequency: np.ndarray):
        if frequency[0]==0.:
            oriented_frequency = np.abs(frequency[1])
            orientation = np.pi/2
        elif frequency[1]==0.:
            oriented_frequency = np.abs(frequency[0])
            orientation = 0.
        else:
            oriented_frequency = np.power(np.power(frequency[0], -2) + np.power(frequency[1], -2), -1/2)
            base_vector = np.array([1,0])
            orientation = np.arccos(np.dot(frequency, base_vector)/
                            (np.linalg.norm(frequency) * np.linalg.norm(base_vector)))
        return oriented_frequency, orientation
    
    oriented_frequencies, orientations = [], []
    for frequency in frequencies:
        oriented, angle = single_orientedfrequency(frequency)

        oriented_frequencies.append(oriented)
        orientations.append(angle)

    return np.array(oriented_frequencies), np.array(orientations)

def determine_angle_from_conv(
        frequency: float, 
        image: np.ndarray, 
        PLOT: bool=False,
        ) -> float:
    r'''
    Returns the angle that minimizes the norm of the convolution between 
        image * gabor_kernel(frequency = `frequency`)
    '''
    from skimage.filters import gabor_kernel
    from scipy import ndimage as ndi

    kernel = gabor_kernel(frequency).real

    df = pd.DataFrame(columns=['angle', 'min', 'max', 'norm'])
    for angle in np.linspace(0, 180, 50):
        rotated_image = ndi.rotate(image, angle)
        convolved = ndi.convolve(rotated_image, kernel, mode='constant')
        convolved_norms = np.array([[np.linalg.norm(x) for x in i] for i in convolved])

        df = pd.concat([
            df,
            pd.DataFrame({
                'angle':[angle], 
                'min':[np.amin(convolved_norms)], 
                'max':[np.amin(convolved_norms)],
                'norm':[np.linalg.norm(convolved)]
                })
            ], ignore_index=True)

        norm_argmin = np.argmin(df['norm'].values)
        conv_angle = df.iloc[norm_argmin]['angle']

    if PLOT:
        _, ax = plt.subplots();
        sns.lineplot(x='angle', y='value', data=df, hue='metric');
        ax.set_yscale('log')
        plt.show();

    return conv_angle


def project_image(
        image: np.ndarray, 
        orientation_vector: np.ndarray
        ) -> np.ndarray:
    r'''
    Project each point (x,y,z) onto the plane define by the orientation_vector and z axis

    Parameters:
        image: np.ndarray, shape `(N, N)`
        orientation_vector: np.ndarray, shape `(2, )`
            vector in the (x,y) plane, onto which to project
    Returns:
        projected: np.ndarray, shape `(N*N, 2)`
            of image onto plane
    '''
    N = image.shape[0]

    projected = np.zeros(shape=(N*N, 2))
    counter = 0
    for i in range(N):
        for j in range(N):
            v = np.array([i,j])
            plane_proj = (np.dot(v, orientation_vector)/ np.linalg.norm(orientation_vector)**2) * orientation_vector 
            assert plane_proj.shape == (2,)
            sign = np.sign(np.dot(v, orientation_vector))

            projected[counter] = np.array([sign * np.linalg.norm(plane_proj), image[i,j]])
            counter += 1
    return projected

def positive_enveloppe(array):
    hull_vertices = ConvexHull(array).vertices
    hull = array[hull_vertices]
    positive_hull = hull[hull[:,1]>0.0]
    return positive_hull[positive_hull[:,0].argsort()]

def max_enveloppe(array):
    df = pd.DataFrame(array, columns=['x','y']).round(decimals=3)
    max_df = df.groupby('x').max()
    return max_df.reset_index().to_numpy()

def ub_enveloppe(array):
    sorted_array = array[array[:,0].argsort()]
    L = sorted_array.shape[0]

    state = sorted_array[0]
    left_ub = [state]
    for point in sorted_array:
        if point[1] > state[1]:
            left_ub.append(point)
            state = point
        # else:
        #     left_ub.append([point[0], state[1]])
    left_ub = np.array(left_ub)

    state = sorted_array[-1]
    right_ub = [state]
    for point in sorted_array[::-1]:
        if point[1] > state[1]:
            right_ub.append(point)
            state = point
        # else:
        #     right_ub.append([point[0], state[1]])
    # right_ub = np.flip(right_ub)

    ub_enveloppe = np.concatenate((left_ub, right_ub[:-1]), axis=0)

    # print(left_ub, right_ub)

    # assert np.linalg.norm(right_ub[:,0]-left_ub[:,0]) < 1e-05
    # ub_enveloppe = np.zeros(shape=(L, 2))
    # ub_enveloppe[:,1] = np.array([min(a,b) for a, b in zip(left_ub[:,1], right_ub[:,1])])
    # ub_enveloppe[:,0] = left_ub[:,0]
    return ub_enveloppe[ub_enveloppe[:,0].argsort()]

def gaussian(x, mu, sigma, a):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def estimate_spatial(
        image: np.ndarray,
        method: Callable,
        oriented_frequency: float = 1., 
        orientation: float = 0.,
        ) -> Tuple[np.ndarray, np.ndarray]:
    
    if np.abs(oriented_frequency) < 1e-10:
        return np.nan*np.ones(shape=(3,2)), np.nan*np.ones(shape=(3,2))
    else:
        # First project onto subspace span by the oriented frequency (and normal to it)
        projection_1 = project_image(
            image=image, 
            orientation_vector=(1/oriented_frequency)*np.array([np.cos(orientation), np.sin(orientation)])
            )
        projection_2 = project_image(
            image=image, 
            orientation_vector=(1/oriented_frequency)*np.array([-np.sin(orientation), np.cos(orientation)])
            )

        # Second, estimate parameters of the gaussian factor
        parameters = []
        covariances = []
        for proj in [projection_1, projection_2]:
            try:
                x_data, y_data = method(proj)[:,0], method(proj)[:,1]
            except ValueError:
                p, c = [np.nan for _ in range(3)], [np.nan for _ in range(3)]
                parameters.append(p)
                covariances.append(c)
                continue

            try:
                p, c = curve_fit(gaussian, x_data, y_data, 
                                    p0=[np.mean(x_data), 3, max(np.amax(y_data), 0.)],
                                    bounds=([-np.inf, 0., 0.], [np.inf, np.inf, np.inf]))
            except RuntimeError:
                p, c = [np.nan for _ in range(3)], [np.nan for _ in range(3)]
                parameters.append(p)
                covariances.append(c)
                continue
            except ValueError:
                print(x_data)
                sys.exit()
            parameters.append(p)
            covariances.append(np.diag(c))
        return np.array(parameters), np.array(covariances)

if __name__=='__main__':
    import torch
    import argparse

    parser = argparse.ArgumentParser(description='Estimate statistics of learned receptive filters.')
    parser.add_argument('--plot', default=False, action='store_true',
                        help='Plot panel of statistics and receptive filters.')
    parser.add_argument('--model-dir', type=str, 
                        default='/outputs/SavedModels/sparsenet/LAPLACE_lambda6.0e-01_N 5000_nf169.pth',
                        help='Model directory and file name from USER DIR')
    args = parser.parse_args()

    ## LOAD
    HOMEDIR = '/home/vg0233/PillowLab/SVAE'
    try:
        pretrained_dict = torch.load(HOMEDIR+args.model_dir)
    except RuntimeError:
        pretrained_dict = torch.load(HOMEDIR+args.model_dir, map_location=torch.device('cpu'))
    features = pretrained_dict['Phi.weight'].detach().cpu().numpy()

    # # Test
    # feature_id = 2
    # feature = features[:,feature_id]
    # image = feature.reshape(12,12)

    # fourier = np.fft.fftshift(np.fft.fft2(image))
    # fourier_norm = np.sqrt(np.square(fourier.real) + np.square(fourier.imag))
    # # print(np.linalg.norm(fourier_norm))
    # signal = fourier_norm #-np.amin(fourier_norm))/(np.amax(fourier_norm)-np.amin(fourier_norm))
    # gm = GMM2d(array=signal, n_components=2, n_samples=10000)
    # freq_domain = np.fft.fftshift(np.fft.fftfreq(12))
    # frequencies = []
    # for mean in gm.means_:
    #     frequencies.append([freq_domain[round(mean[1])], freq_domain[round(mean[0])]])

    # print(gm.means_, gm.covariances_, gm.converged_, np.exp(gm.lower_bound_/144))
    # _, axs = plt.subplots(ncols=2)
    # axs[0].imshow(image)
    # axs[1].imshow(signal)
    # axs[1].scatter(gm.means_[:,1], gm.means_[:,0], c='r')
    # plt.show()
    # sys.exit()
    # freqs = get_frequencies(image, PLOT=False)
    # oriented_frequency, orientation = orient_frequencies(freqs)

    # from scipy.spatial import ConvexHull
    # projection_1 = project_image(image, (1/oriented_frequency[0])*np.array([np.cos(orientation[0]), np.sin(orientation[1])]))
    # projection_2 = project_image(image, (1/oriented_frequency[0])*np.array([-np.sin(orientation[1]), np.cos(orientation[0])]))
    # print(orientation[0])


    # plt.figure();
    # plt.plot(*projection_1.transpose(), '.')
    # plt.plot(*projection_2.transpose(), '.')
    # # plt.plot(*(projection_1[ConvexHull(projection_1).vertices]).transpose(), '.')
    # # plt.plot(*positive_enveloppe(projection_1).transpose(), c='tab:blue')
    # # plt.plot(*positive_enveloppe(projection_2).transpose(), c='tab:orange')
    # # for proj in [projection_1, projection_2]:
    # #     plt.plot(*ub_enveloppe(proj).transpose())
    # #     plt.plot(x_data, gaussian(x_data, *p), color='k')
    
    # plt.show();
    # sys.exit()

    # Compute all features
    # gabor_df = pd.DataFrame(columns=['feature_id','frequency','angle'])
    gabor_df = pd.DataFrame(columns=['feature_id','frequency','angle','2-matrix-norm', 'mu_x', 'mu_y', 'sigma_x', \
                                        'sigma_y', 'mu_x_var', 'mu_y_var', 'sigma_x_var', 'sigma_y_var',])

    for feature_id, feature in enumerate(tqdm(features.transpose())):
        # if feature_id%5 != 0:
        #     continue 
        image = feature.reshape(12,12)

        # Get Gabor frequencies
        freqs = get_frequencies(image, PLOT=False)
        oriented_frequencies, orientations = orient_frequencies(freqs)
        
        # oriented_frequencies = np.unique(oriented_frequencies)
        # orientations = np.unique(orientations)

        # Get Gabor angle
        # angle_gabor = determine_angle_from_conv(oriented_frequency, image)

        # print(oriented_frequencies, orientations)

        ps, cs = estimate_spatial(
            image, method=ub_enveloppe, 
            orientation=orientations[0], oriented_frequency=oriented_frequencies[0]
            )

        gabor_df = pd.concat([
            gabor_df,
            pd.DataFrame({
                'feature_id':[feature_id],# for _ in range(len(oriented_frequencies))],
                'frequency': [oriented_frequencies[0]], 
                'angle': [orientations[0]],
                '2-matrix-norm':[np.linalg.norm(image)],# for _ in range(len(oriented_frequencies))],
                'mu_x':[ps[0,0]],
                'mu_y':[ps[1,0]],
                'sigma_x':[ps[0,1]],
                'sigma_y':[ps[1,1]],
                'mu_x_var':[cs[0,0]],
                'mu_y_var':[cs[1,0]],
                'sigma_x_var':[cs[0,1]],
                'sigma_y_var':[cs[1,1]],
                })
            ], ignore_index=True)
    
    # if args.verbose:
    print(gabor_df)
    #     for key in ['frequency','angle', ]



    # SAVE
    import pathlib
    filepath = pathlib.Path(HOMEDIR, *args.model_dir.split('/')[:-1], 'feature_statistics.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    gabor_df.to_csv(filepath)  

    # ------------------------------------------------------------------
    # Plot

    if args.plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=[10, 3], constrained_layout=True)
        gs = fig.add_gridspec(1, 4, wspace=0.1)

        image_gs = gs[0].subgridspec(3, 3, hspace=0.0, wspace=0.0)
        ax_freq = fig.add_subplot(gs[1])
        ax_angle = fig.add_subplot(gs[2])
        ax_norm = fig.add_subplot(gs[3])

        # norms = [np.log(np.linalg.norm(features[:,i].reshape(12,12))) for i in range(169)]
        # print(norms)
        # ax_norm.scatter(gabor_df['2-matrix-norm'].values, gabor_df['frequency'].values, alpha=0.5)
        # ax_norm.set_xscale('log')
        # ax_norm.set_ylabel("Frequency")
        # ax_norm.set_xlabel('Feature norm')
        # ax_norm.scatter(gabor_df.query('sigma_x_var < 1.0')['sigma_x'].values, 
        #                 gabor_df.query('sigma_x_var < 1.0')['sigma_y'].values)
        # ax_norm.plot(np.linspace(0,5,10), np.linspace(0,5,10), zorder=-1, alpha=0.5)
        sns.scatterplot(data=gabor_df.query('sigma_x_var < 1.0'), x='sigma_x', y='sigma_y', hue='2-matrix-norm', ax=ax_norm)

        ax_freq.set_xlim([0,0.5])

        for i, (s, t) in enumerate(itertools.product(range(3), range(3))):
            ax = fig.add_subplot(image_gs[s, t])
            ax.imshow(features[:,i].reshape(12,12))
            ax.set_xticks([])
            ax.set_yticks([])

        sns.scatterplot(data=gabor_df, x='frequency', y='2-matrix-norm',
                        ax=ax_freq) # , palette='Blues'
        sns.scatterplot(data=gabor_df, x='angle', y='2-matrix-norm',
                        ax=ax_angle) #, palette='Blues'
        # sns.histplot(data=gabor_df.groupby(by='feature_id').mean(), x='angle', 
        #                 bins=10, ax=ax_angle);
        # ax_freq.hist(gabor_df['frequency'].values, bins=np.linspace(0, 0.5, 25))
        # ax_freq.set_xlabel('Frequency')
        # ax_freq.set_ylabel('Count')

        # ax_angle.hist(gabor_df['angle'].values, bins=8)
        # ax_angle.set_ylabel('Angle (rad.)')

        # freq_xticks = ax_freq.get_xticks()
        # ax_freq.set_xticklabels([f'{i+1}/12' for i in range(len(freq_xticks))])

        # bins = np.linspace(0,np.pi,10)
        # widths = np.diff(bins)

        # y, bins = np.histogram(np.sort(gabor_df.groupby(by='feature_id').mean()['angle'].values), bins=bins)
        # ax_angle.bar(bins[:-1], y, align='edge', width=widths, fill=True, edgecolor='k')
        # # ax_angle.plot(bins[:-1], y)
        # # ax_angle.set_xlim([0,180])

        plt.show();