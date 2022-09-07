from optax import chain
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

columns = ['mean', 'std', 'chain_length', 'sampler', 'schedule']

true = [-166.15296, 0.04382]
baseline = [-175.01265, 0.94543]
nsamples = 16

data_array = np.array([
    [-169.00674, 0.36197, 2, 'HMC', 'sigmoid'],
    [-168.86441, 0.32967, 10, 'HMC', 'sigmoid'],
    [-166.77231, 0.17331, 100, 'HMC', 'sigmoid'],
    [-166.17209, 0.06383, 1000, 'HMC', 'sigmoid'],
    [-169.05957, 0.38042, 2, 'Metropolis', 'sigmoid'],
    [-168.98682, 0.45650, 10, 'Metropolis', 'sigmoid'],
    [-169.10210, 0.35713, 100, 'Metropolis', 'sigmoid'],
    [-168.94930, 0.32450, 1000, 'Metropolis', 'sigmoid'],
    [-169.00674, 0.36197, 2, 'HMC', 'linear'],
    [-168.81671, 0.32077, 10, 'HMC', 'linear'],
    [-166.59669, 0.16697, 100, 'HMC', 'linear'],
    [-166.16573, 0.05714, 1000, 'HMC', 'linear'],
    [-169.05957, 0.38042, 2, 'Metropolis', 'linear'],
    [-168.98682, 0.45650, 10, 'Metropolis', 'linear'],
    [-169.10210, 0.35713, 100, 'Metropolis', 'linear'],
    [-168.94930, 0.32450, 1000, 'Metropolis', 'linear']
])

chain_lengths = [2,10,100,1000]

df = pd.DataFrame(data_array, columns=columns)

# # Plot
# fig, ax = plt.subplots(figsize=[8,4]);
# ax.plot(chain_lengths, true[0]*np.ones_like(chain_lengths), c='k', label='True')
# # ax.fill_between(chain_lengths, (true[0]-true[1])*np.ones_like(chain_lengths), (true[0]+true[1])*np.ones_like(chain_lengths), color='tab:grey', alpha=0.3)

# ax.plot(chain_lengths, baseline[0]*np.ones_like(chain_lengths), c='tab:red', label='Naive likelihood weighting')
# # ax.fill_between(chain_lengths, (baseline[0]-baseline[1])*np.ones_like(chain_lengths), (baseline[0]+baseline[1])*np.ones_like(chain_lengths), color='tab:red', alpha=0.3)

# for sampler, color in zip(['HMC', 'Metropolis'], ['tab:blue', 'tab:orange']):
#     for schedule, ls in zip(['linear', 'sigmoid'], ['-', '--']):
#         try:
#             subdf = df.query(f"sampler == '{sampler}' and schedule == '{schedule}'")
#             if subdf.empty: continue
#             means, stds = subdf['mean'].values.astype(float), subdf['std'].values.astype(float)
#             ax.errorbar(x=chain_lengths[:len(means)], y=means, yerr=stds, 
#                         label=f'AIS with {sampler}, {schedule} schedule', color=color, ls=ls)
#         except pd.core.computation.ops.UndefinedVariableError:
#             continue


# # ax.plot([float(i) for i in data_array[:,2]], [float(i) for i in data_array[:,0]], label='AIS with HMC')
# # ax.fill_between([float(i) for i in data_array[:,2]], 
# #     np.array([float(i) for i in data_array[:,0]]) - np.array([float(i) for i in data_array[:,1]]),
# #     np.array([float(i) for i in data_array[:,0]]) + np.array([float(i) for i in data_array[:,1]]), alpha=0.3)

# ax.set_xscale('log');
# ax.set_xlabel('Chain length (# intermediate dists)')
# ax.set_ylabel("Log-likelihood");
# ax.set_title("AIS validation on Gaussian Linear Model")

# # Shrink current axis by 20%
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.6, box.height])

# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.8))

# # ax.legend()

# # plt.savefig('./bin/figures/AIS_validation.png', dpi=300)
# plt.show();


# # =========================================================================

# # Vary latent ndims, fix data at 144, with hmc and sigmoid and 100 cs
# ndims = [10,50,100,144,169, 200]
# ais_ll = [-160.89139,-165.37590,-166.47638, -166.06908, -166.77231, -166.83162]
# ais_ll_std = [0.14733,0.18357,0.22049,0.16990 , 0.17332, 0.24668]

# true_ll = [-160.63547, -164.80186, -165.89351, -165.29851, -166.15287, -166.13543]
# true_ll_std = [0.05435, 0.05151, 0.04378, 0.04150, 0.04205, 0.04152]

# fig, ax = plt.subplots();
# ax.errorbar(x=ndims, y=true_ll, yerr=true_ll_std);
# ax.errorbar(x=ndims, y=ais_ll, yerr=ais_ll_std);
# ax.axvline(x=144, c='tab:grey', zorder=-1);
# plt.show();

# =========================================================================
likelihood_scale = np.exp([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
ais_ll = np.array([-1284.76636, -739.79462, -409.76605, -229.56145, -150.70265, -138.11469, -166.77231, -218.57512, -281.88647, -350.42294, -421.10541, -492.61444])
ais_ll_std = np.array([9.90325, 5.71287, 3.31766, 1.74808, 0.87742, 0.42718, 0.17332, 0.06000, 0.02253, 0.00867, 0.00327, 0.00121])

true_ll = np.array([-445.70607, -327.14045, -223.50964, -155.52575, -126.78433, -132.80816, -166.15287, -218.52868, -281.88057, -350.42175, -421.10514, -492.61438])
true_ll_std = np.array([1.95217, 1.31325, 0.75363, 0.40096, 0.20041, 0.09482, 0.04205, 0.01733, 0.00674,0.00254 , 0.00094, 0.00035])

fig, ax = plt.subplots();
ax.errorbar(x=likelihood_scale, y=true_ll, yerr=true_ll_std, c='k', label='True');
ax.errorbar(x=likelihood_scale, y=ais_ll, yerr=ais_ll_std, label='AIS, chain-length 100');
ax.errorbar(np.exp([-3.0, -2.0, -1.0]),
            y=[-535.30542, -232.12170, -127.83055],
            yerr=[2.50564, 0.81341, 0.24135],
            marker='o', label='AIS, chain-length 1000')
ax.legend();
ax.set_xlabel('Observation noise scale $\sigma$');
ax.set_ylabel('Log-likelihood');
ax.set_xscale('log');

ax2 = ax.twiny()
ax2.plot(likelihood_scale, np.ones_like(likelihood_scale), alpha=0.0)
# ax2.cla()
ax2.set_xticks([np.exp(-2.5),np.exp(2.5)])
ax2.set_xticklabels(['high SNR', 'low SNR'])

ax.set_title('SNR impact on AIS estimate')
plt.savefig('./bin/figures/AIS_SNR.png', dpi=300)
plt.show();

fig, ax = plt.subplots();
ax.errorbar(x=likelihood_scale, y=true_ll-ais_ll, yerr=true_ll_std+ais_ll_std);
ax.set_xlabel('Observation noise scale $\sigma$');
ax.set_ylabel('Log-likelihood difference');
ax.set_xscale('log');
# ax.set_yscale('log');

ax2 = ax.twiny()
ax2.plot(likelihood_scale, np.ones_like(likelihood_scale), alpha=0.0)
# ax2.cla()
ax2.set_xticks([np.exp(-2.5),np.exp(2.5)])
ax2.set_xticklabels(['high SNR', 'low SNR'])

ax.set_title('SNR impact on AIS estimate')
plt.savefig('./bin/figures/AIS_SNR_diff.png', dpi=300)
plt.show();