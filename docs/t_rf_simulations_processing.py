import numpy as np
from numpy import ndarray
from debiased_spatial_whittle.plotting_funcs import plot_marginals
import matplotlib.pyplot as plt
import pickle



def rmse(y: ndarray, y_tilde: ndarray) -> float:
    return np.sqrt(np.mean((y-y_tilde)**2))

with open('MLEs_t_Exp_kernel.pkl', 'rb') as f:
    mles = pickle.load(f)
    
MLEs= list(mles.values())
params = np.log([10.,1.])

mles32, mles64, mles128, mles256 = mles.values()

titles = list(mles.keys())
nus = list(range(5,15+1)) + list(range(20,55,5)) + [9999]
axis_labels = [r'log$\rho$', r'log$\sigma$']
# for i, mles_ in enumerate(MLEs):
#     plot_marginals(mles_, params, title=titles[i], density_labels=nus, axis_labels=axis_labels)

# stop


grids = [32, 64, 128,256]
ngrids = len(grids)
rhos, sigmas = np.array(MLEs).T
mles_array = np.stack([rhos.T,sigmas.T])

import pandas as pd
param_list = ['rho', 'sigma']
index = pd.MultiIndex.from_product([param_list, grids, nus], 
                                   names=["parameters", "grids", "nu"])


df = pd.DataFrame(mles_array.reshape(len(index),2000).T, columns=index)


# index = pd.MultiIndex.from_product([grids, nus], names=["grids", "nu"])
# rhos_df = pd.DataFrame(rhos.reshape(len(index),2000).T, columns=index)
# sigmas_df = pd.DataFrame(sigmas.reshape(len(index),2000).T, columns=index)
# df  = pd.concat([rhos_df,sigmas_df], axis=1, keys=['rho', 'sigma'])

rho_bias = abs(df['rho'].mean()-params[0]).unstack()
rho_rmse = df['rho'].apply(rmse, y_tilde=params[0]).unstack()

std = df.std()

sigmas_bias = abs(df['sigma'].mean()-params[1]).unstack()
sigmas_rmse = df['sigma'].apply(rmse, y_tilde=params[1]).unstack()

fig, ax = plt.subplots(2,3, figsize=(17,10))
fig.suptitle(f'deWhittle MLEs $t$-random field, Exp. kernel, {axis_labels[0]}={round(params[0],2)}, ' \
             f'{axis_labels[1]}={round(params[1],2)}, ' \
             r'$\sigma^2_{\epsilon}$=0.1'    , fontsize=26, color='gray', y=1.02)
ax[0,0].plot(rho_bias.to_numpy().T, 'o-', )
ax[0,1].plot(std['rho'].unstack().to_numpy().T, 'o-')
ax[0,2].plot(rho_rmse.to_numpy().T, 'o-', label=titles)


ax[1,0].plot(sigmas_bias.to_numpy().T, 'o-', )
ax[1,1].plot(std['sigma'].unstack().to_numpy().T, 'o-')
ax[1,2].plot(sigmas_rmse.to_numpy().T, 'o-', label=titles)

ax_titles = ['bias', 'stdev', 'rmse']
for i, plot in enumerate(ax.flatten()):
    plot.set_xticks(np.arange(19), nus, fontsize=14)
    plot.get_xticklabels()[-1].set_rotation(45)
    # for label in plot.get_xticklabels()
    plot.spines[['top', 'right']].set_visible(False)
    if i<3:
        plot.set_title(ax_titles[i], fontsize=22)
    else:
        plot.set_xlabel(r'$\nu$', fontsize=22)
    
    if i==0 or i==3:
        plot.set_ylabel(axis_labels[i//2],fontsize=22)
    
ax[0,2].legend(fontsize=16, bbox_to_anchor=(0.7,0.7))
fig.tight_layout()
plt.show()
