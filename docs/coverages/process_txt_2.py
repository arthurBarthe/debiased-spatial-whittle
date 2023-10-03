import re
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import SquaredExponentialModel

mpl.rcParams.update({'font.size': 16})
mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False


def get_file_name(n: int, C: str = 'C5') -> str:
    '''Get name of file (square grids).'''
    print(n)
    return f'{C}/{C}_DeWhittle_{n}x{n}_SquaredExponentialModel.txt'

C = 'C4'
ns = (64, 128, 256, 512)
# ns = [(n,)*2 for n in (64, 128, 256, 512)]
d = {n: np.loadtxt(get_file_name(n, C=C), skiprows=2, max_rows=500) for n in ns}


rho, sigma, nugget = 7., 3., 0.1  # pick smaller rho

prior_mean = np.array([rho, sigma])
prior_cov = np.array([[1., 0.], [0., .1]])

# grid = RectangularGrid(n)
model = SquaredExponentialModel()   # TODO: try sq. exponential model!!!

prior_label = rf'$\rho \sim N({prior_mean[0]}, {np.diag(prior_cov)[0]})$, $\sigma \sim N({prior_mean[1]}, {np.diag(prior_cov)[1]})$ '

from scipy import stats
unif = stats.uniform(0,1)
qs = np.linspace(0,1, 1000)
theory_quants = unif.ppf(qs)

cm1 = plt.get_cmap('Blues')
cm2 = plt.get_cmap('Greens')
fig,ax = plt.subplots(2,1, figsize=(15,15))
fig.suptitle(f'QQ plots, {C} adjustment, {model.name}, {prior_label}')
ax[0].plot(theory_quants, theory_quants, c='r', linewidth=3, label='std. uniform', zorder=10)
for i, n in enumerate(ns):
    p     = d[n][:, -4:-2]
    adj_p = d[n][:, -2:]
    ax[0].plot(theory_quants, np.quantile(p[:,0], qs), 
               '.', c=cm1(i/len(ns)+0.2), markersize=10, label=f'{n=}', zorder=i)
    ax[0].plot(theory_quants, np.quantile(adj_p[:,0], qs), 
               '.', c=cm2(i/len(ns)+0.2), markersize=10, label=f'{n=}', zorder=i)
    
ax[0].legend(fontsize=18)

ax[1].plot(theory_quants, theory_quants, c='r', linewidth=3, label='std uniform', zorder=10)
for i, n in enumerate(ns):
    p     = d[n][:, -4:-2]
    adj_p = d[n][:, -2:]
    ax[1].plot(theory_quants, np.quantile(p[:,1], qs), 
               '.', c=cm1(i/len(ns)+0.1), markersize=10., label=f'{n=}')
    ax[1].plot(theory_quants, np.quantile(adj_p[:,1], qs), 
               '.', c=cm2(i/len(ns)+0.2), markersize=10., label=f'{n=}')
# ax[1].legend()

ax[0].set_xlabel( r'$\rho$', fontsize=26)
ax[1].set_xlabel( r'$\sigma$', fontsize=26)
for i in range(2):
    ax[i].set_xlim([0,1])
    ax[i].set_ylim([-0.01,1.01])

fig.tight_layout()
plt.show()
