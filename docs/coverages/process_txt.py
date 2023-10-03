import re
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import SquaredExponentialModel

mpl.rcParams.update({'font.size': 16})
mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False


# n = (64,64)
# n = (128,128)
n = (256,256)
# n = (512,512)
rho, sigma, nugget = 7., 3., 0.1  # pick smaller rho

prior_mean = np.array([rho, sigma])
prior_cov = np.array([[1., 0.], [0., .1]])

grid = RectangularGrid(n)

model = SquaredExponentialModel()   # TODO: try sq. exponential model!!!
model.rho = rho
model.sigma = sigma
model.nugget = nugget


q = np.zeros((500,4))
adj_q = np.zeros((500,4))

p = np.zeros((500,2))
adj_p = np.zeros((500,2))

file_name = f'C5/DeWhittle_{n[0]}x{n[1]}_SquaredExponentialModel.txt'
arr = np.loadtxt(file_name, skiprows=2)[:500]

# stop

# params = arr[:, :2]
q     = arr[:, -8:-6]
adj_q = arr[:, -6:-4]
p     = arr[:, -4:-2]
adj_p = arr[:, -2:]    
   

prior_label = rf'$\rho \sim N({prior_mean[0]}, {np.diag(prior_cov)[0]})$, $\sigma \sim N({prior_mean[1]}, {np.diag(prior_cov)[1]})$ '

fig,ax = plt.subplots(2,2, figsize=(15,10))
fig.suptitle(f'Posterior quantile estimates, {n=}, {model.name}, {prior_label}', fontsize=24)#, fontweight='bold')
ax[0,0].hist(p[:,0], bins='sturges', edgecolor='k')
ax[0,1].hist(p[:,1], bins='sturges', edgecolor='k')

ax[1,0].hist(adj_p[:,0], bins='sturges', edgecolor='k')
ax[1,1].hist(adj_p[:,1], bins='sturges', edgecolor='k')

ax[1,0].set_xlabel( r'$\rho$', fontsize=22)
ax[1,1].set_xlabel( r'$\sigma$', fontsize=22)

ax[0,0].set_title('debiased Whittle', color='r',fontsize=20, x=1.05, y=1.05)
ax[1,0].set_title('Adjusted debiased Whittle', color='r',fontsize=20, x=1.05, y=1.05)

# fig.subplots_adjust(hspace=0.3, wspace=-1.0)
fig.tight_layout()
plt.show()

# stop

from scipy import stats
unif = stats.uniform(0,1)
qs = np.linspace(0,1, 1000)
theory_quants = unif.ppf(qs)

fig,ax = plt.subplots(1,2, figsize=(15,7))
fig.suptitle(f'QQ plot, posterior quantiles vs standard uniform, {n=}, {model.name}, {prior_label}')
ax[0].plot(theory_quants, theory_quants, c='r', linewidth=3, label='standard uniform', zorder=10)
ax[0].plot(theory_quants, np.quantile(p[:,0], qs), 
           '.', c='g', markersize=10., label='dewhittle')
ax[0].plot(theory_quants, np.quantile(adj_p[:,0], qs), 
           '.', c='blue', markersize=10., label='adj dewhittle')
ax[0].legend()
ax

ax[1].plot(theory_quants, theory_quants, c='r', linewidth=3, label='standard uniform', zorder=10)
ax[1].plot(theory_quants, np.quantile(p[:,1], qs), 
           '.', c='g', markersize=10., label='dewhittle')
ax[1].plot(theory_quants, np.quantile(adj_p[:,1], qs), 
           '.', c='blue', markersize=10., label='adj dewhittle')
ax[1].legend()


ax[0].set_xlabel( r'$\rho$', fontsize=22)
ax[1].set_xlabel( r'$\sigma$', fontsize=22)

fig.tight_layout()
plt.show()