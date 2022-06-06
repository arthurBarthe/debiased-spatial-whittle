# This example implements the real-data example from the paper. The data used are remote-sensing measurements of Venus'
# topography.

import numpy as np
from debiasedwhittle import sim_circ_embedding, fit, matern
import matplotlib.pyplot as plt
import scipy.io

# load the topography data and standardize by the std
z = scipy.io.loadmat('Frederik53.mat')['topodata']
z = z / np.std(z)

# plot
plt.figure()
plt.imshow(z)
plt.show()

init_guess = np.array([50, .5, 1])

cov = matern
est = fit(z, np.ones_like(z), cov, init_guess, fold=True)
print(est)
