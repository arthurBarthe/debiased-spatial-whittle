from math import prod
import numpy as np
from numpy import ndarray
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def plot_marginals(list_draws: list[ndarray,...],
                               truths:None|ndarray=None, 
                               title:None|str=None,
                               axis_labels:None|list=None,
                               density_labels:None|list=None,
                               shape:None|tuple=None,
                               figsize:tuple=(15,7), 
                               cmap:None|str=None, **plotargs):
    
    '''draws: list of arrays of samples to plot'''
    
    dims   = [draws.shape[1] for draws in list_draws]
    nplots = max(dims)
    
    ndistributions = len(list_draws)
    # color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if axis_labels is None:
        axis_labels = list(range(nplots))
    
    if shape is None:
        shape = (1,nplots)
    else:
        assert prod(shape) >= nplots , 'shape not >= dim(theta)'
    
    if cmap is None:
        cmap = 'viridis'
    cm = plt.get_cmap(cmap)
    
    fig,ax_list = plt.subplots(*shape, figsize=figsize)
    for i, ax in enumerate(ax_list.flatten()):
        
        if i>=nplots:
            fig.delaxes(ax_list.flatten()[i])
            continue
        
        if truths is not None:
            ax.axvline(truths[i], c='k', linewidth=3., linestyle = '--',zorder=3)
                 
        for j in range(ndistributions):
            if i>dims[j]-1:
                continue
            sns.kdeplot(list_draws[j][:,i], fill=False, ax=ax, legend=False, linewidth=3., color=cm(j/ndistributions), **plotargs)
                
        
        ax.set_xlabel(axis_labels[i], fontsize=24)
        ax.tick_params(axis='x', labelsize=18)
            
        ax.set_yticks([])
        ax.set_ylabel(' ')             # .axes.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if title is None:
        title = 'Marginal densities'
    fig.suptitle(title, fontsize=26, y=1.10)
    
    if density_labels is None:
        legend_labels = [f'density #{i}' for i in range(ndistributions)]
    else:
        legend_labels = density_labels.copy()
    if truths is not None:
        legend_labels.insert(0, 'True parameter')
    
    # ax_list[0].set_xlim([-1.5,4])     # bounds
    fig.legend(legend_labels, fontsize=20, bbox_to_anchor=(1.05,1.12))
    fig.tight_layout()
    plt.show()
    return




def ridgeline(data, overlap=0, fill=True, labels=None, n_points=150, figsize=(15,7)):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    curves = []
    ys = []
    
    # xx = np.linspace(np.min(np.concatenate(data)),
                      # np.max(np.concatenate(data)), n_points)
    for i, d in enumerate(data):
        pdf = gaussian_kde(d)
        y = i*(1.0-overlap)
        ys.append(y)
        
        xx = np.linspace(np.min(data[i]),
                          np.max(data[i]), n_points)
        density = pdf(xx)
        if fill:
            plt.fill_between(xx, np.ones(n_points)*y, 
                             density+y, zorder=len(data)-i+1, color=fill)
        plt.plot(xx, density+y, c='k', zorder=len(data)-i+1, lw=1.)
    if labels:
        plt.yticks(ys, labels, fontsize=14)
    