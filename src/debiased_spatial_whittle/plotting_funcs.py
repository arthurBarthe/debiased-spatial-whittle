from math import prod
from numpy import ndarray
import seaborn as sns
import matplotlib.pyplot as plt

def plot_marginals(list_draws: list[ndarray,...],
                               truths:None|ndarray=None, 
                               title:None|str=None,
                               axis_labels:None|list=None,
                               legend_labels:None|list=None,
                               shape:None|tuple=None,
                               figsize:tuple=(15,7), **plotargs):
    
    '''draws: list of arrays of samples to plot'''
    
    dims   = [draws.shape[1] for draws in list_draws]
    nplots = max(dims)
    
    ndistributions = len(list_draws)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if axis_labels is None:
        axis_labels = list(range(nplots))
    
    if shape is None:
        shape = (1,nplots)
    else:
        assert prod(shape) >= nplots , 'shape not >= dim(theta)'
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
            sns.kdeplot(list_draws[j][:,i], fill=False, ax=ax, legend=False, linewidth=3., color=color_cycle[j], **plotargs)
                
        
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
    
    if legend_labels is None:
        legend_labels = [f'density #{i}' for i in range(ndistributions)]
    if truths is not None:
        legend_labels.insert(0, 'True Parameter')
    fig.legend(legend_labels, fontsize=20, bbox_to_anchor=(1.00,1.12))
    fig.tight_layout()
    plt.show()
    return
    