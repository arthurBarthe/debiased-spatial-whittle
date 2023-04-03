from numpy import prod
import seaborn as sns
import matplotlib.pyplot as plt

def plot_marginals(list_draws, truths, title, axis_labels, legend_labels, shape=None, **plotargs):
    '''draws: list of arrays of samples to compare'''
    nplots=len(axis_labels)
    
    ndistributions = len(list_draws)
    dims = [draws.shape[1] for draws in list_draws]
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if shape is None:
        shape = (1,nplots)
    else:
        assert prod(shape) >= nplots , 'shape not >= dim(theta)'
    fig,ax_list = plt.subplots(*shape, figsize=(15,7))
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
    
    fig.suptitle(title, fontsize=26, y=1.10)
    if truths is not None:
        legend_labels = ['True Parameter'] + legend_labels        
    fig.legend(legend_labels, fontsize=20, bbox_to_anchor=(1.00,1.12))
    fig.tight_layout()
    plt.show()
    