# -*- coding: utf-8 -*-
"""
The following module is used for plotting the learning curves for a neural networks which provide us with insight 
about the performance and hyperparamee selection.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_lc(train_stats, valid_stats, num_epochs, label=None):
	"""Plot learning curves.

	Args:
		train_stats: tuple of lists with training error and training accuracy values in each epoch
		valid_stats: tuple of lists with validation error and validation accuracy values in each epoch 
		num_epochs: number of the epochs the network is trained for
		label: label for the plots, depending on the experiment (e.g. number of layers, learning rate, etc)
		
	Returns:
		fig_1: matplotlib object with training and validation error figures
		fig_2: matplotlib object with training and validation accuracy figures
		
	"""
    fig_1 = plt.figure(figsize=(12, 6))
    fig_2 = plt.figure(figsize=(12, 6))
    ax1 = fig_1.add_subplot(1, 2, 1)
    ax2 = fig_1.add_subplot(1, 2, 2)
    ax3 = fig_2.add_subplot(1, 2, 1)
    ax4 = fig_2.add_subplot(1, 2, 2)
    x=np.arange(1,num_epochs+1)

    for stats in train_stats.items():
        value=str(stats[0])
        this_label= label+" = "+str(value) 
        ax1.plot(x, stats[1][0], label= this_label )
        ax3.plot(x, stats[1][1], label= this_label )

    for stats in valid_stats.items():
        value=str(stats[0])
        this_label= label+" = "+str(value)
        ax2.plot(x, stats[1][0], label= this_label )
        ax4.plot(x, stats[1][1], label= this_label )

    ax1.legend(loc=0)
    ax1.set_xlabel('Epoch number')
    ax1.set_ylabel('Training error')     
    ax1.set_xlim([1,num_epochs])
    ax1.grid(linestyle='--')
    ax1.legend(prop={'size':12})
    
    ax2.legend(loc=0)
    ax2.set_xlabel('Epoch number')
    ax2.set_ylabel('Validation error')
    ax2.set_xlim([1,num_epochs])
    ax2.legend(prop={'size':12})
    ax2.grid(linestyle='--')

    ax3.legend(loc=0)
    ax3.set_xlabel('Epoch number')
    ax3.set_xlim([1,num_epochs])
    ax3.set_ylabel('Training accuracy')
    ax3.legend(prop={'size':12})
    ax3.grid(linestyle='--')
    
    ax4.legend(loc=0)
    ax4.set_xlabel('Epoch number')
    ax4.set_ylabel('Validation accuracy')
    ax4.set_xlim([1,num_epochs])
    ax4.grid(linestyle='--')
    ax4.legend(prop={'size':12})   

    return fig_1,fig_2