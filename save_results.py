# -*- coding: utf-8 -*-
"""
This module defines functions to help us store the experiment information.
"""

import sys
import os

def write_to_file(name,write=None):
	"""Write to a file for storing experiment information.

	Args:
		name: the name of the file
		write: string to write on the corresponding file
		
	Returns:
		no return value
		
	"""
    name='results/'+name+'.txt'
    f =open(name,'a')
    f.write(write)
    f.close()
    
    
def save_learning_curves(fig_1,fig_2,label=None):
	"""Save plots for storing the learning curves figures.

	Args:
		fig_1: a matplotlib figure
		fig_2: a matplotlib figure
		
	Returns:
		no return value
		
	"""
    fig_1.tight_layout()
    fig_1.savefig('results/graphs/'+label+'_error.pdf')
    fig_2.tight_layout()
    fig_2.savefig('results/graphs/'+label+'_accuracy.pdf')