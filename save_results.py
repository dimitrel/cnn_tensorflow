# -*- coding: utf-8 -*-
"""Parameter initialisers.

This module defines functions to help us store the experiments information.
"""
import sys
import os


def write_to_file(name,write=None):
    name='results/'+name+'.txt'
    f =open(name,'a')
    f.write(write)
    f.close()
    
    
def save_learning_curves(fig_1,fig_2,label=None):
    fig_1.tight_layout()
    fig_1.savefig('results/graphs/'+label+'_error.pdf')
    fig_2.tight_layout()
    fig_2.savefig('results/graphs/'+label+'_accuracy.pdf')