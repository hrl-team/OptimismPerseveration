#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:53:27 2024
Statistics computation
@author: isabelle
"""

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

#%% import datas
SP=pd.read_pickle('../results/reboot/SP_results_uniform_clean.pkl')

init='uniform'

gaussian=pd.read_pickle(f'../results/reboot/Gaussian_results_{init}_clean.pkl')
rdwalk  =pd.read_pickle(f'../results/reboot/RdWalk_results_{init}_clean.pkl')
reversal=pd.read_pickle(f'../results/reboot/Reversal_results_{init}_clean.pkl')

learning=pd.read_pickle(f'../results/reboot/LearningPeriod_results_{init}_clean.pkl')
difficulty=pd.read_pickle(f'../results/reboot/Difficulty_results_{init}_clean.pkl')
rich_poor=pd.read_pickle(f'../results/reboot/Difficulty_results_{init}_clean.pkl')



#%% filter data and order
keys=[(0.15,0.85),(0.35,0.65),(0.05, 0.55), (0.15, 0.65), (0.35, 0.85), (0.45, 0.95)]
for k in keys:
    del difficulty[k]
difficulty=dict(sorted(difficulty.items()))
    
keys=[(0.15,0.65),(0.35,0.85), (0.05, 0.95), (0.15, 0.85), (0.35, 0.65), (0.45, 0.55)]
for k in keys:
    del rich_poor[k]
rich_poor=dict(sorted(rich_poor.items()))

keys=[(4,40),(16,10)]
for k in keys:
    del gaussian[k]
    del rdwalk[k]
    del reversal[k]
    del learning[k]
gaussian=dict(sorted(gaussian.items()))
rdwalk=dict(sorted(rdwalk.items()))
reversal=dict(sorted(reversal.items()))
learning=dict(sorted(learning.items()))

#%% functions
def compute_biases(data,k):
    cur_dat=data[k]['params']
    perseveration=len(cur_dat[cur_dat[:,4]>0])
    alternation  =len(cur_dat[cur_dat[:,4]<0])
    positivity   =len(cur_dat[cur_dat[:,1]>cur_dat[:,2]])
    negativity   =len(cur_dat[cur_dat[:,1]<cur_dat[:,2]])
    return perseveration, alternation, positivity, negativity

data=rich_poor
for k in data.keys():
    print(k,compute_biases(data,k))
