#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:55:03 2023
This code generate Fig 4a using "Source_data_Fig_4.xlsx. 

@author: wuy19
"""
import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Set paths
root_dir = '/Users/yh520/NatCommun_Wu2024'
data_dir = os.path.join(root_dir, 'Data')
fig_dir = os.path.join(root_dir, 'Figures')

# Change working directory to the figures directory
os.chdir(root_dir + '/Figures')

# Labesl for plots
xlabels = ['Hit Rate', 'False Alarm Rate', 'Criterion', 'Sensitiviy']
ylims = [[0.45,0.55],
         [.15,.3],
         [.30,.55],
         [.6,1.1]]
ylabels = [['0.46', '0.48', '0.50', '0.52', '0.54'],
           ['0.15', '0.20', '0.25', '0.30'],
           ['0.3', '0.35', '0.4', '0.45', '0.5'],
           ['0.6', '0.7', '0.8', '0.9', '1.0']]
behaviors = ['HR', 'FAR', 'c', 'dprime']

  
def perform_ols_regression(x, y):
    """Perform Ordinary Least Squares (OLS) regression."""
    x = sm.add_constant(x)
    return sm.OLS(y, x).fit()
    
def plot_behavior_metric(i, df, bhv, group_mean, std, model):
    """Plot a behavior metric with linear regression and error bars."""   
    fig, ax = plt.subplots(1, 1, figsize=(2, 1.3))
    
    # Plot regression line
    x_vals = np.arange(1.8,3.4,.2)
    ax.plot(x_vals, model.params[0] + model.params[1] * x_vals,color='gray', linewidth=1)
    
    # Plot errorbars
    x_ticks = np.arange(2,3.1, .2)
    ax.errorbar(x_ticks, group_mean, yerr = std, capsize = 2, ecolor='gray', linestyle='')
    
    # Plot group means
    ax.plot(x_ticks, group_mean, marker = 'o', linestyle='', color = 'black', markersize = 4,
             markeredgecolor = 'black', markerfacecolor = 'gray')

    # Configure plot appearance
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('outward', 5))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 5))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
     
    # Set x, y ticks and labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['2','2.2', '2.4', '2.6', '2.8', '3'], fontsize=7)
    ax.set_yticks([float(y) for y in ylabels[i]])
    ax.set_yticklabels(ylabels[i], fontsize=7)
                 
    plt.xlim([1.9,3.1])
    plt.ylim(ylims[i])
    plt.xlabel('SD of the response distributions (sigma)', fontsize= 7)
    plt.ylabel(xlabels[i], fontsize=7)
    plt.show()

def save_figure(fig, path):
    """Save the figure to a specified path."""
    fig.savefig(f'{path}.svg', bbox_inches='tight', dpi=600, transparent=True)
    fig.savefig(f'{path}.png', bbox_inches='tight', dpi=600, transparent=True)   

def main():
    # Load source data
    df_simulation = pd.read_excel(data_dir + '/Source_data_Fig_4.xlsx')

    # Loop through each behavior metric and create plots
    for i, bhv in enumerate(behaviors):
        group_mean = df_simulation.groupby('sd')[bhv].mean().values
        std = df_simulation.groupby('sd')[bhv].std(ddof=1).values
        x = df_simulation['sd'].values
        y = df_simulation[bhv].values
            
        model = perform_ols_regression(x, y)
        plot_behavior_metric(i, df_simulation, bhv, group_mean, std, model)
        #save_figure(fig, fig_dir + '_' + bhv)