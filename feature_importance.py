#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter


def plot_importances(models, model_names, labels=None):
    #models: an model object or a list of model objects. 
    #model_names: a string or a list of strings corresponding to the models.
    #labels: a list of strings. The label order shown in x axis.
    if isinstance(models, RandomForestClassifier):
        models = [models]
        model_names = [model_names]
        
    if labels is None:
        labels = np.unique(np.concatenate([model.feature_names_in_ for model in models]))
                           
    x = np.arange(len(labels))
    model_count = len(models)
    width=1/model_count-0.05
    fig,ax = plt.subplots()
    offset=0.5
                           
    for model, model_name in zip(models, model_names):

        importances = model.feature_importances_
        feature_names = model.feature_names_in_
        
        coordinate = np.array([np.where([i==j for i in labels]) for j in feature_names]).reshape(-1)
        rects = ax.bar(coordinate-(model_count/2-offset)*width, importances,width,label=model_name)
        offset+=1

    ax.grid(axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=45)
    ax.set_ylim(0,0.5)
    ax.set_ylabel('Relative Importance')
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_axisbelow(True)
    ax.legend()
    plt.show()

    #return a matplotlib.figure.Figure object, in case you want to save the figure.
    return fig