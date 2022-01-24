import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter


def plot_importances(models, model_names, labels=None, y_limit=(0,0.5)):

    """Plot feature importances of random forest models.

    Args:
        models: Model object or a list of model objects.
        model_names: Model name or a list of name for each model.
        labels: A list of feature name that x labels show in this order. 
        y_limit: The limit of y axis. 

    Return:
        fig: A matplotlib.figure.Figure object.
    """

    if isinstance(models, RandomForestClassifier):
        models = [models]
        model_names = [model_names]
        
    if labels is None:
        labels = np.unique(np.concatenate([model.feature_names_in_ for model in models]))
    
    # X ticks position                       
    x = np.arange(len(labels))

    # Number of models
    num_models = len(models)

    # Width of bars
    width = 1/num_models-0.05

    # Make the center of the bar groups align with the tick
    offset = 0.5

    # create a figure and an axes objects
    fig,ax = plt.subplots()

                           
    for model, model_name in zip(models, model_names):

        importances = model.feature_importances_
        feature_names = model.feature_names_in_

        #Create an array of the coordinates where bars should plot.
        coordinate = np.array([np.where([i==j for i in labels]) for j in feature_names]).reshape(-1)
        rects = ax.bar(coordinate-(num_models/2-offset)*width, importances,width,label=model_name)
        offset+=1

    ax.grid(axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=45)
    ax.set_ylim(y_limit)
    ax.set_ylabel('Relative Importance')
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_axisbelow(True)
    ax.legend()
    plt.show()

    return fig