#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# use old version of scikit-learn==0.21.3
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

# Feature order for model0 to model5
feature_names = [
    ['GWT', 'distance', 'Slope'],
    ['GWT', 'distance', 'Slope','PGA'],
    ['GWT', 'Elevation', 'distance', 'Slope'],
    ['GWT', 'Elevation', 'distance', 'Slope', 'PGA'],
    ['GWT', 'distance', 'Slope', 'PGA', 'qtncs', 'qtncs_std', 'Ic', 'Ic_std'],
    ['GWT', 'Elevation', 'distance', 'Slope', 'PGA', 'qtncs', 'qtncs_std', 'Ic', 'Ic_std'],
]



def plot_importances(mdtype, num):
    md_dir = './PRJ-2998/Model Usage/RF_' + mdtype + '_Model'+str(num)+'.pkl'
    md = pickle.load(open(md_dir,'rb'))
    _importances = md.feature_importances_
    _feature_names = feature_names[num]
    indices = np.argsort(_importances)[-10:]  # top 10 features
    plt.title('Feature Importances of ' + mdtype + ' Model '+str(num))
    plt.barh(range(len(indices)), _importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [_feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig('feature_importance_' + mdtype + '_model'+str(num)+'.png')
    plt.show()

# Plot feature importances for each model.
for i in ['YN','Displ']:
    for j in range(6):
        plot_importances(i,j)



# Change it if you want the different order.
label_order = ['distance','GWT','Slope','PGA','Elevation','qtncs','qtncs_std','Ic','Ic_std']

def plot_grouped_chart(mdtype, num_list, labels=label_order):
    x = np.arange(len(labels))
    num_group = len(num_list)
    width=1/num_group-0.05
    fig,ax = plt.subplots()
    offset=0.5
    fname=mdtype+'_'
    for num in num_list:
        md_dir = './PRJ-2998/Model Usage/RF_'+mdtype+'_Model'+str(num)+'.pkl'
        md = pickle.load(open(md_dir, 'rb'))
        _importances = md.feature_importances_
        _feature_names = feature_names[num]
        rects = ax.bar(np.array([np.where([i==j for i in labels]) for j in _feature_names]).reshape(-1)-(num_group/2-offset)*width,_importances,width,label=mdtype+'_Model'+str(num))
        offset+=1
        fname+=str(num)

    # figure formating
    ax.grid(axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=45)
    ax.set_ylim(0,0.5)
    ax.set_ylabel('Relative Importance')
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_axisbelow(True)
    ax.legend()
    plt.savefig(fname+'.png')
    plt.show()

# Plot feature importances in grouped bar chart
plot_grouped_chart('YN',[0,1,2,3,5])