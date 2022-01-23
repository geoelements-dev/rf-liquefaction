#!/usr/bin/env python
# coding: utf-8
import pickle
from sklearn.ensemble import RandomForestClassifier
from feature_importance import plot_importances

path = './PRJ-2998/Model Usage/'
model_names = ['RF_YN_Model0','RF_YN_Model3','RF_YN_Model5','RF_Displ_Model0','RF_Displ_Model3','RF_Displ_Model5']
models = [pickle.load(open(path+model_name + '.pkl', 'rb')) for model_name in model_names]

fig = plot_importances(models,model_names)
#fig.savefig('figure_name.png')