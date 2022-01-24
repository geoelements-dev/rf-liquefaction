import argparse
import pickle
from sklearn.ensemble import RandomForestClassifier
from feature_importance import plot_importances

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Input a directory path where the models are.")
args = parser.parse_args()

model_names = ['model_name1', 'model_name2', 'model_name3']
models = [pickle.load(open(args.path+'/'+model_name + '.pkl', 'rb')) for model_name in model_names]

fig = plot_importances(models,model_names)
#fig.savefig('figure_name.png')