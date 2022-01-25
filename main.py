import argparse
import pickle
from sklearn.ensemble import RandomForestClassifier
from feature_importance import plot_importances

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Input a directory path where the models are.")
parser.add_argument("model_names", nargs='+', type=str, help="Input model names")
parser.add_argument("-s", "--save", help="Input the name of figure")
args = parser.parse_args()

models = [pickle.load(open(args.path+'/'+model_name + '.pkl', 'rb')) for model_name in args.model_names]
fig = plot_importances(models,args.model_names)

if args.save is not None:
	fig.savefig(args.save+'.png')
