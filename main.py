import argparse
import classify
from sklearn.externals import joblib

parser = argparse.ArgumentParser(description='Baseline Code-switching Classifier')

parser.add_argument('--train', action='store_true', default=False, help='enable train')
parser.add_argument('--validate', action='store_true', default=False, help='enable validation')
parser.add_argument('--test', action='store_true', default=False, help='enable test')

parser.add_argument('--save_path', type=str, default="models/model.joblib", help='Path to dump model to, must end in .joblib')
parser.add_argument('--load_path', type=str, default="models/model.joblib", help='Path to load model from, must end in .joblib')

args = parser.parse_args()

if __name__ == '__main__':

    # TODO: load data (probably within the ifs)

    if args.load_path:
        try:
            model = joblib.load(args.load_path)
        except :
            print("Sorry, model not found."); exit()

    if args.train:
        model = classify.fit(X, Y, args.save_path)
    elif args.validate:
        Y_pred = classify.predict(X, model)
    elif args.test:
        Y_pred = classify.predict(X, model)

