import argparse
from scripts import classify
from scripts import evaluate
from scripts import prepare_data
from sklearn.externals import joblib

parser = argparse.ArgumentParser(description='Baseline Code-switching Classifier')

parser.add_argument('--train', action='store_true', default=False, help='enable train')
parser.add_argument('--validate', action='store_true', default=False, help='enable validation')
parser.add_argument('--test', action='store_true', default=False, help='enable test')

parser.add_argument('--save_path', type=str, default="models/model.joblib", help='Path to dump model to, must end in .joblib')
parser.add_argument('--load_path', type=str, help='Path to load model from, must end in .joblib')

parser.add_argument('--confusion_matrix', action='store_true', default=False, help='Display confusion matrix if validating or testing')

args = parser.parse_args()

if __name__ == '__main__':

    X,Y = prepare_data.load('val')

    if args.load_path:
        try:
            model = joblib.load(args.load_path)
        except :
            print("Model not found."); exit()

    # if args.train:
    #     X,Y = prepare_data.load('train')
    #     model = classify.fit(X, Y, args.save_path)
    # elif args.validate:
    #     X,Y = prepare_data.load('val')
    #     Y_pred = classify.predict(X, model)
    #     evaluate.evaluate(Y, Y_pred)
    #     if args.confusion_matrix:
    #         evaluate.confusion_matrix(Y, Y_pred)
    # elif args.test:
    #     X,Y = prepare_data.load('test')
    #     Y_pred = classify.predict(X, model)
    #     evaluate.evaluate(Y, Y_pred)
    #     if args.confusion_matrix:
    #         evaluate.confusion_matrix(Y, Y_pred)

