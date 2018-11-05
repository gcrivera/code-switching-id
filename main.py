import argparse
from scripts import classify
from scripts import evaluate
from scripts import prepare_data

parser = argparse.ArgumentParser(description='Baseline Code-switching Classifier')

parser.add_argument('--train', action='store_true', default=False, help='enable train')
parser.add_argument('--validate', action='store_true', default=False, help='enable validation')
parser.add_argument('--test', action='store_true', default=False, help='enable test')

parser.add_argument('--save_path', type=str, default="models/default/", help='Path to dump models to')
parser.add_argument('--load_path', type=str, help='Path to load model from, must end in .joblib')

parser.add_argument('--confusion_matrix', action='store_true', default=False, help='Display confusion matrix if validating or testing')

args = parser.parse_args()

if __name__ == '__main__':

    if args.train_ubm:
        data = prepare_data.load_train_file()
        ubm = classify.fit_ubm(data, args.save_path)
        classify.fit_adap(data, ubm, args.save_path)
        args.load_path = args.save_path
    if args.train:
        data = prepare_data.load_train_file()
        classify.fit(data, args.save_path)
        args.load_path = args.save_path
    if args.validate:
        data = prepare_data.load_test('val')
        scores = classify.predict(data, args.load_path)
        Y,Y_pred = evaluate.get_predictions(scores)
        evaluate.evaluate(Y, Y_pred)
        if args.confusion_matrix:
            evaluate.confusion_matrix(Y, Y_pred)
    if args.test:
        data = prepare_data.load_test('test')
        scores = classify.predict(data, args.load_path)
        Y,Y_pred = evaluate.get_predictions(scores)
        evaluate.evaluate(Y, Y_pred)
        if args.confusion_matrix:
            evaluate.confusion_matrix(Y, Y_pred)

