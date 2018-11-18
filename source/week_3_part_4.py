import numpy as np
import pandas
from sklearn import metrics as mt
from source import create_answer_file


def get_max_accuracy(precision_data):
    accuracy = precision_data[0]
    recall = precision_data[1]
    result = [a for r, a in zip(recall, accuracy) if r >= 0.7]
    return max(result)


data = pandas.read_csv(r'..\data\classification.csv')

# выборка из данных по условию и подсчёт
TP = len(data.loc[(data['true'] == 1) & (data['pred'] == 1)])
FP = len(data.loc[(data['true'] == 0) & (data['pred'] == 1)])
FN = len(data.loc[(data['true'] == 1) & (data['pred'] == 0)])
TN = len(data.loc[(data['true'] == 0) & (data['pred'] == 0)])
create_answer_file('w3_4.txt', f'{TP} {FP} {FN} {TN}')

accuracy_score = round(mt.accuracy_score(data['true'], data['pred']), 3)
precision_score = round(mt.precision_score(data['true'], data['pred']), 3)
recall_score = round(mt.recall_score(data['true'], data['pred']), 3)
f1_score = round(mt.f1_score(data['true'], data['pred']), 3)

create_answer_file('w3_5.txt', f'{accuracy_score} {precision_score} {recall_score} {f1_score}')

data = pandas.read_csv(r'..\data\scores.csv')
scores = {
    'score_logreg': round(mt.roc_auc_score(data['true'], data['score_logreg']), 3),
    'score_svm': round(mt.roc_auc_score(data['true'], data['score_svm']), 3),
    'score_knn': round(mt.roc_auc_score(data['true'], data['score_knn']), 3),
    'score_tree': round(mt.roc_auc_score(data['true'], data['score_tree']), 3)
}
create_answer_file('w3_6.txt', f'{max(scores, key=scores.get)}')

curves = {
    'score_logreg': get_max_accuracy(mt.precision_recall_curve(data['true'], data['score_logreg'])),
    'score_svm': get_max_accuracy(mt.precision_recall_curve(data['true'], data['score_svm'])),
    'score_knn': get_max_accuracy(mt.precision_recall_curve(data['true'], data['score_knn'])),
    'score_tree': get_max_accuracy(mt.precision_recall_curve(data['true'], data['score_tree'])),
}

create_answer_file('w3_7.txt', f'{max(curves, key=curves.get)}')
pass
