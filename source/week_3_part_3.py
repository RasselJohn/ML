import numpy as np
import pandas
from sklearn.metrics import roc_auc_score

from source import create_answer_file


def get_distance(x1, x2, y1, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** .5


def get_roc_auc_score(X, y, C):
    w1 = w2 = 0
    k = 0.1
    y_len = np.size(y)
    distance = 10

    i = 0
    # градиентный спуск
    while distance > 1e-05 and i < 10000:
        sum_1 = sum_2 = 0
        old_w1 = w1
        old_w2 = w2

        for i in range(y_len):
            a = 1 - (1 / (1 + np.exp(- y[i] * (w1 * X['p1'][i] + w2 * X['p2'][i]))))
            sum_1 = sum_1 + y[i] * X['p1'][i] * a
            sum_2 = sum_2 + y[i] * X['p2'][i] * a

        w1 = w1 + k / y_len * sum_1 - k * C * w1
        w2 = w2 + k / y_len * sum_2 - k * C * w2
        distance = get_distance(old_w1, w1, old_w2, w2)
        i += 1

    return round(roc_auc_score(y, 1 / (1 + np.exp(-w1 * X['p1'] - w2 * X['p2']))), 3)


data = pandas.read_csv(r'..\data\data-logistic.csv',
                       names=['result', 'p1', 'p2'])
y = data['result']
X = data[['p1', 'p2']]

result_1 = get_roc_auc_score(X, y, 0)
result_2 = get_roc_auc_score(X, y, 10)

create_answer_file('w3_3.txt', f'{result_1} {result_2}')
