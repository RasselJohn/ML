# Композиция алгоритмов
# Градиентный бустинг
# Подбор гиперпараметров

import matplotlib.pyplot as plot
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss

from source import create_answer_file


def log_loss_results(model, X, y):
    # staged_decision_function - для предсказания качества
    # на обучающей и тестовой выборке на каждой итерации.
    return [
        log_loss(y, [1.0 / (1.0 + np.exp(-y_pred)) for y_pred in pred])
        for pred in model.staged_decision_function(X)
    ]


def create_plots(learning_rate, test_loss, train_loss):
    # График значений log-loss на обучающей и тестовой выборках.
    plot.figure()
    plot.plot(test_loss, 'r', linewidth=2)
    plot.plot(train_loss, 'g', linewidth=2)
    plot.legend(['test', 'train'])
    plot.savefig('../images/rate_' + str(learning_rate) + '.png')


def get_min_loss(test_loss):
    # Минимальное значение метрики и номер итерации, на которой оно достигается.
    min_loss_value = min(test_loss)
    min_loss_index = test_loss.index(min_loss_value)
    return min_loss_value, min_loss_index


def model_test(learning_rate):
    model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=250,
                                       verbose=True, random_state=241)
    model.fit(X_train, y_train)

    train_loss = log_loss_results(model, X_train, y_train)
    test_loss = log_loss_results(model, X_test, y_test)
    create_plots(learning_rate, test_loss, train_loss)
    return get_min_loss(test_loss)


data = pandas.read_csv(r'..\data\gbm-data.csv')
X = data.loc[:, 'D1':'D1776'].values
y = data['Activity'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

min_loss_results = {
    learning_rate: model_test(learning_rate)
    for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]
}

# переобучение (overfitting) или недообучение (underfitting) - определяется по графикам
create_answer_file('w5_2.txt', 'overfitting')

# минимальное значение log-loss и номер итерации,
# на котором оно достигается, при learning_rate = 0.2.
min_loss_value, min_loss_index = min_loss_results[0.2]
create_answer_file('w5_3.txt', '{:0.2f} {}'.format(min_loss_value, min_loss_index))

# RandomForestClassifier с количеством деревьев, равным количеству итераций, на котором
# достигается наилучшее качество у градиентного бустинга
model = RandomForestClassifier(n_estimators=min_loss_index, random_state=241)
model.fit(X_train, y_train)
test_loss = log_loss(y_test, model.predict_proba(X_test)[:, 1])
create_answer_file('w5_4.txt', f'{test_loss}')
