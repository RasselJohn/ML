# Линейные методы классификации
# Персептрон
# Метрики качества
# Стандартизация признаков

import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from source import create_answer_file

data_train = pandas.read_csv(r'..\data\perceptron-train.csv', names=['result', 'p1', 'p2'])
data_test = pandas.read_csv(r'..\data\perceptron-test.csv', names=['result', 'p1', 'p2'])

y_train = data_train['result']
X_train = data_train[['p1', 'p2']]

y_test = data_test['result']
X_test = data_test[['p1', 'p2']]

clf_train = Perceptron(random_state=241)
# обучение персептрона
clf_train.fit(X_train, y_train)
predictions_train = clf_train.predict(X_test)
# определение качества предсказания перцептрона
acc_score_train = accuracy_score(y_test, predictions_train)

# стандартизация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# обучение персептрона после стандартизации признаков
clf_test = Perceptron(random_state=241)
clf_test.fit(X_train_scaled, y_train)
predictions_test = clf_test.predict(X_test_scaled)
acc_score_test = accuracy_score(y_test, predictions_test)

create_answer_file('w2_6.txt', f'{round(acc_score_test - acc_score_train,3)}')
