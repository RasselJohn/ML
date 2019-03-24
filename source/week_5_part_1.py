# Композиция алгоритмов
# Случайный лес
# Регрессия

import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

from source import create_answer_file

data = pandas.read_csv(r'..\data\abalone.csv')

# приводим поле к числовому виду
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = data[[col for col in data.columns if col != 'Rings']]
y = data['Rings']

for estimators_count in range(1, 51):
    clf = RandomForestRegressor(n_estimators=estimators_count, random_state=1)
    clf.fit(X, y)
    k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = cross_val_score(clf, X, y, cv=k_fold, scoring='r2')

    if scores.mean() > 0.52:
        create_answer_file('w5_1.txt', f'{estimators_count}')
        break
