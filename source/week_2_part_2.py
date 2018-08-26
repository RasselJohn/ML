import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

from source import create_answer_file

boston_data = load_boston()
X = boston_data.data
y = boston_data.target

# с маштабированием
X_scale = scale(X)
result_scale = []
result_p = []
for p in np.linspace(1, 10, num=200):
    # определили классификатор
    clf = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
    # определили разбиение
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    # проверяем как хорошо для каждого разбиения происходят предсказания
    scores_scale = cross_val_score(clf, X_scale, y, cv=k_fold, scoring='mean_squared_error')
    result_scale.append(scores_scale.mean())
    result_p.append(p)

max_mean_scale = max(result_scale)
create_answer_file('w2_5.txt', f'{result_p[result_scale.index(max_mean_scale)]}')
