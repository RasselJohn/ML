import pandas
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

from source import create_answer_file

data = pandas.read_csv(r'..\data\wine.data',
                       names=['sort', 'Alcohol', 'acid', 'Ash',
                              'Alcalinity', 'Magnesium', 'phenols',
                              'Flavanoids', 'Nonflavanoid', 'Proanthocyanins',
                              'intensity', 'Hue', 'diluted', 'Proline'])
y = data['sort']
X = data[['Alcohol', 'acid', 'Ash',
          'Alcalinity', 'Magnesium', 'phenols',
          'Flavanoids', 'Nonflavanoid', 'Proanthocyanins',
          'intensity', 'Hue', 'diluted', 'Proline']]

result = []
for k in range(1, 51):
    # определили классификатор
    clf = KNeighborsClassifier(n_neighbors=k)

    # определили разбиение
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    # проверяем как хорошо для каждого разбиения происходят предсказания
    scores = cross_val_score(clf, X, y, cv=k_fold)
    result.append(scores.mean())

max_mean = max(result)
create_answer_file('w2_1.txt', f'{result.index(max_mean) + 1}')
create_answer_file('w2_2.txt', f'{max_mean}')

# с маштабированием
X_scale = scale(X)
result_scale = []
for k in range(1, 51):
    # определили классификатор
    clf = KNeighborsClassifier(n_neighbors=k)
    # определили разбиение
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    # проверяем как хорошо для каждого разбиения происходят предсказания
    scores_scale = cross_val_score(clf, X_scale, y, cv=k_fold)
    result_scale.append(scores_scale.mean())

max_mean_scale = max(result_scale)
create_answer_file('w2_3.txt', f'{result_scale.index(max_mean_scale) + 1}')
create_answer_file('w2_4.txt', f'{max_mean_scale}')
