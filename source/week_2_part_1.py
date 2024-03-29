# Метрические методы классификации
# Метод ближайших соседей
# Кросс-валидация(разбиение)
# Маштабирование признаков
import pandas
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

from source import create_answer_file


def get_means(X, y):
    means = []
    for k in range(1, 51):
        # определили классификатор
        clf = KNeighborsClassifier(n_neighbors=k)

        # определили разбиение
        k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

        # проверяем как хорошо для каждого разбиения происходят предсказания
        scores = cross_val_score(clf, X, y, cv=k_fold)
        means.append(scores.mean())

    return means


data = pandas.read_csv(
    r'..\data\wine.data',
    names=['sort', 'Alcohol', 'acid', 'Ash',
           'Alcalinity', 'Magnesium', 'phenols',
           'Flavanoids', 'Nonflavanoid', 'Proanthocyanins',
           'intensity', 'Hue', 'diluted', 'Proline']
)
y = data['sort']
X = data[['Alcohol', 'acid', 'Ash',
          'Alcalinity', 'Magnesium', 'phenols',
          'Flavanoids', 'Nonflavanoid', 'Proanthocyanins',
          'intensity', 'Hue', 'diluted', 'Proline']]

result = get_means(X, y)
max_mean = max(result)
create_answer_file('w2_1.txt', f'{result.index(max_mean) + 1}')
create_answer_file('w2_2.txt', f'{max_mean}')

# с маштабированием
X_scale = scale(X)
result_scale = get_means(X_scale, y)
max_mean_scale = max(result_scale)
create_answer_file('w2_3.txt', f'{result_scale.index(max_mean_scale) + 1}')
create_answer_file('w2_4.txt', f'{max_mean_scale}')
