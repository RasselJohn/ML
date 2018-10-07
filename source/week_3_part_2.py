import numpy as np

from sklearn import datasets
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from source import create_answer_file

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)

# числовое представление слов методом TF-IDF
vector = TfidfVectorizer()
vector_fit = vector.fit_transform(newsgroups.data)
vector_mapping = vector.get_feature_names()

# подбор оптимального коэффициента для классификатора
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(vector_fit, newsgroups.target)

# обучение модели с лучшим коэффициентом
clf2 = SVC(**gs.best_params_, kernel='linear', random_state=241)
clf2.fit(vector_fit, newsgroups.target)

# какие слова встречаются чаще в 2-х заданных темах
result = [
    vector_mapping[r] for r in
    (np.absolute(clf2.coef_.toarray()[0]).argsort()[-10:][::-1])
]
result.sort()

create_answer_file('w3_2.txt', f'{" ".join(result)}')
