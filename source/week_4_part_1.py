# Линейные методы регрессии
# Гребневая регрессия
# Корректировка данных
# Числовое представление слов

import pandas
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from source import create_answer_file

train_data = pandas.read_csv(r'..\data\salary-train.csv')
test_data = pandas.read_csv(r'..\data\salary-test-mini.csv')

# корректировка данных
handle = lambda x: x.str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)
train_data['FullDescription'] = handle(train_data['FullDescription'])
train_data['LocationNormalized'] = handle(train_data['LocationNormalized'])
train_data['ContractTime'] = handle(train_data['ContractTime'])

# числовое представление слов методом TF-IDF
vector = TfidfVectorizer(min_df=5)
x_train_vector = vector.fit_transform(train_data['FullDescription'])
x_test_vector = vector.transform(test_data['FullDescription'])

# замена пропущенных значений на специальные строковые величины('nan')
train_data['LocationNormalized'].fillna('nan', inplace=True)
train_data['ContractTime'].fillna('nan', inplace=True)

# признаки LocationNormalized и ContractTime являются строковыми,
# и поэтому с ними нельзя работать напрямую - используем DictVectorizer
enc = DictVectorizer()
X_train_categ = enc.fit_transform(
    train_data[['LocationNormalized', 'ContractTime']].to_dict('records')
)
X_test_categ = enc.transform(test_data[['LocationNormalized', 'ContractTime']].to_dict('records'))

# объединение данных
X_train = hstack([x_train_vector, X_train_categ])
X_test = hstack([x_test_vector, X_test_categ])

# обучение модели
clf = Ridge(alpha=1, random_state=241)
clf.fit(X_train, train_data['SalaryNormalized'])
# определение SalaryNormalized для тестовых данных
result = clf.predict(X_test)
create_answer_file('w4_1.txt', f'{round(result[0],2)} {round(result[1],2)}')
