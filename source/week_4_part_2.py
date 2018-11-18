import numpy as np
import pandas
from sklearn.decomposition import PCA

from source import create_answer_file

train_data = pandas.read_csv(r'..\data\close_prices.csv')
train_data_2 = pandas.read_csv(r'..\data\djia_index.csv')
del train_data['date']

pca = PCA(n_components=10)
pca.fit(train_data)

# определение кол-ва компонентов, чтобы объяснить 90% дисперсии
ratio_sum = 0
count = 0
for index, val in enumerate(pca.explained_variance_ratio_):
    ratio_sum += val
    if ratio_sum >= 0.90:
        count = index + 1
        break

x = pca.transform(train_data)
# получение коэфф. Пирсона
coeff = np.corrcoef(x[:, 0], train_data_2['^DJI'])[0, 1]
company_name = train_data.columns[np.argmax(pca.components_[0])]

create_answer_file('w4_2.txt', f'{count}')
create_answer_file('w4_3.txt', f'{coeff}')
create_answer_file('w4_4.txt', f'{company_name}')
