import pandas
from sklearn.svm import SVC

from source import create_answer_file

data_train = pandas.read_csv(r'..\data\svm.csv',
                             names=['result', 'p1', 'p2'])

y_train = data_train['result']
X_train = data_train[['p1', 'p2']]

svc = SVC(C=100000, kernel='linear', random_state=241)
fit = svc.fit(X_train, y_train)
result = ' '.join(str(f) for f in (fit.support_ + 1))

create_answer_file('w3_1.txt', f'{result}')
