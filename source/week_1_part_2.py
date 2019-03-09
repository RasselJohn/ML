import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from source import create_answer_file

data = pandas.read_csv(r'..\data\titanic.csv', index_col='PassengerId')

# т.к. столбец содержит строковые значения - приводим к числовому виду
label = LabelEncoder()
dicts = {}
# задаем список значений для кодирования
label.fit(data.Sex.drop_duplicates())
dicts['Sex'] = list(label.classes_)
# заменяем значения из списка кодами закодированных элементов
data.Sex = label.transform(data.Sex)

# получение столбцов - признаков
props_columns = ['Pclass', 'Fare', 'Age', 'Sex']
x = data[props_columns]

# замена null-значения на 0
x = x.fillna(0)

# получение столбца результатов
result_column = ['Survived']
y = data[result_column]

# построение решающего дерева
tree = DecisionTreeClassifier(random_state=241)
# обучение дерева
tree.fit(x, y)

# получение важности каждого из столбцов
feature_importances = list(tree.feature_importances_)

# 2 столбца, влияющие больше всего на итоговый результат
ind1 = tree.feature_importances_.argsort()[-2]
ind2 = tree.feature_importances_.argsort()[-1]

create_answer_file('w1_7.txt', f'{props_columns[ind1]} {props_columns[ind2]}')
