# Обзор основных возможностей pandas
from collections import Counter

import pandas

from source.utils import create_answer_file

data = pandas.read_csv(r'..\data\titanic.csv', index_col='PassengerId')
group = data.groupby('Sex').Sex.agg([len])

# Какое количество мужчин и женщин ехало на корабле?
create_answer_file('w1_1.txt', f'{group.len.male} {group.len.female}')

# Какой части пассажиров удалось выжить?
# Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен),
# округлив до двух знаков.
create_answer_file('w1_2.txt', f'{round(data.sum().Survived / len(data) * 100, 2)}')

# Какую долю пассажиры первого класса составляли среди всех пассажиров?
create_answer_file(
    'w1_3.txt',
    f'{round((data.groupby("Pclass").Pclass.agg([len]).len[1]) / len(data) * 100, 2)}'
)

# Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
create_answer_file('w1_4.txt', f'{round(data.Age.mean(), 2)} {round(data.Age.median(), 2)}')

# Коррелируют ли число братьев/сестер/супругов с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
create_answer_file('w1_5.txt', f'{data.SibSp.corr(data.Parch)}')

# Какое самое популярное женское имя на корабле?
# Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name).
names = []
for n in data.Name:
    if 'Mrs.' in n:
        try:
            names.append(n.split('(')[1])
        except IndexError:
            pass
    elif 'Miss' in n:
        names.append(n.split(' ')[2])

create_answer_file('w1_6.txt', Counter(names).most_common(1)[0][0])
