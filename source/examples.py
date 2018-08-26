from urllib.request import urlopen
import numpy as np

f = urlopen('https://stepik.org/media/attachments/lesson/16462/boston_houses.csv')
data = np.loadtxt(f, delimiter=',', skiprows=1)

y = data[:, 0]
y = y.reshape(y.shape[0], 1)

X = np.hstack((np.ones_like(y), data[:, 1:]))
X_T = X.T

left_part = np.linalg.inv(X_T.dot(X))
right_part = X_T.dot(y)
result = left_part.dot(right_part)
result = result.reshape((result.shape[0],))
result = ' '.join([str(r) for r in result])
print(result)
