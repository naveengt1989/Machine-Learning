import math
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
Xt = []
Y = []
data = open('input.txt')
for line in data:
    xi, yi = map(float, line.split(','))
    Xt.append(xi)
    Y.append(yi)
data.close()

X =[]
for i in Xt:
	tmp = []
	tmp.append(i)
	tmp.append(1)
	X.append(tmp)	

model = linear_model.LinearRegression(fit_intercept=False)
model.fit(X, Y)
print model.coef_ # Polynomial equation consists of single term. Print coefficients of it.
data = []
# Linear equation in form: y = ax + b. data = [x 1] => model.coef contains [a b]
data.append(float(raw_input()))
data.append(1) 
print np.dot(data,model.coef_)
print model.predict(data)
