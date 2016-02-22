import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
# https://www.hackerrank.com/challenges/predicting-office-space-price
#Description: Given training feature vectors, form enhanced feature vectors of polynomial degree k and use them to predict the value of given test features. Using 2 learning techniques: LINEAR MODEL, SVM

def form_polynomial(v,k,index,cur,res): # generate enhanced linear feature vector of polynomial order k.
    if index==len(v):
    	res.append(cur)
        return
    for i in range(k+1):
        form_polynomial(v,k-i,index+1,cur*(v[index]**i),res)
 
def gen_features_vector(v,k):
    res = []
    form_polynomial(v, k,0,1,res)
    return res

(cols,rows) = map(int, raw_input().split(' '))
features= []
result =[]
poly = PolynomialFeatures(degree=4) # Enhance feature vector with polynomial degree 4
for i in range(rows):
	given = map(float, raw_input().split(' '))
	#features.append(gen_features_vector(given[:-1],3)) 
	features.append(poly.fit_transform(given[:-1])[0]) # generates enhanced polynomial feature vector of degree 4
	result.append(given[-1])

model = linear_model.LinearRegression(fit_intercept=False) # learning method: LINEAR MODEL
#model = svm.SVR(kernel='rbf', C=1e5) # # learning method: SVM

model.fit(features,result)

# test case features

T = int(raw_input())
for i in range(T):
	given = map(float, raw_input().split(' '))
	#print model.coef_
	#print np.dot(gen_features_vector(given,3),model.coef_) # multiple given test data with learning model coefficients 
	#print model.predict(gen_features_vector(given,3)) # enhance given test data and predict value based on trained model
	print round(model.predict(poly.fit_transform(given)[0]),2) # enhance given test data and predict value based on trained model
