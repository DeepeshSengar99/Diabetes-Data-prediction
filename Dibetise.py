import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
print('Shape of diabetes :',diabetes.data.shape)


X = diabetes.data[:,np.newaxis,2]
print('Shape of data sets :',X.shape)


Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,diabetes.target,random_state =3)
# creat linear regression object
SimpleRegression = LinearRegression()

# train the model using training stets
SimpleRegression.fit(Xtrain,Ytrain)

Ypred = SimpleRegression.predict(Xtest)
print('Root mean square error: ',np.sqrt(mean_squared_error(Ytest,Ypred)))

# model score 

print('Regression score function:',r2_score(Ytest,Ypred))

# cofficient and

print('Weight : %.2f'%SimpleRegression.coef_[0])
print('Bias accuracy :%.2f'%SimpleRegression.intercept_)

# model Accuracy
print('Model Accuracy : %.2f'%(SimpleRegression.score(Xtest,Ytest)*100))

# plot output

plt.scatter(Xtest,Ytest,color ='red',s=10)
plt.plot(Xtest,Ypred,color ='blue')
plt.title('Simple Regression')
plt.show()

















