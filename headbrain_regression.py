## **SIMPLE LINEAR REGRESSION**

#PREDICTION of BRAIN SIZE ON HEAD SIZE 
#steps:
#1. Import related modules 
#2. Reading the data
#3. Splitting the data into training and test
#4. Fitting simple linear regression to the traing set
#5. Predict the test result
#6. See the relationship b/w the training data values using scatter diagram
#7. Calculate RMSE value

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

import io
data = pd.read_csv(io.BytesIO(uploaded['headbrain.csv']))
print(data.head())

x = data.iloc[:,2:3].values
print("x values\n",x)
y = data.iloc[:,3:4]
print("y values\n",y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/4,random_state=0)
print("x_train\n",x_train)
print("x_test\n",x_test)
print("y_train\n",y_train)
print("y_test\n",y_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)

plt.scatter(x_train,y_train,c='red')
plt.xlabel('headsize')
plt.ylabel('brainweight')
plt.show()

from scipy.stats import spearmanr
corr,_ = spearmanr(x_train,y_train)
print('Spearmans correlation: %.3f'%corr)

plt.plot(x_test,y_pred)
plt.scatter(x_test,y_test,c='red')
plt.xlabel('headsize')
plt.ylabel('brainweight')

from numpy import cov
from scipy.stats import spearmanr

corr,_ = spearmanr(x_test,y_test)
print('Spearmans Correlation: %.3f' %corr)

print("Final rmse value is = ",np.sqrt(np.mean((y_test-y_pred)**2)))

#The RMSE value of our model is approximately 73% which is not bad.
#A good model should have an RMSE value less than 180. In case you have a higher RMSE value,
#this would mean that you probably need to change your feature
