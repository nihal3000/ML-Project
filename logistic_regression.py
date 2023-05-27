## **LOGISTIC REGRESSION USING SKLEARN PACKAGE**

# PROBLEM STATEMENT : DETERMINE WHETHER AN USER CAN BUY A PRODUCT BY SEEING
# SOCIAL ADVERTISEMENT

# Applicable: When the input or feature is continuos and output or target
#             variable should be categorical or binary classificstion

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/content/Social_Network_Ads.csv')

df['Purchased'].value_counts()

df.head(10)

df_getdummy = pd.get_dummies(data=df, columns=['Gender'])
print(df_getdummy.head())

X = df_getdummy.drop('Purchased',axis=1)
y = df_getdummy['Purchased']
print('data frame without purchased column\n')
print(X.head(10))
print('data frame with only target variable purchased')
print(y.head())



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
print("X_train\n",X_train.head())
print("X_test\n",X_test.head())
print("y_train\n",y_train.head())
print("y_test\n",y_test.head())

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test,y_pred)

accuracy_score(y_train,y_pred=classifier.predict(X_train))

