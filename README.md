# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DEEPIKA R
RegisterNumber: 212223230038
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

# Output:
## Dataset:
![image](https://github.com/user-attachments/assets/2d7524c4-4e2b-49e4-815d-a169ed8b67c6)
## Head values:
![image](https://github.com/user-attachments/assets/9f6f3695-3ec3-4ba3-87d2-79e316e735f8)
## Tailvalue:
![image](https://github.com/user-attachments/assets/d0ddcb1f-bfc7-4f78-8683-c0f4c1e169bf)
## Values of X and Y:
![image](https://github.com/user-attachments/assets/7541a98a-e794-4729-8be7-9a66eb619353)
## Predication values of X and Y:
![image](https://github.com/user-attachments/assets/114d9063-6ab0-4bb5-b300-98c57b459cb6)
## MSE,MAE and RMSE:
![image](https://github.com/user-attachments/assets/96e6e751-8910-4406-bcad-2c98fc06f1f7)
## Training Set:
![image](https://github.com/user-attachments/assets/bf78c2d5-004d-421b-aada-64cc21265717)
## Testing Set:
![image](https://github.com/user-attachments/assets/c4fbbb15-040e-4736-ae28-ab1428d4e717)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
