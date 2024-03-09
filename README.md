# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]

    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta


data=pd.read_csv("C:/Users/SEC/Downloads/50_Startups.csv",header=None)
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_scaled=scaler.fit_transform(X1)
y1_scaled=scaler.fit_transform(y)
print(X)
print(X1_scaled)

theta=linear_regression(X1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction .reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

Developed by: Padmashree
RegisterNumber:  212222040110
*/
```

## Output:
![linear regression using gradient descent](sam.png)
![Screenshot 2024-03-04 144352](https://github.com/yuvarajmonarch/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/8dafef2a-bfee-4144-aa0a-1a553f93ab5e)
![Screenshot 2024-03-04 144405](https://github.com/yuvarajmonarch/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/edf19e3b-d4d0-42d5-be58-c2d578a73afd)
![Screenshot 2024-03-04 144423](https://github.com/yuvarajmonarch/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/9f6be3b9-3b62-4ad9-8d99-231a5270f28b)
![Screenshot 2024-03-04 144433](https://github.com/yuvarajmonarch/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/91066ad0-2735-480f-9b53-9e4a347aba79)
![Screenshot 2024-03-04 144445](https://github.com/yuvarajmonarch/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/d693da73-c874-412d-960f-bba1cf76f30e)
![Screenshot 2024-03-04 144501](https://github.com/yuvarajmonarch/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/90c2429b-add8-445b-94ce-f577e04ca9ef)
![Screenshot 2024-03-04 144514](https://github.com/yuvarajmonarch/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122221735/63e11248-f153-40c6-b1fa-79a0ad4646e7)





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
