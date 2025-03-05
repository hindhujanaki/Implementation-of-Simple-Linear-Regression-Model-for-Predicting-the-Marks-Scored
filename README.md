# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

   

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: G.Hindhu
RegisterNumber: 212223230079 
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset = pd.read_csv('/content/student_scores.csv')
```
```
print(dataset.head())
print(dataset.tail())
```
![{0C026BFC-83E9-4D3D-90DC-193CB9948242}](https://github.com/user-attachments/assets/d60f5bcc-e718-438d-bcfa-c60df6b20bbe)
```
dataset.info()
```

![{3CF58B7D-6143-4D27-B906-F682E4955AD7}](https://github.com/user-attachments/assets/9d5212ac-1d45-4a5d-9652-09f3b3debe63)
```
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)
```

![{0B38E8F8-4212-4365-B968-78E111488171}](https://github.com/user-attachments/assets/dfc0ff33-102f-4a76-9eb0-b7fe00bc0d3f)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
x_train.shape
```

![{B1B278AF-73D7-42F5-B171-CD7991D50070}](https://github.com/user-attachments/assets/c814fc39-f038-4db3-ad1c-09e9b55c1bb0)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
x_test.shape
```

![{18B9EFFA-B636-40F1-8FE1-4F8967CE51B0}](https://github.com/user-attachments/assets/f67565a0-f346-4c13-8915-cc84e0b67f62)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
x_test.shape
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
```

![{191CC03D-DE22-4519-94BB-CDBC7C804627}](https://github.com/user-attachments/assets/dfaedb7d-09e8-414d-86be-850d37f0b2dd)


```
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
```

![{8D53CBDA-C8D8-46D6-B0C2-2720B744AB20}](https://github.com/user-attachments/assets/e0fa39b3-36d7-4085-abc3-120897db7011)

```
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title('Traning set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,reg.predict(x_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()

![Uploading {DE1F16A3-14C2-478E-91D9-F6FFC38799D6}.png…]()

```
```
mse = mean_squared_error(y_test, y_pred)
print('MSE = ', mse)
```
![{877ED146-EF5D-46F8-BAF9-DC6D39EEAB50}](https://github.com/user-attachments/assets/5a8d7b07-3e71-4908-b141-2452d576285c)
```
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
```
![{1E5BF70C-C6C2-408C-A66F-B12407029B40}](https://github.com/user-attachments/assets/3ad4eef7-86db-4a11-8a6f-ac06034c0286)
```
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

![{02AB4923-760A-4632-9101-34E8FDE13C07}](https://github.com/user-attachments/assets/53a4da3b-6e68-4b20-a220-bd53379d1c2b)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
