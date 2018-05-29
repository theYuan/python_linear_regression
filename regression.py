import pandas as pd
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

housing = pd.read_csv("housing.csv",header = None)
#check if we got the data right, Note that we have 506 data
#print(housing.shape)
#print(housing.head())
# check how to index the first column
#print(housing.ix[:,[0]])
# to see first row
#print(housing.iloc[0])
features = [0,1,2,3,4,5,6,7,8,9,10,11,12]
x = housing[features]
y =housing[13]

#构造训练集和测试集
X_train,X_test, y_train, y_test = train_test_split(x, y, random_state=1)
# default split is 75% for training and 25% for testing
linreg = LinearRegression()
model=linreg.fit(X_train, y_train)
print(model)
print(linreg.intercept_)
print (linreg.coef_)

# 预测
y_pred = linreg.predict(X_test)
print(y_pred)
print(y_pred)

# 评价测度 使用均方根误差 RMSE

print(type(y_pred),type(y_test))
print (len(y_pred),len(y_test))
print (y_pred.shape,y_test.shape)
sum_mean=0
for i in range(len(y_pred)):
    sum_mean+=(y_pred[i]-y_test.values[i])**2

sum_erro=np.sqrt(sum_mean/127)
#calculate RMSE by hand
print("RMSE by hand:",sum_erro)

# 做 ROC 曲线
plt.figure()
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
plt.plot(range(len(y_pred)),y_test,'r',label="actual")
plt.legend(loc="upper right") #显示图中的标签
plt.ylabel('Median value of owner-occupied homes in $1000')
plt.show()