import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
def linear_reg(x, y, l=0.00002, times=50000,sigma=0.3):
    w = np.mat(np.zeros((4, 1)))
    m = len(x)
    for i in range(times):
        a=l*(1/m)*x.T*(x*w - y)
        w = w - a
        #print(a)
    return w
data = pd.read_csv('hw2/11_train.csv')
m=len(data)
x = np.array(data.iloc[:, 1:4])
x=np.insert(x,0,values=np.ones(m),axis=1)
y =np.array(data.iloc[:,4])
kf = KFold(10, shuffle=True, random_state=2)
data = np.array(data)
sum1,sum2=0,0
for train_data, test_data in kf.split(data):
    x_train=np.mat(x[train_data])
    y_train=np.mat(y[train_data]).T
    x_test=np.mat(x[test_data])
    y_test=np.mat(y[test_data]).T
    #print(x_train)
    #print(y_train)
    w = linear_reg(x_train, y_train)
    #print(w)
    a=x_test*w-y_test
    b=x_train*w-y_train
    J1=a.T*a/len(x_test)
    J2=b.T*b/len(x_train)
    print(J1,J2)
    sum1=sum1+J1
    sum2=sum2+J2
final_cost1=sum1/10
final_cost2=sum2/10
print(final_cost1)
print(final_cost2)