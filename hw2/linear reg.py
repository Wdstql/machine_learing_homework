import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
def linear_reg(x, y, l=0.001, times=10000,sigma=0.3):
    w = np.mat(np.zeros((4, 1)))
    m = len(x)
    for i in range(times):
        a=l*(1/m)*x.T*(x*w - y)
        w = w*(1-l*sigma/m) - a
        #print(a)
    return w

def feature_scaling(x):
    x_norm=np.mat(x)
    mu=np.mean(x_norm,axis=0)
    std=np.std(x_norm,axis=0,ddof=1)#无偏
    print(mu,std)
    x_norm=(x_norm-mu)/std
    return x_norm.A

data = pd.read_csv('hw2/11_train.csv')
m=len(data)
x = np.array(data.iloc[:, 1:4])                                                                                               
y =np.array(data.iloc[:,4])
data = np.array(data)
#feature scaling
x=feature_scaling(x)
x=np.insert(x,0,values=np.ones(m),axis=1)

kf = KFold(10, shuffle=True)
sigmas=[i for i in np.arange(0,0.01,0.001)]#正则项参数
w=np.mat([0,0,0,0])
for i in sigmas:
    sum1,sum2=0,0
    for train_data, test_data in kf.split(data):
        x_train=np.mat(x[train_data])
        y_train=np.mat(y[train_data]).T
        x_test=np.mat(x[test_data])
        y_test=np.mat(y[test_data]).T
        #print(x_train)
        #print(y_train)
        w = linear_reg(x_train, y_train,sigma=i)
        #print(w)
        a=x_test*w-y_test
        b=x_train*w-y_train
        J1=(a.T*a+i*(w.T*w))/len(x_test)
        J2=(b.T*b+i*(w.T*w))/len(x_train)
        #print(J1,J2)
        sum1=sum1+J1
        sum2=sum2+J2
    final_cost1=sum1/10.0
    final_cost2=sum2/10.0
    print('当sigma={0},训练误差为{1},测试误差为{2}'.format(i,final_cost2,final_cost1))
