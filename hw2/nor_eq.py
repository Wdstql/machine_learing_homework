import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

if __name__ == "__main__":
    data = pd.read_csv('hw2/11_train.csv')
    m=len(data)
    x = np.array(data.iloc[:, 1:4])                                                                                               
    y =np.array(data.iloc[:,4])
    data = np.array(data)
    x=np.insert(x,0,values=np.ones(m),axis=1)
    sum1=0
    sum2=0
    kf = KFold(10, shuffle=True)
    for train_data, test_data in kf.split(data):
        x_train=np.mat(x[train_data])
        y_train=np.mat(y[train_data]).T
        x_test=np.mat(x[test_data])
        y_test=np.mat(y[test_data]).T
        w=(x_train.T*x_train).I*x_train.T*y_train
        print('w=',w)
        a=x_test*w-y_test
        b=x_train*w-y_train
        J1=(a.T*a)/len(x_test) 
        J2=(b.T*b)/len(x_train)
        sum1=sum1+J1
        sum2=sum2+J2
    final_cost1=sum1/10.0
    final_cost2=sum2/10.0
    print('当sigma={0},训练误差为{1},测试误差为{2}'.format(0,final_cost2,final_cost1))