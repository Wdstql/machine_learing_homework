from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

def best_C_param (x,y):
    kf=KFold(n_splits=10,shuffle=True,random_state=0)
    C_param=[0.001,0.01,0.1,1,10,100]
    result=[]
    for c_parm in C_param:
        print('C_param',c_parm)
        recall_score_lr_kf=[]
        for k,(train,test) in enumerate(kf.split(x,y)):
            #print(y.iloc[train])
            model = LogisticRegression(C=c_parm,penalty='l2')
            lr_kf=model.fit(x.iloc[train],y.iloc[train].values.ravel())
            pred_lr_kf=lr_kf.predict(x.iloc[test])
            recall_score_lrkf=recall_score(y.iloc[test],pred_lr_kf)
            recall_score_lr_kf.append(recall_score_lrkf)
            print('iteration',k,'recall score',recall_score_lrkf)
        result.append(np.mean(recall_score_lr_kf))
        print(c_parm,np.mean(recall_score_lr_kf))
        print('bets mean recall score',max(result))

data=pd.read_csv('hw1/11_train.csv',header=None)
m=len(data)
print(m)
print(data)
X=data.iloc[:,0:2]
print(X)
X.insert(0,'',np.ones(m))
Y=data.iloc[:,2]
print('using whole datasets X and Y')
print(X,Y)
best_C_param(X,Y)
