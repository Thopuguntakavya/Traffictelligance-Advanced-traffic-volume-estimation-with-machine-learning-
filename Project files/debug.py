import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
data = pd.read_csv(r"C:\Users\rish2\OneDrive\Desktop\ml project\traffic volume.csv")

data.isnull().sum()

data['temp']=data['temp'].fillna(data['temp'].mean())
data['rain']=data['rain'].fillna(data['rain'].mean())
data['snow']=data['snow'].fillna(data['snow'].mean())
data['weather'] = data['weather'].fillna(1)

data[["day","month","year"]] = data["date"].str.split("-", expand=True)
data[["hours","minutes","seconds"]] = data["Time"].str.split(":", expand=True)
data.drop(columns=['date','Time'],axis=1,inplace=True)
# data[["day", "month", "year", "hours", "minutes", "seconds"]] = data[["day", "month", "year", "hours", "minutes", "seconds"]].astype(int)

#splitting into independant and dependant variables
y=data['traffic_volume']
x=data.drop(columns=['traffic_volume'],axis=1)

names=x.columns
x=scale(x)
x=pd.DataFrame(x,columns=names)
print(x.head())

#splitting the data into train data and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

lin_reg = linear_model.LinearRegression()
Dtree = tree.DecisionTreeRegressor()
Rand = ensemble.RandomForestRegressor()
svr = svm.SVR()
XGB = xgb.XGBRegressor()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

lin_reg.fit(x_train,y_train)
Dtree.fit(x_train,y_train)
Rand.fit(x_train,y_train)
svr.fit(x_train,y_train)
XGB.fit(x_train,y_train)

p1 = lin_reg.predict(x_train)
p2 = Dtree.predict(x_train)
p3 = Rand.predict(x_train)
p4 = svr.predict(x_train)
p5 = XGB.predict(x_train)

from sklearn import metrics
print(metrics.r2_score(p1,y_train))
print(metrics. r2_score(p2,y_train))
print(metrics.r2_score(p3,y_train))
print(metrics. r2_score(p4,y_train))
print(metrics. r2_score(p5, y_train))

p1 = lin_reg.predict(x_test)
p2 = Dtree.predict(x_test)
p3 = Rand.predict(x_test)
p4 = svr.predict(x_test)
p5 = XGB.predict(x_test)

print(metrics. r2_score(p1,y_test))
print(metrics. r2_score(p2,y_test))
print(metrics. r2_score(p3,y_test))
print(metrics.r2_score(p4,y_test))
print(metrics. r2_score(p5,y_test))

MSE = metrics.mean_squared_error(p3, y_test)
t1=np.sqrt(MSE)
print(t1)

import pickle
pickle.dump(Rand, open("model.pkl", 'wb'))
pickle.dump(x_scaled, open("scale.pkl", 'wb'))