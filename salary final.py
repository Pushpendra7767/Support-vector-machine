# import packages
import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# print dataset
s_train = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Support vector machine\\SalaryData_Test(1).csv")
s_test = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Support vector machine\\SalaryData_Train(1).csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native","Salary"]
# plot boxplot
sns.boxplot(x="age",y="Salary",data=s_train,palette = "hls")
sns.boxplot(x="age",y="Salary",data=s_test,palette = "hls")
# function for fitting test & train data
number = preprocessing.LabelEncoder()
for i in string_columns:
    s_train[i] = number.fit_transform(s_train[i])
    s_test[i] = number.fit_transform(s_test[i])
# split dataset into trainx,y & testx,y
colnames = s_train.columns
len(colnames[0:13])
trainX = s_train[colnames[0:13]]
trainY = s_train[colnames[13]]
testX  = s_test[colnames[0:13]]
testY  = s_test[colnames[13]]
# apply poly kernel
model_poly = SVC(kernel = "poly")
model_poly.fit(trainX,trainY)
pred_test_poly = model_poly.predict(testX)
np.mean(pred_test_poly==testY) 
# apply rbf kernel
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(trainX,trainY)
pred_test_rbf = model_rbf.predict(testX)
np.mean(pred_test_rbf==testY) 































