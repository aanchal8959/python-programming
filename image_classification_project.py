import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline
#using pandas to read database stored in same folder
data=pd.read_csv('mnist_train.csv')
#viewing column head
data.head()
#extracting data from dataset and viewing them up close
a=data.iloc[3,1:].values
#reshaping the extracted data into resonable size
a=a.reshape(28,28).astype("uint8")
plt.imshow(a)
#preparing data
#separating label and data values
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)
#check data
x_train.head()
#calling random forest classifier
rf=RandomForestClassifier(n_estimators=100)
#fit the model
rf.fit(x_train,y_train)
#prediction on test data
pred=rf.predict(x_test)
pred
#check prediction accuracy
s=y_test.values

#calculating no. of correctly predicted value
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1
count
#total values that prediction code was run on
len(pred)
#accuracy percentage
(count/len(pred))*100
