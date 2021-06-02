##############################problem 1 #########################
##############decison tree###############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#loading the dataset
df=pd.read_csv(r"C:/Users/usach/Desktop/decison tree and rf/Company_Data.csv")
df

 
# Check for missing values
df.isna().sum()

df['Sales']


#Importing LabelEncoder form Sklearn, to do label encoding for given columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['ShelveLoc']=le.fit_transform(df['ShelveLoc'])
df['Urban']=le.fit_transform(df['Urban'])
df['US']=le.fit_transform(df['US'])

predictors=df.drop(['Sales'],axis=1)
predictors


target=df.iloc[:,:1]
target

#Splitiing The Data to train the model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(predictors,target,train_size=0.8,random_state=0)

#Importing DecisionTreeRegressor from sklearn
#Decision tree builds regression or classification models in the form of a tree structure.
#It breaks down a dataset into smaller and smaller subsets while 
#at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes.
from sklearn.tree import DecisionTreeRegressor
clf=DecisionTreeRegressor()

clf.fit(x_train,y_train)
Out: DecisionTreeRegressor()
y_pred1=clf.predict(x_test)
y_pred1

#testing the accuracy
from sklearn.metrics import r2_score
r2_score(y_pred1,y_test)

###############random forest ###################
#importing RandomForestClassifier from sklearn
#Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression.
#A Random Forest operates by constructing several decision trees during training time.
#and outputting the mean of the classes as the prediction of all the trees.
from sklearn.ensemble import RandomForestRegressor
clf1=RandomForestRegressor()
clf1.fit(x_train,y_train)
 
#Predicting the model
y_pred2=clf1.predict(x_test)
y_pred2

#Accuracy
from sklearn.metrics import r2_score
r2_score(y_test,y_pred2)

#########################problem 2 ##########################
###################decision tree and random forest###############
#importing the data set
df=pd.read_csv(r"C:/Users/usach/Desktop/decison tree and rf/Diabetes.csv")
df

# Check for missing values
df.isna().sum()

#lable encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df[" Class variable"]=le.fit_transform(df[" Class variable"])

x=df.iloc[:,0:8]
x

y=df[" Class variable"]
y

#train test spliting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.80,random_state=0)
#model selection
from sklearn.tree import DecisionTreeRegressor
clf=DecisionTreeRegressor()

from sklearn.ensemble import RandomForestClassifier
clf_random=RandomForestClassifier()
 
#fitting the model
clf.fit(x_train,y_train)

clf_random.fit(x_train,y_train)

#predicting the model
y_pred=clf.predict(x_test)
y_pred

y_predt1=clf.predict(x_train)#for test data prediction
y_predt1

y_pred_random=clf_random.predict(x_test)
y_pred_random

#accuracy
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)

accuracy_score(y_pred_random,y_test)

accuracy_score(y_predt1,y_train)#for train accuracy

from sklearn import tree
clf=tree.DecisionTreeClassifier(random_state=0)

clf=clf.fit(x,y)

plt.figure(figsize=(200,180))
tree.plot_tree(clf,filled=True)

############################problem 3 #######################
################decision tree and random forest###############
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#importing the dataset

df=pd.read_csv(r"C:/Users/usach/Desktop/decison tree and rf/Fraud_check.csv")
df

sns.pairplot(df)

df["TaxInc"] = pd.cut(df["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
FraudCheck =df.drop(columns=["Taxable.Income"])


# Check for missing values
FraudCheck.isna().sum()

#doing label encoding for catagorical features
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
FraudCheck['Undergrad']=le.fit_transform(df['Undergrad'])
FraudCheck['Urban']=le.fit_transform(df['Urban'])
FraudCheck['Marital.Status']=le.fit_transform(df['Marital.Status'])
FraudCheck['TaxInc']=le.fit_transform(df['TaxInc'])


predictors=FraudCheck.iloc[:,FraudCheck.columns!='TaxInc']
predictors

target=FraudCheck['TaxInc']
target

#Train Test Split
# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.80, random_state=0)
#Model Selection
# Train the Regression DT
from sklearn.tree import DecisionTreeRegressor
clf_dt=DecisionTreeRegressor()
clf_dt.fit(x_train,y_train)
# train the regressor RF
from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier()
clf_rf.fit(x_train,y_train)

y_pred_dt=clf_dt.predict(x_test)
y_pred_dt

y_pred_rf=clf_rf.predict(x_test)
y_pred_rf

#Measuring accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_pred_dt,y_test)

accuracy_score(y_pred_rf,y_test)

###########################problem 4##########################
#####################decsion tree and random forest################
#importing the libray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#load the data set
df=pd.read_csv(r"C:/Users/usach/Desktop/decison tree and rf/HR_DT.csv")
df

# Check for missing values
df.isna().sum()
sns.pairplot(df)
predictors=df.iloc[:,-2:-1]
predictors
target=df.iloc[:,-1:]
target
#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(predictors,target,test_size=0.80,random_state=0)
#model selection
from sklearn.tree import DecisionTreeRegressor
clf_dt=DecisionTreeRegressor()

from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier()

#fitting the model
clf_dt.fit(x_train,y_train)
clf_rf.fit(x_train,y_train)

#predicting the model

y_pred_dt=clf_dt.predict(x_test)
y_pred_dt
y_pred_rf=clf_rf.predict(x_test)
y_pred_rf
from sklearn.metrics import r2_score
r2_score(y_pred_dt,y_test)

r2_score(y_pred_rf,y_test)

