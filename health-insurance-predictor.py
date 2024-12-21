import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
from feature_engine.outliers import ArbitraryOutlierCapper
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

#importing the Health_insurance_cost dataset
health=pd.read_csv(r"C:\Users\maaza\Downloads\Documents\College\2nd Year\AI COE Internship\medical-charges.csv")

# print rows and columns
print (health.shape)
# print the columns/variables
print (health.columns)
# print the whole dataset
print(health)
#check if the dataset contains any empty datapoints
print(health.isnull().sum())

# removing the empty datapoints
# removing the premium missing records
health1=health.dropna(subset=['premium'])
print(health1.isnull().sum())

# print the new dataset after removing the empty datapoints
print(health1.info)

# creating category array for sex, region, smoking status
cat_col=[]
for i in health1.columns:
    if health1[i].dtypes==object:
        cat_col.append(i)
print(cat_col)

# couting the unique elements in the category columns for different elements
for i in cat_col:
    print(i)
    print(health[i].unique())
    print(health[i].value_counts())
    print('*'*30)

# calculate the total count, mean, standard deviation, minimum to max
print(health1.describe().T)

# creating pie chart for sex, smoking status, and region
# features = ['sex', 'smoker', 'region']
 
# plt.subplots(figsize=(20, 10))
# for i, col in enumerate(features):
#     plt.subplot(1, 3, i + 1)
 
#     x = health1[col].value_counts()
#     plt.pie(x.values,
#             labels=x.index,
#             autopct='%1.1f%%')
# plt.show()

# creating bar graphs
# features = ['sex', 'children', 'smoker', 'region']
 
# plt.subplots(figsize=(20, 10))
# for i, col in enumerate(features):
#     plt.subplot(2, 2, i + 1)
#     health1.groupby(col).mean()['premium'].plot.bar()
# plt.show()

# creating scatter plot
# features = ['age', 'bmi']
 
# plt.subplots(figsize=(17, 7))
# for i, col in enumerate(features):
#     plt.subplot(1, 2, i + 1)
#     sns.scatterplot(data=health1, x=col,
#                    y='premium',
#                    hue='smoker')
# plt.show()

# Data preprocessing

# creating boxplot for age
sns.boxplot(health1['age'])
plt.show()
# creating boxplot for bmi
sns.boxplot(health1['bmi'])
plt.show()

# cleaning the outlying data
Q1=health1['bmi'].quantile(0.25)
Q2=health1['bmi'].quantile(0.5)
Q3=health1['bmi'].quantile(0.75)
iqr=Q3-Q1
lowlim=Q1-1.5*iqr
upplim=Q3+1.5*iqr
print(lowlim)
print(upplim)


arb=ArbitraryOutlierCapper(min_capping_dict={'bmi':13.7},max_capping_dict={'bmi':47.29})
health1[['bmi']]=arb.fit_transform(health1[['bmi']])
sns.boxplot(health1['bmi'])
# plt.show()


print(health1['bmi'].skew())
print(health1['age'].skew())


# now encoding the data -> 
# sex: male = 0, female  =1 ; 
# smoker: yes=1, no=0 ;
# region: northwest = 0, northeast = 1, southeast = 2, southwest = 3 

health1['sex']=health1['sex'].map({'male':0,'female':1})
health1['smoker']=health1['smoker'].map({'yes':1,'no':0})
health1['region']=health1['region'].map({'northwest':0, 'northeast':1,'southeast':2,'southwest':3})
print(health1.info)

#correlation between the numerical input varaibles
print(health1.corr())


# model development and training
X=health1.drop(['premium'],axis=1)
Y=health1[['premium']]
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
l1=[]
l2=[]
l3=[]
cvs=0

# dividing in testing and validating data sets
for i in range(40,50):
    xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=i)
    lrmodel=LinearRegression()
    lrmodel.fit(xtrain,ytrain)
    l1.append(lrmodel.score(xtrain,ytrain))
    l2.append(lrmodel.score(xtest,ytest))
    cvs=(cross_val_score(lrmodel,X,Y,cv=5,)).mean()
    l3.append(cvs)
    df1=pd.DataFrame({'train acc':l1,'test acc':l2,'cvs':l3})
print(df1)

# Linear Regression model
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)
lrmodel=LinearRegression()
lrmodel.fit(xtrain,ytrain)
print(lrmodel.score(xtrain,ytrain))
print(lrmodel.score(xtest,ytest))
print(cross_val_score(lrmodel,X,Y,cv=5,).mean())



# SVR model
from sklearn.metrics import r2_score
svrmodel=SVR()
svrmodel.fit(xtrain,ytrain)
ypredtrain1=svrmodel.predict(xtrain)
ypredtest1=svrmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain1))
print(r2_score(ytest,ypredtest1))
print(cross_val_score(svrmodel,X,Y,cv=5,).mean())



# RandomForestRegressor model
rfmodel=RandomForestRegressor(random_state=42)
rfmodel.fit(xtrain,ytrain)
ypredtrain2=rfmodel.predict(xtrain)
ypredtest2=rfmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain2))
print(r2_score(ytest,ypredtest2))
print(cross_val_score(rfmodel,X,Y,cv=5,).mean())
from sklearn.model_selection import GridSearchCV
estimator=RandomForestRegressor(random_state=42)
param_grid={'n_estimators':[10,40,50,98,100,120,150]}
grid=GridSearchCV(estimator,param_grid,scoring="r2",cv=5)
grid.fit(xtrain,ytrain)
print(grid.best_params_)
rfmodel=RandomForestRegressor(random_state=42,n_estimators=120)
rfmodel.fit(xtrain,ytrain)
ypredtrain2=rfmodel.predict(xtrain)
ypredtest2=rfmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain2))
print(r2_score(ytest,ypredtest2))
print(cross_val_score(rfmodel,X,Y,cv=5,).mean())

# GradientBoostingRegressor model
gbmodel=GradientBoostingRegressor()
gbmodel.fit(xtrain,ytrain)
ypredtrain3=gbmodel.predict(xtrain)
ypredtest3=gbmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain3))
print(r2_score(ytest,ypredtest3))
print(cross_val_score(gbmodel,X,Y,cv=5,).mean())
from sklearn.model_selection import GridSearchCV
estimator=GradientBoostingRegressor()
param_grid={'n_estimators':[10,15,19,20,21,50],'learning_rate':[0.1,0.19,0.2,0.21,0.8,1]}
grid=GridSearchCV(estimator,param_grid,scoring="r2",cv=5)
grid.fit(xtrain,ytrain)
print(grid.best_params_)
gbmodel=GradientBoostingRegressor(n_estimators=19,learning_rate=0.2)
gbmodel.fit(xtrain,ytrain)
ypredtrain3=gbmodel.predict(xtrain)
ypredtest3=gbmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain3))
print(r2_score(ytest,ypredtest3))
print(cross_val_score(gbmodel,X,Y,cv=5,).mean())


#XGBRegressor model
xgmodel=XGBRegressor()
xgmodel.fit(xtrain,ytrain)
ypredtrain4=xgmodel.predict(xtrain)
ypredtest4=xgmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain4))
print(r2_score(ytest,ypredtest4))
print(cross_val_score(xgmodel,X,Y,cv=5,).mean())
from sklearn.model_selection import GridSearchCV
estimator=XGBRegressor()
param_grid={'n_estimators':[10,15,20,40,50],'max_depth':[3,4,5],'gamma':[0,0.15,0.3,0.5,1]}
grid=GridSearchCV(estimator,param_grid,scoring="r2",cv=5)
grid.fit(xtrain,ytrain)
print(grid.best_params_)
xgmodel=XGBRegressor(n_estimators=15,max_depth=3,gamma=0)
xgmodel.fit(xtrain,ytrain)
ypredtrain4=xgmodel.predict(xtrain)
ypredtest4=xgmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain4))
print(r2_score(ytest,ypredtest4))
print(cross_val_score(xgmodel,X,Y,cv=5,).mean())


# calculating importance of features
feats=pd.DataFrame(data=grid.best_estimator_.feature_importances_,index=X.columns,columns=['Importance'])
print(feats)

# filtering the most important features
important_features=feats[feats['Importance']>0.01]
important_features


# finally choosing the best model
print("Final Model:")
health1.drop(health1[['sex','region']],axis=1,inplace=True)
Xf=health1.drop(health1[['premium']],axis=1)
X=health1.drop(health1[['premium']],axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xf,Y,test_size=0.2,random_state=42)
finalmodel=XGBRegressor(n_estimators=15,max_depth=3,gamma=0)
finalmodel.fit(xtrain,ytrain)
ypredtrain4=finalmodel.predict(xtrain)
ypredtest4=finalmodel.predict(xtest)
print("Train Accuracy : ",r2_score(ytrain,ypredtrain4))
print("Test Accuracy : ",r2_score(ytest,ypredtest4))
print("Cross Value Score : ",cross_val_score(finalmodel,X,Y,cv=5,).mean())

# to save our model
from pickle import dump
dump(finalmodel,open('insurancemodelf.pkl','wb'))

# now predicting for the new data
new_data=pd.DataFrame({'age':19,'sex':'male','bmi':27.9,'children':0,'smoker':'no','region':'northeast'},index=[0])
new_data['smoker']=new_data['smoker'].map({'yes':1,'no':0})
new_data=new_data.drop(new_data[['sex','region']],axis=1)
print(finalmodel.predict(new_data))
