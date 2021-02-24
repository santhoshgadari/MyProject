#!/usr/bin/env python
# coding: utf-8

# # Customer Purchage Pridiction Analysis

# ### Here we collected data from the showroom related to sales of KTM bike, the obective is that we have pridict whether the customer will purchage the bike by analysing previous data like age, profission, salary..etc

# ### Importing required librares

# In[47]:


# Importing data handling librares
import pandas as pd
import numpy as np
from collections import OrderedDict

## Importing data visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns

## Importing statistical libraries
import scipy.stats as scipy_stats


## Model building algorithms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# #### Importing dataset

# In[2]:


df=pd.read_excel("data.xlsx")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[ ]:





# In[ ]:





# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# ## Data Preprocessing 

# 
# ### a) Data Encoding

# One can find many ways to handle  data. Some of them categorical data are,
# 1. <span style="color: blue;">**Nominal data**</span> -->If data is not in any order --> <span style="color: green;">**OneHotEncoder**</span> is used in this case
# 2. <span style="color: blue;">**Ordinal data**</span> -->If data is in order --> <span style="color: green;">**LabelEncoder**</span> is used in this case
# 

# In[8]:


##Since data is in numrical format no need of Data Encoding here


# ### No need of Data Encoding here

# ### b) Outlier Treatment 

# In[9]:


# visualization before outlier treatment
fig = plt.figure(figsize = (15, 7))
sns.boxplot(data = df, orient = 'h')


# In[10]:


## Preparing a custom based EDA report
def custom_summary(df):
    result = []

    for col in list(df.columns):
        stats = OrderedDict({'FeatureName': col,'count':df[col].count(),'Datatype':df[col].dtype,
                                 'Mean':round(df[col].mean(),2),'SD':round(df[col].std(),2),'Variance':round(df[col].var(),2),
                                 'Min':round(df[col].min(),2),'Q1':round(df[col].quantile(0.25),2),'Median':round(df[col].median(),2),
                                 'Q3':round(df[col].quantile(0.75),2),'Max':round(df[col].max(),2),'Range':round(df[col].quantile(1),2)-round(df[col].quantile(0),2),
                                  'IQR':round(df[col].quantile(0.75),2)-round(df[col].quantile(0.25),2),
                                 'Kurtosis':round(df[col].kurt(),2),'Skewness':round(df[col].skew(),2)})
        result.append(stats)
        if df[col].skew() < -1:
            sk_label = 'Highly Negatively Skewed'
        elif -1 <= df[col].skew() < -0.5:
            sk_label = 'Moderately Negatively skewed'
        elif -0.5 <= df[col].skew() < 0:
            sk_label = 'Fairly Symmetric(Negative)'  
        elif 0 <= df[col].skew() < 0.5:
            sk_label = 'Fairly Symmetric(Positive)'
        elif 0.5 <= df[col].skew() < 1:
            sk_label = 'Moderately Positively skewed'
        elif df[col].skew() > 1:
            sk_label = 'Highly Positively Skewed'
        else:
            sk_label = 'Error'        
        stats['Skewness Comment'] = sk_label
        Upper_limit = stats['Q3'] + (1.5 *stats['IQR'])
        Lower_limit = stats['Q1'] - (1.5 *stats['IQR'])
        if len([x for x in df[col] if x < Lower_limit or x > Upper_limit ]) > 1:
            Out_Label = 'Has Outlier'
        else:
            Out_Label = 'No Outlier'
        stats['Outlier comment'] = Out_Label
    resultdf = pd.DataFrame(data = result)
    return resultdf 


# In[11]:


custom_summary(df)


# In[48]:


def OutlierDetectionPlots(df, col):
    f,(ax1, ax2, ax3) = plt.subplots(1,3, figsize = (20,10))
    #col = 'cement'
    ax1.set_title(col + "Boxplot")
    ax1.set_xlabel('Box Density')
    ax1.set_ylabel(col + 'value')
    sns.boxplot(df[col], ax = ax1, orient= 'v',color='red' )

    ## Plottng Histogram with Outliers
    sns.distplot(df[col], ax = ax2, color= 'red',fit= scipy_stats.norm)
    ax2.axvline(df[col].mean(), color = 'gray',linestyle = 'dashed')
    ax2.axvline(df[col].median(), color = 'black',linestyle = 'dashed')
    ax2.set_title(col + 'Histogram with Outliers ')
    ax2.set_xlabel('Density')
    ax2.set_ylabel(col + 'values')

    ## Plottng Histogram without Outliers
    upper_bound, lower_bound = np.percentile(df[col], [5,95])
    y = pd.DataFrame(np.clip(df[col], upper_bound, lower_bound)) # clip filters values less than 5% and more than 95
    sns.distplot(y[col], ax = ax3, color = 'green',fit= scipy_stats.norm)
    ax3.axvline(df[col].mean(), color = 'gray',linestyle = 'dashed')
    ax3.axvline(df[col].median(), color = 'black',linestyle = 'dashed')
    ax3.set_title(col + 'Histogram without Outliers ')
    ax3.set_xlabel('Density')
    ax3.set_ylabel(col + 'values')
    
    plt.show()


# In[49]:


for col in list(df.columns):
    OutlierDetectionPlots(df,col)


# ### We can see now only age is having some outlers apart from that all columns are normally distributed

# In[12]:


df.columns


# In[13]:


for col in df.columns:
    sns.boxplot(df[col])
      


# ###  c). Imputation for missing value treatment

# In[14]:


df.isnull().sum()


# # Data Selection for Model Building

# ## a).Splitting the data into train and test

# In[15]:


x = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[16]:


x.head()


# In[17]:


y.head()


# In[18]:


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


# In[19]:


print("no.of training sampels = ",len(x_train))
print("no.of testing sampels = ",len(x_test))


# #  Model Building

# ### a). Logistic Regression

# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[21]:


lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)
lr_pred = lr_model.predict(x_test)


# In[22]:


accuracy_score(y_test, lr_pred)


# In[23]:


print(classification_report(y_test, lr_pred))


# ### b). Decision Tree

# In[24]:


dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)


# In[25]:


dt_pred = dt_model.predict(x_test)


# In[26]:


accuracy_score(y_test, dt_pred)


# ### c).  Naive Bayes

# In[27]:


# Naive bayes
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
nb_pred = nb_model.predict(x_test)
accuracy_score(y_test, nb_pred)


# ###  d).KNN

# In[28]:


knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
knn_pred = knn_model.predict(x_test)
accuracy_score(y_test, knn_pred)


# ### e).SVM

# In[29]:


## SVM
from sklearn.svm import SVC
SVM = SVC()
SVM.fit(x_train, y_train)
SVM_pred = SVM.predict(x_test)
accuracy_score(y_test,SVM_pred)


# ### f).Random Forest

# In[30]:


## Random Forest
# random forest
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)
accuracy_score(y_test, rf_pred)


# ### Assigning parameters to all the algorithms

# In[31]:


Vlr_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)
nb_model = GaussianNB(priors=None, var_smoothing=1e-09)
rf_model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
knn_model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=5, p=2,
           weights='uniform')
dt_model = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
svm_model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)


# ### Hyper Parameter Tuning

# In[32]:


parms_rf = {'criterion':['gini', 'entropy'], 'n_estimators':[10, 100, 200], 'min_sample_splits':[8, 10, 12], 'max_features':['sqrt', 'log2']}
parms_lr = {'random_state': [0], 'solver': ['liblinear', 'ibfgs']}
parms_knn = {'n_neighbors':[5, 10], 'weights':['uniform', 'distance'], 'algorithm':['ball_tree', 'kd_tree'], 'leaf_size':[30, 40]}
parms_dt = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features':['log2', 'sqrt'], 'class_weight':['balanced']}
parms_nb = {'priors':['None'], 'var_smoothing':['float']}
parms_svm = {'kernel':['rbf', 'linear', 'poly'], 'gamma':['auto', 'scale'], 'shrinking':['bool'], 'class_weight':['balanced'], 'decision_function_shape':['ovo']}


# In[33]:


model = GridSearchCV(rf_model, parms_rf, cv=10)
model = GridSearchCV(lr_model, parms_lr, cv=10)
model = GridSearchCV(knn_model, parms_knn, cv=10)
model = GridSearchCV(dt_model, parms_dt, cv=10)
model = GridSearchCV(nb_model, parms_nb, cv=10)
model = GridSearchCV(svm_model, parms_svm, cv=10)
model.estimator


# In[34]:


def stratifiedkfold(x, y, model):
    from sklearn.model_selection import StratifiedKFold
    accuracy_list = []
    skf = StratifiedKFold(n_splits= 10, random_state= 100)
    skf.get_n_splits(x,y)
    for train_index, test_index in skf.split(x,y):
        x1_train, x1_test = x.iloc[train_index], x.iloc[test_index]
        y1_train, y1_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(x1_train, y1_train)
        y_pred = model.predict(x1_test)
        accuracy_list.append(accuracy_score(y1_test, y_pred))
    return accuracy_list 


# In[35]:


lr = LogisticRegression()
dt = DecisionTreeClassifier()
nb = GaussianNB()
rf = RandomForestClassifier()
knn = KNeighborsClassifier()
svm = SVC()


# In[36]:


print("mean accuracy for Logistic Regression", np.mean(stratifiedkfold(x,y,lr)))
print("mean accuracy for Decision Tree", np.mean(stratifiedkfold(x,y,dt)))
print("mean accuracy for Navie Bayes", np.mean(stratifiedkfold(x,y,nb)))
print("mean accuracy for Random Forest", np.mean(stratifiedkfold(x,y,rf)))
print("mean accuracy for svm", np.mean(stratifiedkfold(x,y,svm)))
print("mean accuracy for KNN", np.mean(stratifiedkfold(x,y,knn)))


# In[37]:


def stratifiedkfold(x, y, model):
    from sklearn.model_selection import StratifiedKFold
    accuracy_list = []
    skf = StratifiedKFold(n_splits = 10, random_state = 100)
    skf.get_n_splits(x, y)
    for train_index, test_index in skf.split(x, y):
        x1_train, x1_test = x.iloc[train_index], x.iloc[test_index]
        y1_train, y1_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(x1_train, y1_train)
        model_pred = model.predict(x1_test)
        accuracy_list.append(accuracy_score(y1_test, model_pred))
        return accuracy_list


# In[38]:


skf_lr = np.mean(stratifiedkfold(x, y, lr_model))
skf_rf = np.mean(stratifiedkfold(x, y, rf_model))
skf_nb = np.mean(stratifiedkfold(x, y, nb_model))
skf_knn = np.mean(stratifiedkfold(x, y, knn_model))
skf_dt = np.mean(stratifiedkfold(x, y, dt_model))
skf_svm = np.mean(stratifiedkfold(x, y, svm))


# In[39]:


skf_lr = np.max(stratifiedkfold(x, y, lr_model))
skf_rf = np.mean(stratifiedkfold(x, y, rf_model))
skf_nb = np.mean(stratifiedkfold(x, y, nb_model))
skf_knn = np.mean(stratifiedkfold(x, y, knn_model))
skf_dt = np.mean(stratifiedkfold(x, y, dt_model))
skf_svm = np.mean(stratifiedkfold(x, y, svm))


# In[40]:


print("Mean accuracy of Logistic Regression:", skf_lr)
print("Mean accuracy of Random Forest:", skf_rf)
print("Mean accuracy of Naive bayes:", skf_nb)
print("Mean accuracy of KNN:", skf_knn)
print("Mean accuracy of Decision Tree:", skf_dt)
print("Mean accuracy of SVM:", skf_svm)


# ### Before cross valdation Logistc regression and Naive Bayes giving best accuracy 

# ### After cross valdation Decision tree is giving best Accuracy
