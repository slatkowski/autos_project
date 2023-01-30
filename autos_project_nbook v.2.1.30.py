#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns
from pandas_profiling import ProfileReport
import time
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


get_ipython().system('git clone https://github.com/slatkowski/autos_project')
df = pd.read_csv('autos_project/autos.csv')


# In[ ]:


def DFCounter(df):
    for col in df.columns:
        print(df[col].value_counts())
        print('_______________________')
        
DFCounter(df)


# In[ ]:


print(df['dateCrawled'].min())
print(df['dateCrawled'].max())


# In[ ]:


df.set_index('index', inplace=True)


# In[ ]:


df.info()


# In[ ]:


df.describe().apply(lambda x: x.apply('{0:.2f}'.format))


# In[ ]:


from collections import Counter
import random
def NANFiller(df):
    def columnFiller(series):
        nan_c = len(series[series.isna()])
        nnan_c = series[series.notna()]
        count_nn = Counter(nnan_c)    
        new_val = random.choices(list(count_nn.keys()), weights = list(count_nn.values()), k=nan_c)
        series[series.isna()] = new_val
        return series
    for col in df.columns:
        df[col]=columnFiller(df[col])
        
NANFiller(df)


# In[ ]:


df.info()


# In[ ]:


df = df.loc[(df.brand == 'volkswagen') | (df.brand == 'bmw') | (df.brand == 'mercedes_benz') | (df.brand == 'opel') | (df.brand == 'audi')]


# In[ ]:


df = df[(df['yearOfRegistration'] >= 1976) & (df['yearOfRegistration'] <= 2016)]
df = df[(df['price'] >= 250) & (df['price'] <= 60000)]
df = df[(df['powerPS'] >= 25) & (df['powerPS'] <= 600)]


# In[ ]:


df.info()


# In[ ]:


bins = [0, 16999, 28999, 39999, 69999, 89999]

df['postalCode'] = pd.cut(df['postalCode'], bins=bins,
       labels=['Eastern', 'Northern', 'Central', 'Western', 'Southern'])

df['postalCode']


# In[ ]:


df['postalCode'].fillna('Central', inplace=True)
df['postalCode']


# In[ ]:


DFCounter(df)


# In[ ]:


df.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


# In[ ]:


encode_list = ['gearbox', 'vehicleType', 'notRepairedDamage', 'fuelType', 'postalCode']


# In[ ]:


for i in encode_list:
    df[i] = le.fit_transform(df[i])
    print(df[i].name)
    print(le.classes_)
    print(np.unique(df[i]))
    print('__________')


# In[ ]:


df.info()


# In[ ]:


df['brand'].value_counts()


# In[ ]:


min_cnt = df['brand'].value_counts().min()
df = df.groupby('brand').sample(min_cnt)


# In[ ]:


df['brand'].value_counts()


# In[ ]:


df.describe().apply(lambda x: x.apply('{0:.2f}'.format))


# In[ ]:


df.columns


# In[ ]:


df.drop(columns=['dateCrawled', 'name', 'seller', 'offerType', 'abtest', 'model', 'monthOfRegistration',
       'dateCreated', 'nrOfPictures', 'lastSeen'], inplace=True)


# In[ ]:


df.info()


# In[ ]:


report = ProfileReport(df, infer_dtypes=False)
report


# In[ ]:


pivot = pd.pivot_table(df, index='brand', values = ['price', 'vehicleType', 'yearOfRegistration',
                                                      'gearbox', 'powerPS', 'fuelType',
                                                      'notRepairedDamage', 'kilometer', 'postalCode'], 
                       aggfunc= [np.mean, np.median, np.std, min, max])
pd.options.display.max_columns = None
display(pivot)


# In[ ]:


import plotly.express as px

for i, col in enumerate(df.columns):
    plt.figure(i)
    fig = px.ecdf(df, x=col, color="brand")
    fig.show()


# In[ ]:


X = df.drop(columns='brand')
y = df['brand']


# In[ ]:


X.sample(5)


# In[ ]:


y.sample(5)


# In[ ]:


y = le.fit_transform(y)
y


# In[ ]:


from sklearn.feature_selection import mutual_info_classif
 
importances = mutual_info_classif(X, y)
 
feature_info = pd.Series(importances, X.columns).sort_values(ascending=False)
print(f'Three features providing the biggest information gain are:\n{feature_info.head(5)}')
print(f'Their cumulative information gain equals {np.around(np.sum(feature_info.head(5)), 2)}.')


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

lr = LogisticRegression()

def SelectorChoiceDisplay(estimator, X, y):
    selector = SelectFromModel(estimator=estimator).fit(X, y)
    print(selector.threshold_)
    print(X.columns)
    print(selector.get_support())
    print(selector.transform(X))
    
SelectorChoiceDisplay(lr, X, y)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
SelectorChoiceDisplay(rfc, X, y)


# In[ ]:


from sklearn.svm import LinearSVC

lsvc = LinearSVC()
SelectorChoiceDisplay(lsvc, X, y)


# In[ ]:


X = df[['price', 'vehicleType', 'yearOfRegistration',
        'powerPS', 'postalCode', 'gearbox']]


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify=y)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                    y_train,
                                                    test_size = 0.25,
                                                    random_state = 42,
                                                    stratify=y_train)


# In[ ]:


print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_val shape: {y_val.shape}')
print(f'y_test shape: {y_test.shape}')

