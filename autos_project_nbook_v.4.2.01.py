#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''At first, we're going to import necessary libraries.'''

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns
from pandas_profiling import ProfileReport
import time
import math
import warnings
warnings.filterwarnings("ignore")


# In[2]:


'''The second action is a repository cloning from GitHub and dataset loading.'''

get_ipython().system('git clone https://github.com/slatkowski/autos_project')
path_to_file = 'autos_project/autos.csv'
df = pd.read_csv(path_to_file)
df


# In[3]:


'''Next we should count all values - function to use in this purpose is defined below.'''

def DFCounter(df):
    for col in df.columns:
        print(df[col].value_counts())
        print('_______________________')
        
DFCounter(df)


# In[4]:


'''"Andere" means "other" in German.
These values can hide any other value from adequate column,
so replacing them with NaN would be a correct action.'''

df.replace('andere', np.nan, inplace=True)


# In[5]:


'''Data used in this project comes from url https://data.world/data-society/used-cars-data.
It has been scraped from eBay Kleinanzeigen and refers to car selling advertisements.
We have to check when the first and the last advertisement
have been published to make data filtering correct.'''

print(f"Date of the first advertisement: {df['dateCrawled'].min()}.")
print(f"Date of the last advertisement: {df['dateCrawled'].max()}.")


# In[6]:


'''Column "index" contains unique values from 0 to 371528, so we can set it as index column.'''

df.set_index('index', inplace=True)


# In[7]:


'''First information about a DataFrame columns names, non-null values and data types.
Columns "vehicleType", "gearbox", "model", "fuelType" and "notRepairedDamage" contain NaN values.
Also some of the columns which should be the base of prediction are object (str) columns.'''

df.info()


# In[8]:


'''Next, take a look at the descriptive statistics of DataFrame
(values round to two places after a comma to better readability).
Unfortunately, there are many outliers - maximum value of column "price"
overpasses 2 bilions of euro, we have value (values) with price = 0,
in column "yearOfRegistration" we have cars "registered" in year 1000 and 9999.
Also column "kilometer" may show not enough variance - max value (150000)
appears as a median.'''

df.describe().apply(lambda x: x.apply('{0:.2f}'.format))


# In[9]:


'''To replace NaNs with values we should define a function
which replaces them with values according to probability of their appearance
in the column where NaNs appear.'''

from collections import Counter
import random
def NANFiller(df):
    #1. calling function columnFiller to modify column
    def columnFiller(series):
        #2. assigning number of NaN-s in column to a variable
        nan_c = len(series[series.isna()])
        #3. taking values from column with no NaN and assigning them to a temporary Series
        nnan_c = series[series.notna()]
        #4. counting not-NaN values from temporary Series
        count_nn = Counter(nnan_c)    
        #5. choosing random values according to probabilities of their apperance
        new_val = random.choices(list(count_nn.keys()), weights = list(count_nn.values()), k=nan_c)
        series[series.isna()] = new_val
        #6. returning column with new values
        return series
    #6. repeating operation above for the whole DataFrame
    for col in df.columns:
        df[col]=columnFiller(df[col])
        
NANFiller(df)


# In[10]:


'''The next description of DataFrame - as we can see, NaNs have been replaced.
We can assume that the columns used in prediction should have numeric values
(except for column "nrOfPicture" in which we have only one value)
and columns "vehicleType", "gearbox", "fuelType", "postalCode" and "notRepairedDamage"
with values transformed into discrete numbers.'''

df.info()


# In[11]:


'''Our model's task is to predict brand of cars using features above.
We're interested only in predicting five most popular brands: VW, BMW, Mercedes-Benz, Opel and Audi.'''

df = df.loc[(df.brand == 'volkswagen') | (df.brand == 'bmw') | (df.brand == 'mercedes_benz') | (df.brand == 'opel') | (df.brand == 'audi')]


# In[12]:


'''The next operation is getting rid of outliers. We're taking into consideration only cars not registered earlier 
than in 1980, with price in price bracket 500-40000 euros and engine power (in HP/PS) bracket 40 
(the power of engines mounted in the weakest versions of VW 1302/1303 Beetle and VW Polo) to 500 
(majority of brands, predominantly Mercedes-Benz, have more powerful models, but they are not in common use
and can be considered as outliers).'''

df = df[(df['yearOfRegistration'] >= 1980) & (df['yearOfRegistration'] <= 2016)]
df = df[(df['price'] >= 500) & (df['price'] <= 40000)]
df = df[(df['powerPS'] >= 40) & (df['powerPS'] <= 500)]


# In[13]:


df.info()


# In[14]:


'''Now let's transform postal codes into categories.
Basing on an administative division of Germany we can take 5 categories:
- codes 00000 to 16999 - Eastern Germany,
- 17000 to 28999 - Northern Germany,
- 29000 to 39999 and 90000 to 99999 - Central Germany,
- 40000 to 69999 - Western Germany, 
- 70000 to 89999 - Southern Germany.'''

bins = [0, 16999, 28999, 39999, 69999, 89999]

df['postalCode'] = pd.cut(df['postalCode'], bins=bins,
       labels=['Eastern', 'Northern', 'Central', 'Western', 'Southern'])

df['postalCode']


# In[15]:


'''Pd.cut can't transform two bins into one category.
The second one is transformed into NaN, which we can easily fill.'''

df['postalCode'].fillna('Central', inplace=True)
df['postalCode']


# In[16]:


DFCounter(df)


# In[17]:


df.info()


# In[18]:


'''Now we have to balance classes - at first, let's count them.'''

df['brand'].value_counts()


# In[19]:


'''Audi is the least numerous - 28725 adverts.
To make our model more sensible to each class and metrics intuitive,
we have to make quantity of adverts in brands equal.'''

min_cnt = df['brand'].value_counts().min()
df = df.groupby('brand').sample(min_cnt)

df['brand'].value_counts()


# In[20]:


'''Now we drop the columns with no importance in modelling.'''

df.drop(columns=['dateCrawled', 'name', 'seller', 'offerType', 'abtest', 'model', 'monthOfRegistration',
       'dateCreated', 'nrOfPictures', 'lastSeen'], inplace=True)


# In[21]:


df.columns


# In[22]:


'''To transform columns with strings into categorical,
we'll define function based on LabelEncoder.'''

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
encode_list = ['gearbox', 'vehicleType', 'notRepairedDamage', 'fuelType', 'postalCode']

def EncodingDesc(series, le):
    #1. transformation of pd.Series/pd.DataFrame.column
    transformed = le.fit_transform(series)
    #2. displaying the name of Series/column
    print(series.name)
    #3. displaying names of unique classes
    print(le.classes_)
    #4. displaying classes values after encoding
    print(np.unique(transformed))
    #5. returning transformed Series/column
    return transformed

for i in encode_list:
    df[i] = EncodingDesc(df[i], le)
    print('______________________')


# In[23]:


'''Last description of basic statistics should show us that the data are ready to processing.'''

df.describe().apply(lambda x: x.apply('{0:.2f}'.format))


# In[24]:


df.info()


# In[25]:


'''To visualize values, we'll take the convention:
- discrete values will be visualize by countplots,
- to visualize continuous variable we'll use KDE Plot.'''

'''Plots show that:

1. Mercedes-Benz is the only brand appearing in adverts with more models with automatic (0) than manual (1) gearbox. The remaining brands has advantage
in manual gearboxes; BMW and Audi have 2 times more models with them, VW and Opel 4-5 times more.

2. Variance in "vehicleType" column is interesting. Offers with Audi are rich when it comes to combis (4) and limousines (5), but the other types
are relatively rare. About a half of available BMWs are limousines, combis are two times less frequent, offer of cabrios (1) and coupes (2) is quite rich too,
the other types are marginal. Mercedes-Benz has great amount of limousines, too. Adverts with this brand have also visible amount of buses (0) and SUVs (6).
Opel is the the only brand offering the biggest amount of hatchbacks (3, in dataset named "kleinwagen" - literally "little car"). VW has relatively 
the most diverse palette of car types.

3. Each one of brand is characterised by more gasoline (0) than diesel (2) engines in column "fuelType". Advantage of gasoline engines is, except for Opel, not big.
Very rare are offers with LPG fuel types (5), CNG (1), electric (3) and hybrid (4) engines appear in individual cases.

4. Column "notRepairedDamage" doesn't provide any variance. Adverts in each class are more frequently concerning non-damaged (1) cars, damaged (0) 
appear 6-7 less often.

5. Values in columns "postalCode" are quite similar.
'''

sns.set(style="whitegrid") 

for i, col in enumerate(df[encode_list]):
    plt.figure(i)
    ax = sns.countplot(x=col, data=df, hue='brand')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


# In[26]:


'''When it comes to continuous variables:

1. Opel is much cheaper than other brands and gains about 100% proportion before 25000 euros. BMW, Audi and Mercedes-Benz
have similar increment of price and gain 100% at about 45000-50000 euros. VW is centrally in the middle between them.

2. Mercedes-Benz has the greatest amount of models older than 20 years available. The biggest amount of Opels and VWs
comes from last years of XX century, Mercedeses - from the first half of 00s' of XXI century, BMWs and Audis - the second half
of 00s.

3. VWs and Opels has much less powerful models and distrubutions of them are similar. BMW has models no less powerful than 70 HP,
Audi's and Mercedes-Benz's distributions have minimal values at the same point but Audi's reach its peak near BMW's.
VW's and Opel's cars power rarely overpass 250 HP, on the other hand there are quite a lot of BMW's, Audi's and Mercedes' 
vehicles with powers above 320 HP.

4. Just like we were afraid of, cars mileage (shown in "kilometer" column) doesn't give us enough variance, but because of 
connection with other variables at first we left them in set.
'''


for i, col in enumerate(df[['price', 'yearOfRegistration', 'powerPS', 'kilometer']]):
    plt.figure(i)
    ax = sns.kdeplot(x=col, data=df, hue='brand', fill=True)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


# In[27]:


'''Now let's generate the full report of data. As we can see, there are not much variables with strong correlation. 
Quite strong positive correlation takes place between price and registration year or engine power.
Negative correlation occurs between mileage and registration year or price and between gearbox type and engine power.
Car brands don't correlate strong with any of the other variables, but that's positive signal, because there's no "data leak".
The only continuous variable with distrubution quite similar to normal/Gaussian is the year of registration, 
the rest have strictly skew distributions.'''

report = ProfileReport(df, infer_dtypes=False)
report


# In[28]:


'''Using a pivot table we can take a look at descriptive stats of each brand.
Basing on mean, standard deviation and our previous observations we can see 
that the variables providing the greatest variance are: fuel type, engine power, price and vehicle type.'''

pivot = pd.pivot_table(df, index='brand', values = ['price', 'vehicleType', 'yearOfRegistration',
                                                      'gearbox', 'powerPS', 'fuelType',
                                                      'notRepairedDamage', 'kilometer', 'postalCode'], 
                       aggfunc= [np.mean, np.median, np.std, min, max])
pd.options.display.max_columns = None
display(pivot)


# In[29]:


'''Now we can divide dataset to features (X) and labels (y).'''

X = df.drop(columns='brand')
y = df['brand']


# In[30]:


'''As we can see below, division of the set ended successfully.'''
print(X.sample(5))
print(y.sample(5))


# In[31]:


'''Now we should encode brand names into categorical numbers.'''

y = le.fit_transform(y)
print(y)
print(type(y))


# In[32]:


#Operations made before this cell shouldn't be modified later.


# In[33]:


from sklearn.feature_selection import mutual_info_classif
 
importances = mutual_info_classif(X, y)
 
feature_info = pd.Series(importances, X.columns).sort_values(ascending=False)
print(f'Five features providing the biggest information gain are:\n{feature_info.head(5)}')
print(f'Their cumulative information gain equals {np.around(np.sum(feature_info.head(5)), 2)}.')


# In[34]:


from sklearn.feature_selection import SelectFromModel

def SelectorChoiceDisplay(estimator, X, y):
    selector = SelectFromModel(estimator=estimator).fit(X, y)
    print(f'Threshold value used for feature selection: {selector.threshold_}.\n')
    cols = pd.Series(X.columns, name='Is feature supported?')
    support = pd.Series(selector.get_support(), name='Answer')
    concatted = pd.concat([cols, support], join='inner', axis=1)
    print(concatted)


# In[35]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_features=None)
SelectorChoiceDisplay(rfc, X, y)


# In[36]:


X_selected = df[['price', 'vehicleType', 'yearOfRegistration',
        'powerPS', 'gearbox']]


# In[37]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify=y)


# In[38]:


X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                    y_train,
                                                    test_size = 0.25,
                                                    random_state = 42,
                                                    stratify=y_train)


# In[39]:


print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_val shape: {y_val.shape}')
print(f'y_test shape: {y_test.shape}')


# In[40]:


first_model = RandomForestClassifier()

start = time.time()
first_model.fit(X_train, y_train)
first_model_train_pred = first_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of RandomForestClassifier with default params: {mins} mins and {seconds} s.")


# In[41]:


from sklearn.metrics import classification_report

first_model_val_pred = first_model.predict(X_val)
print(f'1st model - training metrics:\n\n{classification_report(y_train, first_model_train_pred)}')
print('_____________________________________________________\n')
print(f'1st model - validation metrics:\n\n{classification_report(y_val, first_model_val_pred)}')


# In[44]:


from sklearn.model_selection import GridSearchCV

rfc = RandomForestClassifier()
params = {
    'max_depth': [5, 10, 15],
    'n_estimators': [150, 200, 300]
}

second_model = GridSearchCV(estimator = rfc, param_grid = params, cv = 10)
start = time.time()
second_model.fit(X_train, y_train)
pred = second_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of RandomForestClassifier with params chosen by GridSearchCV: {mins} mins and {seconds} s.")


# In[47]:


print(second_model.best_estimator_)
print(second_model.best_score_)


# In[46]:


second_model_val_pred = second_model.predict(X_val)
print(f'2nd model - training metrics:\n\n{classification_report(y_train, pred)}')
print('_____________________________________________________\n')
print(f'2nd model - validation metrics:\n\n{classification_report(y_val, second_model_val_pred)}')


# In[48]:


third_model = RandomForestClassifier(
    max_depth=20,
    n_estimators=500
)

start = time.time()
third_model.fit(X_train, y_train)
third_model_train_pred = third_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of RandomForestClassifier with maximal depth 20 and 500 estimators: {mins} mins and {seconds} s.")


# In[49]:


third_model_val_pred = third_model.predict(X_val)
print(f'3rd model - training metrics:\n\n{classification_report(y_train, third_model_train_pred)}')
print('_____________________________________________________\n')
print(f'3rd model - validation metrics:\n\n{classification_report(y_val, third_model_val_pred)}')


# In[50]:


fourth_model = RandomForestClassifier(
    max_depth=30,
    n_estimators=1000
)

start = time.time()
fourth_model.fit(X_train, y_train)
fourth_model_train_pred = fourth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of RandomForestClassifier with maximal depth 30 and 1000 estimators: {mins} mins and {seconds} s.")


# In[52]:


fourth_model_val_pred = fourth_model.predict(X_val)
print(f'4th model - training metrics:\n\n{classification_report(y_train, fourth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'4th model - validation metrics:\n\n{classification_report(y_val, fourth_model_val_pred)}')


# In[53]:


fifth_model = RandomForestClassifier(
    max_depth=25,
    n_estimators=750
)

start = time.time()
fifth_model.fit(X_train, y_train)
fifth_model_train_pred = fifth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of RandomForestClassifier with maximal depth 25 and 750 estimators: {mins} mins and {seconds} s.")


# In[55]:


fifth_model_val_pred = fifth_model.predict(X_val)
print(f'5th model - training metrics:\n\n{classification_report(y_train, fifth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'5th model - validation metrics:\n\n{classification_report(y_val, fifth_model_val_pred)}')


# In[ ]:




