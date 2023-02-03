#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Data used in this project comes from url https://data.world/data-society/used-cars-data.
It has been scraped from eBay Kleinanzeigen and refers to car selling advertisements published in years 2014-2016.
The main goal of this project is assigning cars produced by five producers which vehicles appear the most often.

At first, we're going to import necessary libraries.'''

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


'''We have to check when the first and the last advertisement
have been published to make data filtering correct.'''

df['dateCreated'] = pd.to_datetime(df['dateCreated'])
df['lastSeen'] = pd.to_datetime(df['lastSeen'])

print(f"Date of the first advertisement: {df['dateCreated'].min()}.")
print(f"Date of the last advertisement: {df['dateCreated'].max()}.")


# In[6]:


'''Duration of advertisement appearance on the site (expressed in hours) can be also useful feature.
It may show how attractive cars from particular brands are.
To create this feature, as a DataFrame column, we need to subtract values from columns
"lastSeen" and "dateCreated", and divide them by instance of class pd.Timedelta.'''

delta = pd.Timedelta(hours=1)

df['advertDuration'] = (df['lastSeen'] - df['dateCreated'])/delta
df['advertDuration']


# In[7]:


'''Column "index" contains unique values from 0 to 371528, so we can set it as index column.'''

df.set_index('index', inplace=True)


# In[8]:


'''First information about a DataFrame columns names, non-null values and data types.
Columns "vehicleType", "gearbox", "model", "fuelType" and "notRepairedDamage" contain NaN values.
Also some of the columns which should be the base of prediction, like "vehicleType", "fuelType" or
"gearbox" are object (str) columns.'''

df.info()


# In[9]:


'''Next, take a look at the descriptive statistics of DataFrame
(values round to two places after a comma to better readability).
Unfortunately, there are many outliers - maximum value of column "price"
overpasses 2 bilions of euro, we have value (values) with price = 0,
in column "yearOfRegistration" we have cars "registered" in year 1000 and 9999.
Also column "kilometer" may show not enough variance - max value (150000)
appears as a median.'''

df.describe().apply(lambda x: x.apply('{0:.2f}'.format))


# In[10]:


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


# In[11]:


'''The next description of DataFrame - as we can see, NaNs have been replaced.
We can assume that the columns used in prediction should have numeric values
(except for column "nrOfPicture" in which we have only one value)
and columns "vehicleType", "gearbox", "fuelType" and "postalCode"
with values transformed into discrete numbers.'''

df.info()


# In[12]:


'''Just like it was said at the beginning, we're interested only in predicting
five most popular brands: VW, BMW, Mercedes-Benz, Opel and Audi.'''

df = df.loc[(df.brand == 'volkswagen') | (df.brand == 'bmw') | (df.brand == 'mercedes_benz') | (df.brand == 'opel') | (df.brand == 'audi')]


# In[13]:


'''The next operation is getting rid of outliers. We're taking into consideration only cars not registered earlier 
than in 1980, with price in price bracket 250-40000 euros and engine power (in HP/PS) bracket 40 
(the power of engines mounted in the weakest versions of popular models VW 1302/1303 Beetle and VW Polo) to 500 
(majority of brands, predominantly Mercedes-Benz, have more powerful models, but they are not in common use
and can be considered as outliers). Also we aren't going to consider adverts present on the website
longer that 840 hours (35 days, which means 5 full weeks).'''

df = df[(df['yearOfRegistration'] >= 1980) & (df['yearOfRegistration'] <= 2016)]
df = df[(df['price'] >= 250) & (df['price'] <= 40000)]
df = df[(df['powerPS'] >= 40) & (df['powerPS'] <= 500)]
df = df[df['advertDuration']<=840]


# In[14]:


'''We can see that operations above made the dataset about two times smaller.
Number of samples decreased from 371528 to 188485.'''

df.info()


# In[15]:


'''Now let's transform postal codes into categories. Basing on an postal division of Germany 
(into 10 leitzones from 0 to 9: https://en.wikipedia.org/wiki/Postal_codes_in_Germany) 
we can take 10 categories.'''

bins = [0, 9999, 19999, 29999, 39999,
        49999, 59999, 69999, 79999, 89999, 99999]

df['postalCode'] = pd.cut(df['postalCode'], bins=bins,
       labels=['Leitzone 0', 'Leitzone 1', 'Leitzone 2', 'Leitzone 3',
              'Leitzone 4', 'Leitzone 5', 'Leitzone 6', 'Leitzone 7',
              'Leitzone 8', 'Leitzone 9'])


# In[16]:


bins = [0, 149999, 999999]

df['kilometer'] = pd.cut(df['kilometer'], bins=bins,
                         labels=['Less than 150000 km', 'Over 150000 km'])


# In[17]:


DFCounter(df)


# In[18]:


print(df.info())


# In[19]:


'''Now we have to balance classes - at first, let's count them.'''

df['brand'].value_counts()


# In[20]:


'''Audi is the least numerous - 28318 adverts.
To make our model more sensible to each class and metrics intuitive,
we have to make quantity of adverts in brands equal.'''

min_cnt = df['brand'].value_counts().min()
df = df.groupby('brand').sample(min_cnt)

df['brand'].value_counts()


# In[21]:


'''Next we have to drop the columns with no importance in modelling.'''

df.drop(columns=['dateCrawled', 'name', 'seller', 'offerType', 'abtest', 'model', 'monthOfRegistration',
       'dateCreated', 'nrOfPictures', 'lastSeen'], inplace=True)


# In[22]:


df.columns


# In[23]:


'''To transform columns with strings into categorical,
we'll define function based on LabelEncoder.'''

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
encode_list = ['gearbox', 'vehicleType', 'fuelType', 'postalCode', 'notRepairedDamage', 'kilometer']

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


# In[24]:


'''Last description of basic statistics should show us that the data are ready to processing.'''

df.describe().apply(lambda x: x.apply('{0:.2f}'.format))


# In[25]:


df.info()


# In[26]:


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


# In[27]:


'''When it comes to continuous variables:

1. Opel is much cheaper than other brands and gains about 100% proportion before 25000 euros. BMW, Audi and Mercedes-Benz
have similar increment of price and gain 100% at about 45000-50000 euros. VW is centrally in the middle between them.

2. Mercedes-Benz has the greatest amount of models older from years 1980-1994 available. The biggest amount of Opels and VWs
comes from last years of XX century, Mercedeses - from the first half of 00s' of XXI century, BMWs and Audis - the second half
of 00s.

3. VWs and Opels has much less powerful models and distrubutions of them are similar. BMW has models no less powerful than 70 HP,
Audi's and Mercedes-Benz's distributions have minimal values at the same point but Audi's reach its peak near BMW's.
VW's and Opel's cars power rarely overpass 250 HP, on the other hand there are quite a lot of BMW's, Audi's and Mercedes' 
vehicles with powers above 320 HP.

'''


for i, col in enumerate(df[['price', 'yearOfRegistration', 'powerPS', 'advertDuration']]):
    plt.figure(i)
    ax = sns.kdeplot(x=col, data=df, hue='brand', fill=True)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


# In[28]:


'''Now let's generate the full report of data. As we can see, there are not much variables with strong correlation. 
Quite strong positive correlation takes place between price and registration year or engine power.
Negative correlation occurs between mileage and registration year or price and between gearbox type and engine power.
Car brands don't correlate strong with any of the other variables, but that's positive signal, because there's no "data leak".
The only continuous variable with distrubution quite similar to normal/Gaussian is the year of registration, 
the rest have strictly skew distributions.'''

report = ProfileReport(df, infer_dtypes=False)
report


# In[29]:


'''Using a pivot table we can take a look at descriptive stats of each brand.
Basing on mean, standard deviation and our previous observations we can see 
that the variables providing the greatest variance are: fuel type, engine power, price and vehicle type.'''

pivot = pd.pivot_table(df, index='brand', values = ['price', 'vehicleType', 'yearOfRegistration', 'kilometer',
                                                    'notRepairedDamage', 'gearbox', 'powerPS', 'fuelType','postalCode'], 
                       aggfunc= [np.mean, np.median, np.std, min, max])
pd.options.display.max_columns = None
display(pivot)


# In[30]:


'''Now we can divide dataset to features (X) and labels (y).'''

X = df.drop(columns='brand')
y = df['brand']


# In[31]:


'''As we can see below, division of the set ended successfully.'''
print(X.sample(5))
print(y.sample(5))


# In[32]:


'''Now we should encode brand names into categorical numbers.'''

y = le.fit_transform(y)
print(y)
print(type(y))


# In[33]:


#Operations made before this cell shouldn't be modified later.


# In[94]:


#first models are going to be created with usage of feature selection
#we'll use methods: mutual_info_classif and SelectFromModel 


# In[34]:


from sklearn.feature_selection import mutual_info_classif
 
importances = mutual_info_classif(X, y)
 
feature_info = pd.Series(importances, X.columns).sort_values(ascending=False)
print(f'Five features providing the biggest information gain are:\n{feature_info.head(5)}')
print(f'Their cumulative information gain equals {np.around(np.sum(feature_info.head(5)), 2)}.')


# In[35]:


from sklearn.feature_selection import SelectFromModel

def SelectorChoiceDisplay(estimator, X, y):
    selector = SelectFromModel(estimator=estimator).fit(X, y)
    print(f'Threshold value used for feature selection: {selector.threshold_}.\n')
    cols = pd.Series(X.columns, name='Is feature supported?')
    support = pd.Series(selector.get_support(), name='Answer')
    concatted = pd.concat([cols, support], join='inner', axis=1)
    print(concatted)


# In[36]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_features=None)
SelectorChoiceDisplay(rfc, X, y)


# In[39]:


X_selected = df[['price', 'vehicleType', 
                'powerPS', 'gearbox',
                'yearOfRegistration', 'advertDuration']]


# In[95]:


rfc = RandomForestClassifier(max_features=None)
SelectorChoiceDisplay(rfc, X_selected, y)


# In[153]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_selected,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify=y)


# In[154]:


X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                    y_train,
                                                    test_size = 0.25,
                                                    random_state = 42,
                                                    stratify=y_train)


# In[43]:


print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_val shape: {y_val.shape}')
print(f'y_test shape: {y_test.shape}')


# In[44]:


first_model = RandomForestClassifier()

start = time.time()
first_model.fit(X_train, y_train)
first_model_train_pred = first_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 1st RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[45]:


from sklearn.metrics import classification_report

first_model_val_pred = first_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, first_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, first_model_val_pred)}')


# In[46]:


from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [3,4,5], 
    'min_samples_split': [2,4,6], 
    'min_samples_leaf': [1,2,3], 
    'n_estimators': [25,50,100]
}

rfc2 = RandomForestClassifier()
second_model = GridSearchCV(rfc2, param_grid=params, cv=10, verbose=10)
start = time.time()
second_model.fit(X_train, y_train)
second_model_train_pred = second_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 2nd RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[47]:


print(second_model.best_estimator_)
print(second_model.best_params_)
print(second_model.best_score_)


# In[49]:


second_model_val_pred = second_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, second_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, second_model_val_pred)}')


# In[55]:


#model with tuned by GridSearch params doesn't have enough decisive space, so classifier is underfitted - 48% on both sets
#this random forest can be too flat


# In[52]:


params = {
    'max_depth': [5,10], 
    'min_samples_split': [2,6], 
    'min_samples_leaf': [1,2], 
    'n_estimators': [100, 250]
}

rfc3 = RandomForestClassifier()
third_model = GridSearchCV(rfc3, param_grid=params, cv=10, verbose=10)
start = time.time()
third_model.fit(X_train, y_train)
third_model_train_pred = third_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 3rd RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[53]:


print(third_model.best_estimator_)
print(third_model.best_params_)
print(third_model.best_score_)


# In[54]:


third_model_val_pred = third_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, third_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, third_model_val_pred)}')


# In[61]:


#deeper forest has far better results - 65% on training and 63% on validation set
#however, it can have better results with greater max depth i more estimators


# In[156]:


params = {
    'max_depth': [10,15], 
    #we won't adjust min_samples_leaf and min_samples_split this time
    'n_estimators': [250, 500]
}

rfc4 = RandomForestClassifier(min_samples_split=6)
fourth_model = GridSearchCV(rfc4, param_grid=params, cv=10, verbose=10)
start = time.time()
fourth_model.fit(X_train, y_train)
fourth_model_train_pred = fourth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 4th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[59]:


print(fourth_model.best_estimator_)
print(fourth_model.best_params_)
print(fourth_model.best_score_)


# In[62]:


fourth_model_val_pred = fourth_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, fourth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, fourth_model_val_pred)}')


# In[63]:


#classifier's scores grew to 82% on training and 72% on validation


# In[65]:


params = {
    'max_depth': [15,20],
    'min_samples_split': [2,4,6],
    'min_samples_leaf': [1,2,3]
    #this time we won't adjust n_estimators - constant number will be 250
}

rfc5 = RandomForestClassifier(n_estimators=250)
fifth_model = GridSearchCV(rfc5, param_grid=params, cv=10, verbose=10)
start = time.time()
fifth_model.fit(X_train, y_train)
fifth_model_train_pred = fifth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 5th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[66]:


print(fifth_model.best_estimator_)
print(fifth_model.best_params_)
print(fifth_model.best_score_)


# In[67]:


fifth_model_val_pred = fifth_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, fifth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, fifth_model_val_pred)}')


# In[68]:


sixth_model = RandomForestClassifier(
    n_estimators=250,
    max_depth=20,
    max_features=None
    #min_samples_leaf and min_samples_split - default (appropriately 1 and 2)
)

start = time.time()
sixth_model.fit(X_train, y_train)
sixth_model_train_pred = sixth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 6th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[69]:


sixth_model_val_pred = sixth_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, sixth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, sixth_model_val_pred)}')


# In[70]:


seventh_model = RandomForestClassifier(
    max_depth=20,
    max_features=None
    #n_estimators, min_samples_leaf and min_samples_split - default (appropriately 100, 1 and 2)
)

start = time.time()
seventh_model.fit(X_train, y_train)
seventh_model_train_pred = seventh_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 7th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[71]:


seventh_model_val_pred = seventh_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, seventh_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, seventh_model_val_pred)}')


# In[72]:


#as we can see, the problem isn't an estimators number, but a tree depth


# In[73]:


params = {
    'max_depth': np.linspace(16,30, num=8)
    #n_estimators, min_samples_leaf and min_samples_split - default (appropriately 100, 1 and 2)
}

rfc8 = RandomForestClassifier(max_features=None)
eighth_model = GridSearchCV(rfc8, param_grid=params, cv=10, verbose=10)
start = time.time()
eighth_model.fit(X_train, y_train)
eighth_model_train_pred = eighth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 8th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[74]:


print(eighth_model.best_estimator_)
print(eighth_model.best_params_)
print(eighth_model.best_score_)


# In[77]:


eighth_model_val_pred = eighth_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, eighth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, eighth_model_val_pred)}')


# In[82]:


#classifier with depth 18 has better scores on validation set despite having worse scores on training


# In[83]:


#we should also check if model demands using all the features in the set or not


# In[79]:


params = {
    'max_features': [None, 'sqrt', 'log2']
}

rfc9 = RandomForestClassifier(max_depth=18)
ninth_model = GridSearchCV(rfc9, param_grid=params, cv=10, verbose=10)
start = time.time()
ninth_model.fit(X_train, y_train)
ninth_model_train_pred = ninth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 9th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[80]:


print(ninth_model.best_estimator_)
print(ninth_model.best_params_)
print(ninth_model.best_score_)


# In[81]:


ninth_model_val_pred = ninth_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, ninth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, ninth_model_val_pred)}')


# In[84]:


#max_feature param has been set to "None", so model uses all the features


# In[86]:


#last thing we should check in this segment is the influence of estimators' number on efficiency of the model


# In[89]:


params = {
    'n_estimators': np.arange(50, 550, step=50)
}

rfc10 = RandomForestClassifier(max_depth=18, max_features=None)
tenth_model = GridSearchCV(rfc10, param_grid=params, cv=10, verbose=10)
start = time.time()
tenth_model.fit(X_train, y_train)
tenth_model_train_pred = tenth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 10th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[90]:


print(tenth_model.best_estimator_)
print(tenth_model.best_params_)
print(tenth_model.best_score_)


# In[91]:


tenth_model_val_pred = tenth_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, tenth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, tenth_model_val_pred)}')


# In[92]:


#model with 400 estimators, according to GridSearchCV, had the best params
#but actually they don't differ from model with 50 and 100 estimators, which are much faster


# In[155]:


from sklearn.metrics import accuracy_score

model_no = 1

for i in [first_model, second_model, third_model,
         fourth_model, sixth_model, seventh_model,
         eighth_model, ninth_model, tenth_model]:
    prediction = i.predict(X_test)
    print(f'Accuracy score of model no {model_no}: {accuracy_score(y_test, prediction)}')
    model_no +=1


# In[145]:


#next models will be created without feature selection
#we're going to use all the features in the X set
#bearing in mind dependencies observed during our first tries of modelling


# In[149]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify=y)


# In[150]:


X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                    y_train,
                                                    test_size = 0.25,
                                                    random_state = 42,
                                                    stratify=y_train)


# In[98]:


print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_val shape: {y_val.shape}')
print(f'y_test shape: {y_test.shape}')


# In[99]:


#first model (11th overall) will be created based on default params of RFC


# In[102]:


eleventh_model = RandomForestClassifier()

start = time.time()
eleventh_model.fit(X_train, y_train)
eleventh_model_train_pred = eleventh_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 11th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[103]:


eleventh_model_val_pred = eleventh_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, eleventh_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, eleventh_model_val_pred)}')


# In[104]:


#eleventh model has perfect scores on training and 77% on validation - it's better than the scores
#of the best model using only the selected features


# In[105]:


params = {
    'max_depth': [10,15], 
    'min_samples_split': [2,6], 
    'min_samples_leaf': [1,2], 
    'n_estimators': [100, 250]
}

rfc12 = RandomForestClassifier()
twelfth_model = GridSearchCV(rfc12, param_grid=params, cv=10, verbose=10)
start = time.time()
twelfth_model.fit(X_train, y_train)
twelfth_model_train_pred = twelfth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 12th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[107]:


print(twelfth_model.best_estimator_)
print(twelfth_model.best_params_)
print(twelfth_model.best_score_)


# In[106]:


twelfth_model_val_pred = twelfth_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, twelfth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, twelfth_model_val_pred)}')


# In[109]:


#results of adjusted by GridSearchCV params aren't optimal, but they're quite better than analogical model created on all features from X
#scores on training set may show that the model is a little underfitted


# In[110]:


params = {
    'max_depth': [16,18], #we'll check the depths of random forest which gave the best results on limited features
    'min_samples_split': [2,4,6],
    'min_samples_leaf': [1,2,3],
    'max_features': [None, 'sqrt', 'log2']
}

rfc13 = RandomForestClassifier()
thirteenth_model = GridSearchCV(rfc13, param_grid=params, cv=10, verbose=10)
start = time.time()
thirteenth_model.fit(X_train, y_train)
thirteenth_model_train_pred = thirteenth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 13th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[112]:


print(thirteenth_model.best_estimator_)
print(thirteenth_model.best_params_)
print(thirteenth_model.best_score_)


# In[113]:


thirteenth_model_val_pred = thirteenth_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, thirteenth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, thirteenth_model_val_pred)}')


# In[116]:


#79% on validation is the best result so far - this model seems optimal
#results are better when max_features = None


# In[117]:


params = {
    'max_depth': [17,19,21], #we'll check only the depths of random forest with no limit on max_features
}

rfc14 = RandomForestClassifier(max_features=None)
fourteenth_model = GridSearchCV(rfc14, param_grid=params, cv=10, verbose=10)
start = time.time()
fourteenth_model.fit(X_train, y_train)
fourteenth_model_train_pred = fourteenth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 14th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[118]:


print(fourteenth_model.best_estimator_)
print(fourteenth_model.best_params_)
print(fourteenth_model.best_score_)


# In[120]:


fourteenth_model_val_pred = fourteenth_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, fourteenth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, fourteenth_model_val_pred)}')


# In[121]:


#this model has minimal advantage over the fourteenth


# In[122]:


params = {
    'n_estimators': [25,50,100], #we'll check only the influence of estimators' numbers
}

rfc15 = RandomForestClassifier(max_depth=19, max_features=None)
fifteenth_model = GridSearchCV(rfc15, param_grid=params, cv=10, verbose=10)
start = time.time()
fifteenth_model.fit(X_train, y_train)
fifteenth_model_train_pred = fifteenth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 15th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[123]:


print(fifteenth_model.best_estimator_)
print(fifteenth_model.best_params_)
print(fifteenth_model.best_score_)


# In[124]:


fifteenth_model_val_pred = fifteenth_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, fifteenth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, fifteenth_model_val_pred)}')


# In[125]:


#also this time model with 100 estimators acted the best


# In[135]:


#penultimate algorithm we're going to test is model with different numbers of sample splits and sample leafs to fit


# In[131]:


params = {
    'min_samples_split': [2,4,6],
    'min_samples_leaf': [1,2,3]
}

rfc16 = RandomForestClassifier(max_depth=19, max_features=None)
sixteenth_model = GridSearchCV(rfc16, param_grid=params, cv=10, verbose=10)
start = time.time()
sixteenth_model.fit(X_train, y_train)
sixteenth_model_train_pred = sixteenth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 16th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[132]:


print(sixteenth_model.best_estimator_)
print(sixteenth_model.best_params_)
print(sixteenth_model.best_score_)


# In[133]:


sixteenth_model_val_pred = sixteenth_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, sixteenth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, sixteenth_model_val_pred)}')


# In[134]:


#just like in other cases, the best version of RFC is that one with max_depth=19, max_features=None
#default n_estimators, min samples split and leaf


# In[136]:


#the last algorithm to test will be that one with max_depth=20


# In[137]:


seventeenth_model = RandomForestClassifier(max_depth=20, max_features=None)
start = time.time()
seventeenth_model.fit(X_train, y_train)
seventeenth_model_train_pred = seventeenth_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of 17th RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[138]:


seventeenth_model_val_pred = seventeenth_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, seventeenth_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, seventeenth_model_val_pred)}')


# In[139]:


#that model also has 79% of accuracy on validation set


# In[140]:


#to test all the models, we have make a prediction on test set


# In[151]:


from sklearn.metrics import accuracy_score

model_no = 11

for i in [eleventh_model, twelfth_model,
          thirteenth_model, fourteenth_model, fifteenth_model,
          sixteenth_model, seventeenth_model]:
    prediction = i.predict(X_test)
    print(f'Accuracy score of model no {model_no}: {accuracy_score(y_test, prediction)}')
    model_no +=1


# In[152]:


#just like we've before, model number 16 showed the best results, therefore we should implement it on later work

