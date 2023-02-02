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


# In[ ]:


#Operations made before this cell shouldn't be modified later.


# In[35]:


from sklearn.feature_selection import mutual_info_classif
 
importances = mutual_info_classif(X, y)
 
feature_info = pd.Series(importances, X.columns).sort_values(ascending=False)
print(f'Five features providing the biggest information gain are:\n{feature_info.head(5)}')
print(f'Their cumulative information gain equals {np.around(np.sum(feature_info.head(5)), 2)}.')


# In[36]:


from sklearn.feature_selection import SelectFromModel

def SelectorChoiceDisplay(estimator, X, y):
    selector = SelectFromModel(estimator=estimator).fit(X, y)
    print(f'Threshold value used for feature selection: {selector.threshold_}.\n')
    cols = pd.Series(X.columns, name='Is feature supported?')
    support = pd.Series(selector.get_support(), name='Answer')
    concatted = pd.concat([cols, support], join='inner', axis=1)
    print(concatted)


# In[37]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_features=None)
SelectorChoiceDisplay(rfc, X, y)


# In[34]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify=y)


# In[35]:


X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                    y_train,
                                                    test_size = 0.25,
                                                    random_state = 42,
                                                    stratify=y_train)


# In[40]:


print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_val shape: {y_val.shape}')
print(f'y_test shape: {y_test.shape}')


# In[41]:


first_model = RandomForestClassifier()

start = time.time()
first_model.fit(X_train, y_train)
first_model_train_pred = first_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of RandomForestClassifier with default params: {mins} mins and {seconds} s.")


# In[42]:


from sklearn.metrics import classification_report

first_model_val_pred = first_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, first_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, first_model_val_pred)}')


# In[50]:


#not tuned RandomForestClassifier had 100% efficiency on training set and 78% on validation set


# In[ ]:


#classifying problem isn't obvious and it's problematic to find one feature explaining difference between classes
#because of that we should test other classifiers, the most efficient in complex problems
#let's test artificial neural networks


# In[43]:


from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()
sc = StandardScaler()


# In[44]:


X_train_std = sc.fit_transform(X_train)
X_val_std = sc.transform(X_val)
X_test_std = sc.transform(X_test)


# In[46]:


start = time.time()
mlp.fit(X_train_std, y_train)
mlp_train_pred = mlp.predict(X_train_std)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of MLPClassifier with default params: {mins} mins and {seconds} s.")


# In[58]:


mlp_val_pred = mlp.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, mlp_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, mlp_val_pred)}')


# In[48]:


mlp2 = MLPClassifier(
    activation='relu', 
    hidden_layer_sizes=(400, 250, 125, 50, 5),
    early_stopping=True,
    max_iter=3000
)
    

start = time.time()
mlp2.fit(X_train_std, y_train)
mlp2_train_pred = mlp2.predict(X_train_std)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of MLPClassifier: {mins} mins and {seconds} s.")


# In[52]:


mlp2_val_pred = mlp2.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, mlp2_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, mlp2_val_pred)}')


# In[51]:


#10 minutes of fitting - 20% of accuracy on validation set; only scores on training set are better
#MLPClassifier is unable to generalize in this case


# In[54]:


from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier()

start = time.time()
abc.fit(X_train_std, y_train)
abc_train_pred = abc.predict(X_train_std)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of AdaBoostClassifier with default params: {mins} mins and {seconds} s.")


# In[55]:


abc_val_pred = abc.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, abc_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, abc_val_pred)}')


# In[56]:


abc2 = AdaBoostClassifier(
    n_estimators=250,
    learning_rate=0.5,
    base_estimator=RandomForestClassifier()
)

start = time.time()
abc2.fit(X_train_std, y_train)
abc2_train_pred = abc2.predict(X_train_std)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of AdaBoostClassifier: {mins} mins and {seconds} s.")


# In[57]:


abc2_val_pred = abc2.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, abc2_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, abc2_val_pred)}')


# In[65]:


#let's test KNeighborsClassifier


# In[72]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

start = time.time()
knn.fit(X_train_std, y_train)
knn_train_pred = knn.predict(X_train_std)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of KNeighborsClassifier with default params: {mins} mins and {seconds} s.")


# In[73]:


knn_val_pred = knn.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, knn_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, knn_val_pred)}')


# In[74]:


knn2 = KNeighborsClassifier(
    n_neighbors=25,
    weights='distance',
    algorithm='brute'
)

start = time.time()
knn2.fit(X_train_std, y_train)
knn2_train_pred = knn2.predict(X_train_std)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of KNeighborsClassifier: {mins} mins and {seconds} s.")


# In[75]:


knn2_val_pred = knn2.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, knn2_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, knn2_val_pred)}')


# In[76]:


#KNC fitted perfectly on training set, but showed absolutely no progress when it came to valid set


# In[77]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

start = time.time()
gbc.fit(X_train_std, y_train)
gbc_train_pred = gbc.predict(X_train_std)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of GradientBoostingClassifier: {mins} mins and {seconds} s.")


# In[78]:


gbc_val_pred = gbc.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, gbc_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, gbc_val_pred)}')


# In[79]:


#not tuned GBC also turned out unefficient - the last two estimators will be ExtraTreesClassifier and MultinomialNB


# In[80]:


from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier()

start = time.time()
etc.fit(X_train_std, y_train)
etc_train_pred = etc.predict(X_train_std)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of ExtraTreesClassifier: {mins} mins and {seconds} s.")


# In[81]:


etc_val_pred = etc.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, etc_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, etc_val_pred)}')


# In[82]:


#ETC had the same problem as KNC and will not be considered later


# In[36]:


from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

start = time.time()
mnb.fit(X_train, y_train)
mnb_train_pred = mnb.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of MultinomialNB with default params: {mins} mins and {seconds} s.")


# In[39]:


mnb_val_pred = mnb.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, mnb_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, mnb_val_pred)}')


# In[44]:


#because MNB with default params acted instantly, we'll try to adjust params with GridSearchCV

from sklearn.model_selection import GridSearchCV

params = {
    'alpha': np.linspace(0, 1, num=10),
}

mnb = MultinomialNB()
gs = GridSearchCV(mnb, param_grid=params, cv=5, verbose=10)
start = time.time()
gs.fit(X_train, y_train)
mnb_train_pred = gs.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of MultinomialNB with params adjusted by GridSearchCV: {mins} mins and {seconds} s.")


# In[45]:


print(gs.best_estimator_)
print(gs.best_params_)


# In[46]:


mnb_val_pred = gs.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, mnb_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, mnb_val_pred)}')


# In[ ]:


#MultinomialNB didn't show any good scores at all


# In[64]:


#because RFC showed the best metrics both sets without tuning, we should focus on it

