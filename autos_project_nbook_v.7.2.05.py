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
no longer that 840 hours (35 days, which means 5 full weeks).'''

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


# In[36]:


print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_val shape: {y_val.shape}')
print(f'y_test shape: {y_test.shape}')


# In[37]:


from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier(max_depth=19, 
                                max_features=None)

start = time.time()
rfc_model.fit(X_train, y_train)
rfc_model_train_pred = rfc_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of selected RandomForestClassifier model: {mins} mins and {seconds} s.")


# In[38]:


from sklearn.metrics import classification_report

rfc_model_val_pred = rfc_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, rfc_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, rfc_model_val_pred)}')


# In[103]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

brands = le.classes_

def conf_matrix_show(estimator, X, y, classes, size: tuple):
    fig, ax = plt.subplots(figsize=size)
    cmd = ConfusionMatrixDisplay.from_estimator(estimator, X, y, display_labels=classes, ax=ax)
    ax.grid(False)
    

conf_matrix_show(rfc_model, X_train, y_train, brands, size=(10,10))
conf_matrix_show(rfc_model, X_val, y_val, brands, size=(10,10))


# In[132]:


#next model - we'll try and compare deep learning efficiency to our first model
#we'll create model with dense layers which are the most adequate to tabular data


# In[51]:


#first step is normalizing the X values


# In[47]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.fit_transform(X_val)
X_test_sc = scaler.transform(X_test)


# In[76]:


"""First model is going to have 4 dense layers:
- 360 neurons with ReLU activation function,
- 180 neurons - ReLU,
- 30 neurons - ReLU,
- exit layer - 5 neurons (1 for each class) - function Softmax.
We'll also set EarlyStopping patience = 10, therefore after 10 epochs with no progress
training will be interrupted."""

import tensorflow as tf

tf.random.set_seed(42)

seq_model = tf.keras.Sequential([
    tf.keras.layers.Dense(360, activation='relu'),
    tf.keras.layers.Dense(180, activation='relu'),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

seq_model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                  patience=10)
]

h = seq_model.fit(
    X_train_sc, y_train, batch_size=32,
    validation_data=(X_val_sc, y_val),
    epochs=150, callbacks=callback
)


# In[93]:


""""First model with usage of deep neural network had solid results - about 72% on training set and 65% on validation set.
We should display the run of loss function of training and valid set, do do accuracy on both sets."""

fig, ax = plt.subplots(figsize=(9,6))

plt.plot(h.history['loss'], label='training loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.grid(True)
plt.show()


# In[94]:


fig, ax = plt.subplots(figsize=(9,6))

plt.plot(h.history['accuracy'], label='training accuracy')
plt.plot(h.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[95]:


'''To display confusion matrix we should define the class containing an estimator,
which results we're going to see.'''

class estimator:
  _estimator_type = ''
  classes_=[]
  def __init__(self, model, classes):
    self.model = model
    self._estimator_type = 'classifier'
    self.classes_ = classes
  def predict(self, X):
    y_prob= self.model.predict(X)
    y_pred = y_prob.argmax(axis=1)
    return y_pred

classifier = estimator(seq_model, list(brands))


# In[104]:


'''To display confusion matrix we should use our defined function'''

conf_matrix_show(classifier, X_train_sc, y_train, list(brands), size=(10,10))
conf_matrix_show(classifier, X_val_sc, y_val, list(brands), size=(10,10))


# In[119]:


seq_model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(600, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dense(300, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dense(25, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

seq_model2.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                  patience=20)
]

h = seq_model2.fit(
    X_train_sc, y_train, batch_size=32,
    validation_data=(X_val_sc, y_val),
    epochs=300, callbacks=callback
)


# In[ ]:


fig, ax = plt.subplots(figsize=(9,6))

plt.plot(h.history['loss'], label='training loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(9,6))

plt.plot(h.history['accuracy'], label='training accuracy')
plt.plot(h.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[120]:


"""Second tf model got stuck at about 66-67% on training and validation sets."""

classifier2 = estimator(seq_model2, list(brands))

conf_matrix_show(classifier2, X_train_sc, y_train, list(brands), size=(10,10))
conf_matrix_show(classifier2, X_val_sc, y_val, list(brands), size=(10,10))


# In[121]:


"""As we can see, it has bigger problems than the first with detecting VWs."""


# In[122]:


"""The third model will be smaller than first two, but we'll use regularization."""


seq_model3 = tf.keras.Sequential([
    tf.keras.layers.Dense(180, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'),
    tf.keras.layers.Dense(30, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

seq_model3.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                  patience=10)
]

h = seq_model3.fit(
    X_train_sc, y_train, batch_size=40,
    validation_data=(X_val_sc, y_val),
    epochs=150, callbacks=callback
)


# In[ ]:


fig, ax = plt.subplots(figsize=(9,6))

plt.plot(h.history['loss'], label='training loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(9,6))

plt.plot(h.history['accuracy'], label='training accuracy')
plt.plot(h.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


classifier3 = estimator(seq_model3, list(brands))

conf_matrix_show(classifier3, X_train_sc, y_train, list(brands), size=(10,10))
conf_matrix_show(classifier3, X_val_sc, y_val, list(brands), size=(10,10))


# In[123]:


"""Third model seemed underfitted, maybe it's because of too big regularization element."""

seq_model4 = tf.keras.Sequential([
    tf.keras.layers.Dense(180, activation='relu'),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

seq_model4.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                  patience=6)
]

h = seq_model4.fit(
    X_train_sc, y_train, batch_size=40,
    validation_data=(X_val_sc, y_val),
    epochs=150, callbacks=callback
)


# In[ ]:


fig, ax = plt.subplots(figsize=(9,6))

plt.plot(h.history['loss'], label='training loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(9,6))

plt.plot(h.history['accuracy'], label='training accuracy')
plt.plot(h.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


classifier4 = estimator(seq_model4, list(brands))

conf_matrix_show(classifier4, X_train_sc, y_train, list(brands), size=(10,10))
conf_matrix_show(classifier4, X_val_sc, y_val, list(brands), size=(10,10))


# In[124]:


"""About 10% better results without regularization. We'll test the biggest network again."""

seq_model5 = tf.keras.Sequential([
    tf.keras.layers.Dense(600, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

seq_model5.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                  patience=6)
]

h = seq_model5.fit(
    X_train_sc, y_train, batch_size=32,
    validation_data=(X_val_sc, y_val),
    epochs=200, callbacks=callback
)


# In[125]:


#after 37 epochs scores are only a little better - 74,5% on training and 68,5% on validation.


# In[ ]:


fig, ax = plt.subplots(figsize=(9,6))

plt.plot(h.history['loss'], label='training loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(9,6))

plt.plot(h.history['accuracy'], label='training accuracy')
plt.plot(h.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


classifier5 = estimator(seq_model5, list(brands))

conf_matrix_show(classifier5, X_train_sc, y_train, list(brands), size=(10,10))
conf_matrix_show(classifier5, X_val_sc, y_val, list(brands), size=(10,10))


# In[127]:


"""The next deep model is going to have the most expanded architecture."""

seq_model6 = tf.keras.Sequential([
    tf.keras.layers.Dense(900, activation='relu'),
    tf.keras.layers.Dense(600, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

seq_model6.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                  patience=6)
]

h = seq_model6.fit(
    X_train_sc, y_train, batch_size=32,
    validation_data=(X_val_sc, y_val),
    epochs=200, callbacks=callback
)


# In[128]:


fig, ax = plt.subplots(figsize=(9,6))

plt.plot(h.history['loss'], label='training loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.grid(True)
plt.show()


# In[129]:


fig, ax = plt.subplots(figsize=(9,6))

plt.plot(h.history['accuracy'], label='training accuracy')
plt.plot(h.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[130]:


#Network's final scores - about 77,5% accuracy on training set and 71% on validation set.


# In[131]:


classifier6 = estimator(seq_model5, list(brands))

conf_matrix_show(classifier6, X_train_sc, y_train, list(brands), size=(10,10))
conf_matrix_show(classifier6, X_val_sc, y_val, list(brands), size=(10,10))


# In[135]:


#Dense neural network gave much worse results than RandomForestClassifier despite uncomparable longer time of fitting


# In[152]:


#Conclusion - deep neural networks demand much longer learning process and don't give good enough metrics.

