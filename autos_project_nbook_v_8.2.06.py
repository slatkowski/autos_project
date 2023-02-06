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

'''First group of plots shows that:

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
appear 6-7 less often. Similar situation shows up on plot referring to "kilometer" column.

5. Opels are the most often for sale in Leitzones 4, 5, 6 (west of Germany). Leitzones 8 and 9 (south-eastern and part of central Germany) are the only regions
where Audis and BMWs appear the most frequent. VW "wins" in Leitzone 0 (east of Germany), 2 (north of Germany) and 3 (large part of central Germany), and is
only a little less popular than Mercedes-Benz in Leitzone 1 (north-eastern Germany with the capital city of Berlin). Mercedes' advert frequency is relatively the biggest
is Leitzone 1 and 7 (south-western Germany).

'''

sns.set(style="whitegrid") 

for i, col in enumerate(df[encode_list]):
    plt.figure(i)
    ax = sns.countplot(x=col, data=df, hue='brand')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


# In[27]:


'''When it comes to continuous variables:

1. Opel is much cheaper than other brands and exhaust the distribution before 25000 euros. BMW, Audi and Mercedes-Benz
have similar increment of price and gain 100% at about 45000-50000 euros. VW is centrally in the middle between them.
Peak of all brands' adverts frequency is at level about 1500-3000 euros. All distributions are strictly positive skew.

2. Mercedes-Benz has the greatest amount of models older from years 1980-1994 available. The biggest amount of Opels and VWs
comes from last years of XX century, Mercedeses - from the first half of 00s' of XXI century, BMWs and Audis - the second half
of 00s. For each producer distrubitions of these feature are close to normal.

3. VWs and Opels has much less powerful models and distrubutions of them are similar. BMW has models no less powerful than 70 HP,
Audi's and Mercedes-Benz's distributions have minimal values at the same point but Audi's reach its peak near BMW's.
VW's and Opel's cars power rarely overpass 250 HP, on the other hand there are quite a lot of BMW's, Audi's and Mercedes' 
vehicles with powers above 320 HP.

4. Distrubution showing duration of adverts (which we can consider as effectiveness of advert - how long does it take to sell 
a car from the publication) are quite similar and positive skew - the most adverts cause the transaction in 2-3 days.
The leaders in these category are the brands regarded as the popular brands - VW and Opel.

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


'''We're going to need split the sets on training, validation and test via train_test_split method.'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                    y_train,
                                                    test_size = 0.25,
                                                    random_state = 42,
                                                    stratify=y_train)

print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_val shape: {y_val.shape}')
print(f'y_test shape: {y_test.shape}')


# In[34]:


'''Now we will implement the model chosen during the earlier works. This one will be RandomForestClassifier instance.
Advantages of this algorithm is stability, immunity to overfitting and strong predicting force, compared to neural networks, 
(but much easier to implement and faster to fit) and ability to recognize dependencies between data, unable to see by other
sklearn algorithms.
This model also doesn't demand standarized data. We can use them, but it doesn't have influence on training
and when it comes to visualization, it makes the data less readable.'''

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


# In[35]:


'''Model has 96% accuracy on training set, 79% on validation and 79% on test set. We need to remember that dependencies
between data are not obvious and each other sklearn classifier had big problems with catching them.
Therefore this model's ability to generalize should be regarded as very strong.'''

from sklearn.metrics import classification_report

rfc_model_val_pred = rfc_model.predict(X_val)
rfc_model_test_pred = rfc_model.predict(X_test)
print(f'Training metrics:\n\n{classification_report(y_train, rfc_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, rfc_model_val_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, rfc_model_test_pred)}')


# In[36]:


'''We should display confusion matrix. Let's create function in this purpose.'''

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

brands = le.classes_

def conf_matrix_show(estimator, X, y, classes):
    #1. Drawing the plot, setting the plot's default size.
    fig, ax = plt.subplots(figsize=(8,8))
    #2. Creating the instance of confusion matrix from estimator and set, and displaying classes to predict.
    cmd = ConfusionMatrixDisplay.from_estimator(estimator, X, y, display_labels=classes, ax=ax)
    #3. Disabling grid - it makes confusion matrix much more readable. 
    ax.grid(False)

    
conf_matrix_show(rfc_model, X_train, y_train, brands)
conf_matrix_show(rfc_model, X_val, y_val, brands)
conf_matrix_show(rfc_model, X_test, y_test, brands)


# In[39]:


'''Model has the weakest predicting strength when it comes to VWs (they're relatively often misrecognized as Opels and Audis),
but copes with its predefined task well.'''


# In[40]:


'''Next model we created is the deep neural network model with dense layers.
This kind of layers is the most adequate to tabular data, unstructurized data.
We should start with normalizing the X values.'''

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.fit_transform(X_val)
X_test_sc = scaler.transform(X_test)


# In[43]:


"""We're going to try two models using deep learning.
The model used below uses pretty expanded architecture - 7 hidden layers (640-320-160-80-40-20-10 neurons) with ReLU activation function
and exit layer with 5 neurons (1 for each class) and exit function Softmax.
We'll also set EarlyStopping patience = 6, therefore after 6 epochs with no progress in accuracy on validation set
training will be interrupted."""

import tensorflow as tf

tf.random.set_seed(42)

deep_model = tf.keras.Sequential([
    tf.keras.layers.Dense(640, activation='relu'),
    tf.keras.layers.Dense(320, activation='relu'),
    tf.keras.layers.Dense(160, activation='relu'),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax') #so
])

deep_model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                  patience=6)
]

start = time.time()

history = deep_model.fit(
    X_train_sc, y_train, batch_size=32,
    validation_data=(X_val_sc, y_val),
    epochs=60, callbacks=callback
)
    
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of selected Keras Dense model: {mins} mins and {seconds} s.")


# In[45]:


'''First deep learning model's scores stopped at about 77% of accuracy on training set and
70% on validation and test sets. Training lasted about 18 minutes. Model has quite good metrics, 
but worse than RFC, and it trains much longer.'''

pred_train = deep_model.predict(X_train_sc)
scores_train = deep_model.evaluate(X_train_sc, y_train, verbose=0)
print(f'Accuracy on training data: {scores_train[1]}% \n Error on training data: {1 - scores_train[1]}')   

pred_val = deep_model.predict(X_train_sc)
scores_val = deep_model.evaluate(X_val_sc, y_val, verbose=0)
print(f'Accuracy on validation data: {scores_val[1]}% \n Error on training data: {1 - scores_val[1]}')   
 
pred_test = deep_model.predict(X_test_sc)
scores_test = deep_model.evaluate(X_test_sc, y_test, verbose=0)
print(f'Accuracy on test data: {scores_test[1]}% \n Error on test data: {1 - scores_test[1]}')    


# In[47]:


""""Let's make plots to display the trajectory of loss function and accuracy on training and validation sets.
As we can see in both cases, at 30th (from 42) epoch model stopped endorsing progress.
The accuracies shown on plots are almost the mirror images of loss values."""

fig, ax = plt.subplots(figsize=(9,6))

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[48]:


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

classifier = estimator(deep_model, list(brands))


# In[49]:


'''Now we're using predefined function to show confusion matrixes of every prediction.'''

conf_matrix_show(classifier, X_train_sc, y_train, list(brands))
conf_matrix_show(classifier, X_val_sc, y_val, list(brands))
conf_matrix_show(classifier, X_test_sc, y_test, list(brands))


# In[50]:


'''Also this model has some problems in classifying VWs and - what's surprising - BMWs,
but it has solid metrics.
The next model is going to be more simple - less layers (5) and neurons in hidden layers (500-250-100-50), 
and, what's our intention, faster in training. We'll also make the batch larger and decrease number of epochs.'''

deep_model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(500, activation='relu'), #ReLU simplifies the computing and solves the disappearing gradient problem
    tf.keras.layers.Dense(250, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

deep_model2.compile(
    optimizer='adam', #when using ADAM, we don't have to tune learning rate
    loss='sparse_categorical_crossentropy', #when we have two and more labels, SCC is recommended
    metrics='accuracy'
)

callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                  patience=6)
]

start = time.time()

history = deep_model2.fit(
    X_train_sc, y_train, batch_size=50,
    validation_data=(X_val_sc, y_val),
    epochs=50, callbacks=callback
)
    
stop = time.time()
mins = math.floor((stop - start)/60)
seconds = math.ceil((stop - start)%60)
print(f"Training time of selected Keras Dense model: {mins} mins and {seconds} s.")


# In[52]:


'''The less complex model's scores were about 76% on training and 68% on evaluating sets.
Again, at about 30th epoch (from 45) progress of training became insignificant.'''

pred_train = deep_model2.predict(X_train_sc)
scores_train = deep_model2.evaluate(X_train_sc, y_train, verbose=0)
print(f'Accuracy on training data: {scores_train[1]}% \n Error on training data: {1 - scores_train[1]}')   

pred_val = deep_model2.predict(X_train_sc)
scores_val = deep_model2.evaluate(X_val_sc, y_val, verbose=0)
print(f'Accuracy on validation data: {scores_val[1]}% \n Error on training data: {1 - scores_val[1]}')   
 
pred_test = deep_model2.predict(X_test_sc)
scores_test = deep_model2.evaluate(X_test_sc, y_test, verbose=0)
print(f'Accuracy on test data: {scores_test[1]}% \n Error on test data: {1 - scores_test[1]}')  

fig, ax = plt.subplots(figsize=(9,6))

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[54]:


'''Confusion matrix shows that the problem with this model is weak recall of VW and Audi records.
On validation and test sets that brands had only about 60% of true positive predictions.'''

classifier = estimator(deep_model2, list(brands))

conf_matrix_show(classifier, X_train_sc, y_train, list(brands))
conf_matrix_show(classifier, X_val_sc, y_val, list(brands))
conf_matrix_show(classifier, X_test_sc, y_test, list(brands))


# In[60]:


'''Final conclusions:
- classification problem wasn't easy to solve - data had many dependencies not catched by almost all of
tradition machine learning algorithms from sklearn, because distributions of majority of features 
were overlapping,
- RandomForestClassifier turned out as effective alternative to neural networks - hard to overfit, 
seeing most of dependencies in data and more accurate than deep learning models,
- simple neural network models are not enough to make efficient forecasts - to make effective model,
we need to build complex architecture and we have no warrancy of better results than RFC,
- therefore RFC more is the best of above because of its results, easiness of implementation and learning time.
'''

