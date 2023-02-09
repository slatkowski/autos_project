# 1. Import of necessary libraries.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas_profiling import ProfileReport
import time
import math
import warnings
warnings.filterwarnings("ignore")

'''2. Dataset creation.'''

url = 'https://raw.githubusercontent.com/slatkowski/autos_project/main/autos.csv'
df = pd.read_csv(url)
print(df.head())
print(df.tail())

'''3. Next we should count all values - function to use in this purpose is defined below.'''


def frame_counter(frame):
    # 1. Iteration over columns
    for column in df.columns:
        # 2. Printing of every column values and their numbers
        print(frame[column].value_counts())
        # 3. Cross line to separate and make printing clear
        print('_______________________')


frame_counter(df)

'''4. We have to check when the first and the last advertisement
have been published to make data filtering correct.'''

print(f"Date of the first advertisement: {df['dateCreated'].min()}.")
print(f"Date of the last advertisement: {df['dateCreated'].max()}.")

'''5. Column "index" contains unique values from 0 to 371528, so we can set it as index column.'''

df.set_index('index', inplace=True)

'''6. First information about a DataFrame columns names, non-null values and data types.
Columns "vehicleType", "gearbox", "model", "fuelType" and "notRepairedDamage" contain NaN values.
Also some of the columns which should be the base of prediction, like "vehicleType", "fuelType" or
"gearbox" are object (str) columns.'''

print(df.info())

'''7. Displaying columns with NaN-s; columns "vehicleType", "gearbox", 
"model", "fuelType" and "notRepairedDamage" contain missing values.'''

plt.figure(figsize=(10, 6))
sns.displot(
    data=df.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)

plt.show()

'''8. Next, take a look at the descriptive statistics of DataFrame
(values round to two places after a comma to better readability).
Unfortunately, there are many outliers - maximum value of column "price"
overpasses 2 billions of euro, we have value (values) with price = 0,
in column "yearOfRegistration" we have cars "registered" in year 1000 and 9999.
Also column "kilometer" may show not enough variance - max value (150000)
appears as a median.'''

description = df.describe().apply(lambda x: x.apply('{0:.2f}'.format))
print(description)

'''9. The next description of DataFrame - as we can see, NaNs have been replaced.
We can assume that the columns used in prediction should have numeric values
(except for column "nrOfPicture" in which we have only one value)
and columns "vehicleType", "gearbox", "fuelType" and "postalCode"
with values transformed into discrete numbers.'''

print(df.info())

'''10. Some of the premium brands are hidden behind the term "sonstige_autos". We have to "decode" that records.'''

lst = ['Ferrari', 'Maserati', 'Lexus', 'Aston', 'Bugatti', 'McLaren',
       'Acura', 'Royce', 'Bentley', 'Lamborghini', 'Tesla', 'Infiniti']

lower_lst = [item.lower() for item in lst]
upper_lst = [item.upper() for item in lst]

lst = lst + lower_lst + upper_lst

for i in lst:
    for name in df['name'].values:
        if i in name:
            df.loc[df['name'].str.contains(i), 'brand'] = 'Premium brand'

'''11. Now we have to create two groups of brands (premium and other) instead of original car producers.'''

lst = [(['volkswagen', 'opel', 'ford', 'renault',
         'peugeot', 'fiat', 'seat', 'mazda', 'skoda',
         'citroen', 'nissan', 'toyota', 'hyundai',
         'mitsubishi', 'honda', 'kia', 'suzuki',
         'chevrolet', 'chrysler', 'dacia', 'daihatsu',
         'subaru', 'trabant', 'daewoo', 'rover', 'smart',
         'lada', 'sonstige_autos'], 'Other brand'),
       (['bmw', 'mercedes_benz', 'audi', 'mini', 'volvo',
         'alfa_romeo', 'porsche', 'land_rover', 'jaguar',
         'lancia', 'saab', 'jeep'], 'Premium brand')]

repl_dict = {}
for x, y in lst:
    repl_dict.update(dict.fromkeys(x, y))

df['brand'] = df['brand'].replace(repl_dict)

print(df['brand'].value_counts())

'''12. Next we have to drop the columns with no importance in modelling.'''

df.drop(columns=['dateCrawled', 'name', 'seller', 'offerType', 'abtest',
                 'model', 'monthOfRegistration', 'notRepairedDamage',
                 'dateCreated', 'nrOfPictures', 'lastSeen', 'kilometer', 'postalCode'], inplace=True)

print(df.columns)

'''13. Replacing hidden missing values as NaN-s (in German: "andere" - "other"  
"sonstige autos" - "other cars").'''

df.replace('andere', np.nan, inplace=True)

'''14. The next operation is getting rid of outliers. We're taking into consideration only cars registered in the XXI 
Century, with price in price bracket 200-55000 euros and engine power (in HP/PS) bracket 39 (HP of Fiat Seicento,
one of the less, if not the least powerful car available in XXI Century in Germany) to 500 (some of the brands have 
more powerful models, but they are not common and can be considered as outliers).
'''

df = df[(df['yearOfRegistration'] >= 2001) & (df['yearOfRegistration'] <= 2016)]
df = df[(df['price'] >= 200) & (df['price'] <= 50000)]
df = df[(df['powerPS'] >= 39) & (df['powerPS'] <= 500)]

'''15. We can see that operations above made the dataset about two times smaller.
Number of samples decreased about two times.'''

print(df.info())

'''16. Now we have to balance classes - at first, let's count them.'''

print(df['brand'].value_counts())

'''17. Premium brands are less numerous.
To make our model more sensible to both class and metrics intuitive,
we have to make quantity of adverts in brands equal.'''

min_cnt = df['brand'].value_counts().min()
df = df.groupby('brand').sample(min_cnt)

print(df['brand'].value_counts())

'''18. To replace NaNs with values we should define a function
which replaces them with values according to probability of their appearance
in the column where NaNs appear.'''


def nanfiller(frame):
    # 1. calling function columnFiller to modify column
    def columnfiller(series):
        # 2. assigning number of NaN-s in column to a variable
        nan_c = len(series[series.isna()])
        # 3. taking values from column with no NaN and assigning them to a temporary Series
        nnan_c = series[series.notna()]
        # 4. counting not-NaN values from temporary Series
        from collections import Counter
        import random
        count_nn = Counter(nnan_c)
        # 5. choosing random values according to probabilities of their appearance
        new_val = random.choices(list(count_nn.keys()), weights=list(count_nn.values()), k=nan_c)
        series[series.isna()] = new_val
        # 6. returning column with new values
        return series

    # 7. repeating operation above for the whole DataFrame
    for series in df.columns:
        frame[series] = columnfiller(frame[series])


nanfiller(df)

'''19. To transform columns with strings into categorical,
we'll define function based on LabelEncoder.'''

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encode_list = ['gearbox', 'vehicleType', 'fuelType']


def encodingdesc(series, encoder):
    # 1. transformation of pd.Series/pd.DataFrame.column
    transformed = encoder.fit_transform(series)
    # 2. displaying the name of Series/column
    print(series.name)
    # 3. displaying names of unique classes
    print(encoder.classes_)
    # 4. displaying classes values after encoding
    print(np.unique(transformed))
    # 5. returning transformed Series/column
    return transformed


for i in encode_list:
    df[i] = encodingdesc(df[i], le)
    print('______________________')

'''20. Last descriptions of basic statistics should show us that the data are ready to processing.'''

print(description)

print(df.info())

'''21. To visualize values, we'll take the convention:
- discrete values will be visualized by countplots,
- to display continuous variable we'll use KDE Plot.

Observations from discrete features plot:
1. Nearly 50% of premium brands cars have automatic gearbox in equipment - when it comes to other brands,
percentage falls below 20%.
2. Not premium brands have huge advantage (even four times more) in producing buses (0) and hatchbacks (3). In other 
segments (1 - cabrio, 2 - coupe, 4 - combi, 5 - sedans/limousines) premium brands have much more models on sale - mainly 
in first two of mentioned categories. Numbers of SUV-s (6) are almost equal in both classes.
3. Premium brands have a little more diesels (2) than petrol (0) engines, not premium brands - over a half more petrol 
than diesel. Other kinds of fuel (CNG - 1, electric - 3, hybrid - 4 and LPG - 5) have a marginal meaning.

When it comes to continuous variables:
1. Price distributions are both positive skew, but this of premium brands is much more flattened.
Non premium brands have the most models under 3000 euros, premium reaches peak at about 4000 euros.
At about 6000 euros distributions cross and premium brands start to have more models to offer.
Over 30000 euros almost all offered models are premium.
2. Distributions at "year of registration" column are quite similar, when it comes to years 2001-2005. 
Premium brands appear in more offers from years 2006-2011, offers of newer non premium cars (2012-2016) 
are much more frequent.
3. The biggest variance provides column "power PS". The biggest amount of non premium cars appear with power 
not exceeding 150 HP/PS (peak at about 100 HP) and then number falls drastically.
Premium cars more often have bigger power, exceeding even 400 HP (a little amount of non premium vehicles 
have more than 250 PS).

'''

sns.set(style="whitegrid")

for i, col in enumerate(df[encode_list]):
    plt.figure(i)
    ax = sns.countplot(x=col, data=df, hue='brand')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.show()

for i, col in enumerate(df[['price', 'yearOfRegistration',
                            'powerPS']]):
    plt.figure(i)
    ax = sns.kdeplot(x=col, data=df, hue='brand', fill=True)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.show()

'''22. Analytic report shows that dependent variable has quite much correlation with variable "power". 
None of the independent variables have big correlations between them, so there's no need to perform 
feature extraction.
Also there's no variable with distribution similar to normal. We can also observe that report sees a visible
amount of duplicate rows, but it's probably due to the fact that some of the adverts refer to the similar car models.'''

report = ProfileReport(df, infer_dtypes=False)
report.to_file('profile_report.html')

'''23. Using a pivot table we can take a look at descriptive stats of each brand.
Basing on mean, standard deviation and our previous observations we can see 
that the variables providing the greatest variance are: fuel type, engine power, gearbox and price.'''

pivot = pd.pivot_table(df,
                       index='brand',
                       values=['price', 'vehicleType',
                               'yearOfRegistration', 'gearbox',
                               'powerPS', 'fuelType'],
                       aggfunc=[np.mean, np.median, np.std, min, max])
pd.options.display.max_columns = None
print(pivot)

'''24. Now we can divide dataset to features (X) and labels (y).'''

X = df.drop(columns='brand')
y = df['brand']

'''25. As we can see below, division of the set ended successfully.'''
print(X.sample(5))
print(y.sample(5))

'''26. Now we should encode brand names into categorical numbers.'''

y = le.fit_transform(y)
print(y)
print(type(y))

'''27. We're going to need split the sets on training, validation and test via train_test_split method.'''

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,
                                                  y,
                                                  test_size=0.3,
                                                  random_state=42,
                                                  stratify=y)

X_val, X_test, y_val, y_test = train_test_split(X_val,
                                                y_val,
                                                test_size=0.5,
                                                random_state=42,
                                                stratify=y_val)

print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_val shape: {y_val.shape}')
print(f'y_test shape: {y_test.shape}')

'''28. Normalization of X sets using StandardScaler.'''

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

'''29. Time for modelling. To solve classification problem we've chosen XGBClassifier.
It's efficient classifier which advantage is gradient boosting, using to improve the results of learning,
basing on previous iteration faults.'''

import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    max_depth=10,  # depth of trees
    n_estimators=500,  # no of trees/epochs - not too big to prevent overfitting
    learning_rate=0.1,
    # step size shrinkage - the next learning iteration should be smaller to prevent overfitting;
    # less value demands more iterations
    objective='binary:logistic',  # because it's binary classification issue, the objective couldn't be other
    eval_metric='error',  # measure of model learning progress and the base of possible early stopping
    gamma=0.05,  # minimum loss reduction required to make a further partition on a leaf node of the tree
    reg_lambda=0.7,  # L2 regularization term
    reg_alpha=0.05,  # L1 regularization term
    tree_method='approx'
    # the tree construction algorithm - approximate greedy algorithm using quantile sketch and gradient histogram
    # other params - default
)

start = time.time()
xgb_model.fit(
    X_train_sc, y_train,
    eval_set=[(X_train_sc, y_train), (X_val_sc, y_val)],  # evaluation - on validation set
    early_stopping_rounds=15  # after 15 iterations with no progress training will be interrupted
)

xgb_model_train_pred = xgb_model.predict(X_train_sc)
stop = time.time()
mins = math.floor((stop - start) / 60)
seconds = math.ceil((stop - start) % 60)
print(f"Training time of XGBClassifier model: {mins} mins and {seconds} s.")

'''30. Let's display metrics of model.'''

from sklearn.metrics import classification_report

xgb_model_val_pred = xgb_model.predict(X_val_sc)
xgb_model_test_pred = xgb_model.predict(X_test_sc)
print(f'Training metrics:\n\n{classification_report(y_train, xgb_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, xgb_model_val_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, xgb_model_test_pred)}')

'''31. Model's metrics are good - 92% on training, 90% on validation and on test set.
We should also make plots showing the course of loss function for training and validation set and displaying, 
at which iteration training was optimal.'''

results = xgb_model.evals_result()

from sklearn.metrics import ConfusionMatrixDisplay

plt.figure(figsize=(9, 8))
plt.plot(results["validation_0"]["error"], label="Training loss")
plt.plot(results["validation_1"]["error"], label="Validation loss")
plt.axvline(xgb_model.best_iteration, color="gray", label="Optimal tree number")
plt.xlabel("Number of trees")
plt.ylabel("Loss")
plt.legend()
plt.show()

'''32. To visualize confusion matrix by heatmap, we'll create a function and call it for each set.'''

brands = le.classes_


def conf_matrix_show(estimator, prediction, truth, classes):
    # 1. Drawing the plot, setting the plot's default size.
    fig, ax = plt.subplots(figsize=(8, 8))
    # 2. Creating the instance of confusion matrix from estimator and set, and displaying classes to predict.
    ConfusionMatrixDisplay.from_estimator(estimator,
                                          prediction,
                                          truth,
                                          display_labels=classes,
                                          ax=ax,
                                          values_format=' ')
    # 3. Disabling grid - it makes confusion matrix much more readable.
    ax.grid(False)
    plt.show()


conf_matrix_show(xgb_model, X_train_sc, y_train, brands)
conf_matrix_show(xgb_model, X_val_sc, y_val, brands)
conf_matrix_show(xgb_model, X_test_sc, y_test, brands)

'''33. At the end, let's random decision tree from estimators ensemble.'''

fig, ax = plt.subplots(figsize=(50,50))
xgb.plot_tree(xgb_model, num_trees=1, rankdir='LR', ax=ax)
plt.show()
