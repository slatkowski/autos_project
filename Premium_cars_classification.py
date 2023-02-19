# Import of libraries.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import math
import warnings
import random
import xgboost as xgb
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from imblearn.ensemble import BalancedRandomForestClassifier

warnings.filterwarnings("ignore")


''' Dataset creation, verification of table's contents.
We'll load dataset using the created function. '''


def load_dataset(filename, index_col=None):
    # 1. loading pd.DataFrame from csv file
    dataset = pd.read_csv(f'{filename}', index_col=index_col)
    # 2. removing a columns displaying limit
    pd.options.display.max_columns = None
    # 3. printing head and tail columns
    print(dataset.head())
    print(dataset.tail())
    return dataset


url = 'https://raw.githubusercontent.com/slatkowski/autos_project/main/autos.csv'

df = load_dataset(filename=url)

''' Checking the adverts' publication time to make data filtering correct.
We have to check when the first and the last advertisement have been published.'''

print(f"Date of the first advertisement: {df['dateCrawled'].min()}.")
print(f"Date of the last advertisement: {df['dateCrawled'].max()}.")

''' Setting index column if its values are unique.
Columns "vehicleType", "gearbox", "model", "fuelType" and "notRepairedDamage" contain NaN values.
Also some of the columns which should be the base of prediction, like "vehicleType", "fuelType" or
"gearbox" are object (str) columns.'''

print(df.columns)
print(df['index'].value_counts())

df.set_index('index', inplace=True)

# First information about a DataFrame columns names, non-null values and data types.

print(df.info())

''' First descriptive statistics of DataFrame (values round to two places after a comma to better readability),
outliers searching.
Unfortunately, there are many outliers - f. e. maximum value of column "price" overpasses 2 billions of euro, 
we have value (or even values) with price = 0, in column "yearOfRegistration" we have cars "registered" in year 
1000 and 9999.
Also column "kilometer" may show not enough variance - max value (150000) appears as a median.'''

description = df.describe().apply(lambda c: c.apply('{0:.2f}'.format))
print(description)

'''Counting all values from object cols using defined function.'''


def frame_counter(frame):
    # 1. Iteration over columns
    for series in frame.columns:
        # 2. Printing of every column values and their numbers
        print(frame[series].value_counts())
        # 3. Cross line to separate and make printing clear
        print('_______________________')


str_cols = df.select_dtypes(include='object')
frame_counter(str_cols)

'''Displaying columns with NaN-s.
Columns "vehicleType", "gearbox", "model", "fuelType" and "notRepairedDamage" contain missing values.'''

sns.displot(
    data=df.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.5
)

plt.show()

'''Labels creation - extracting non mentioned premium brands from "name" column.
Some of the premium brands are hidden behind the term "sonstige_autos". We have to "decode" that records.'''


lst = ['ferrari', 'maserati', 'lexus', 'aston', 'bugatti', 'mclaren',
       'acura', 'royce', 'bentley', 'lamborghini', 'tesla', 'infiniti']

df['name'] = df['name'].str.lower()

for i in lst:
    for name in df['name'].values:
        if i in name:
            df.loc[df['name'].str.contains(i), 'brand'] = 'Premium brand'

'''Brands' division into two categories (premium and other) instead of original car producers.'''

lst = [(['volkswagen', 'opel', 'ford', 'renault',
         'peugeot', 'fiat', 'seat', 'mazda', 'skoda',
         'citroen', 'nissan', 'toyota', 'hyundai',
         'mitsubishi', 'honda', 'kia', 'suzuki',
         'chevrolet', 'chrysler', 'dacia', 'daihatsu',
         'subaru', 'trabant', 'daewoo', 'rover', 'smart',
         'lada', 'lancia', 'saab', 'sonstige_autos'], 'Other brand'),
       (['bmw', 'mercedes_benz', 'audi', 'mini', 'volvo',
         'alfa_romeo', 'porsche', 'land_rover', 'jaguar',
         'jeep'], 'Premium brand')]

repl_dict = {}
for x, y in lst:
    repl_dict.update(dict.fromkeys(x, y))

df['brand'] = df['brand'].replace(repl_dict)

print(df['brand'].value_counts())

'''Getting rid of the outliers due to expert knowledge.
We're taking into consideration only cars registered in the XXI century, 
with price in price bracket 200-60000 euros and engine power (in HP/PS) bracket 39 (HP of Fiat Seicento) to 550 
(some of the brands have more powerful models, but they are not common and can be considered as outliers).'''

df = df[(df['yearOfRegistration'] >= 2001) & (df['yearOfRegistration'] <= 2016)]
df = df[(df['price'] >= 200) & (df['price'] <= 60000)]
df = df[(df['powerPS'] >= 39) & (df['powerPS'] <= 550)]

'''Replacing hidden missing values as NaN-s (in German: "andere" - "other").'''

df.replace('andere', np.nan, inplace=True)

'''We can see that operations above made the dataset about two times smaller.'''

print(df.info())

'''Creation of new feature.
Our data suffers from lack of continuous features. We'll create one by dividing prices per engine powers of vehicles.
It's intuitive to assume that we'll have to pay more for 1 HP/PS of premium car engine's power.
'''


df['priceOf1PS'] = df['price'] / df['powerPS']
df = df[df['priceOf1PS'] <= 300]

'''Next look at the values numbers should show us which columns we can drop without information loss.'''


str_cols = df.select_dtypes(include='object')
frame_counter(str_cols)

'''Removing unimportant columns.
We should remove the columns which don't bring useful information (referring to dates, 'name'), 
not referring to cars' properties ('abtest' - it brings us information about the version of the website), 
undifferentiated ('nrOfPictures', 'seller' and 'offerType') and being a data leakage ('model').
'''

df.drop(columns=['dateCrawled', 'name', 'monthOfRegistration', 'seller',
                 'model', 'dateCreated', 'lastSeen', 'nrOfPictures',
                 'offerType', 'abtest'], inplace=True)

print(df.columns)

'''Filtering went correctly, but we still have missing values in four columns.'''


print(df.info())
description = df.describe().apply(lambda c: c.apply('{0:.2f}'.format))
print(description)

sns.displot(
    data=df.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.5
)

plt.show()


'''Missing data replacement - function replacing NaN-s with values due to their frequency.
It replaces them with values according to probability of their appearance in the column where NaNs appear.'''


def nan_filler(frame):
    # 1. calling function columnFiller to modify column
    def column_filler(series):
        # 2. assigning number of NaN-s in column to a variable
        nan_c = len(series[series.isna()])
        # 3. taking values from column with no NaN and assigning them to a temporary Series
        nnan_c = series[series.notna()]
        # 4. counting not-NaN values from temporary Series
        count_nn = Counter(nnan_c)
        # 5. choosing random values according to probabilities of their appearance
        new_val = random.choices(list(count_nn.keys()), weights=list(count_nn.values()), k=nan_c)
        series[series.isna()] = new_val
        # 6. returning column with new values
        return series

    # 7. repeating operation above for the whole DataFrame
    for col in frame.columns:
        frame[col] = column_filler(frame[col])


nan_filler(df)

'''Columns' 'postalCode' and 'kilometer' values don't bring us useful information, when they're continuous.
Good idea might be transforming them into discrete.'''

bins = [0, 9999, 19999, 29999, 39999,
        49999, 59999, 69999, 79999, 89999, 99999]

df['postalCode'] = pd.cut(df['postalCode'], bins=bins,
                          labels=['Leitzone 0', 'Leitzone 1', 'Leitzone 2', 'Leitzone 3',
                                  'Leitzone 4', 'Leitzone 5', 'Leitzone 6', 'Leitzone 7',
                                  'Leitzone 8', 'Leitzone 9'])

bins = [0, 29999, 59999, 99999, 149999, 150001]

df['kilometer'] = pd.cut(df['kilometer'], bins=bins,
                         labels=['0-29999', '30000-59999', '60000-99999', '100000-149999', 'Over 150000 km'])

'''Last insight into analytic reports.
Changes caused about two-fold decrease of samples number.
Data are ready for processing now.'''

description = df.describe().apply(lambda c: c.apply('{0:.2f}'.format))
print(description)
print(df.info())
str_cols = df.select_dtypes(include='object')
frame_counter(str_cols)

'''On normalized plots we can see that:
1. Non-premium cars have huge advantage in hatchback (kleinwagen) and bus models.
Premium models are relatively more likely to be limousines, cabriolets, coupes and estates (kombi).
SUVs frequency are almost the same with a small advantage of premium brands.
2. Advantage of manual gearboxes is merely visible in premium brands group and much bigger in non-premium category.
3. Distributions of first registration dates are quite similar. The most models from each group come from
years 2001-2009. Bigger amount of relatively new models (from years 2014-2016) are non-premium.
4. In column 'powerPS' we can see that there's not much samples of cars with power over 550 HP.
However, we can see that distributions have similar shape, but different values - 
- for non-premium peak is about 100-120 HP, for premium - 180-200 HP.
5. Relatively more premium cars have mileages over 150000 km.
6. More premium cars have diesel engines than petrol, non-premium - opposite. Other fuel types have marginal meaning.
7. Premium cars adverts are relatively more frequent in Southern Germany (Leitzones 6-9). 
However, there's no region in which we have big advantage of any of the groups.
8. Also column showing if car is damaged at the time when advert is published doesn't show much variance.
9. Price distribution of premium brands is much more flattened.
Non-premium brands' distribution reach their peak earlier and falls rapidly. 
From about 10000 euros premium cars are two times more numerous.
That situation is quite similar in column 'priceOf1PS', but differences between both distibutions are closer.
10. Column 'notRepairedDamage', because of lack of variance, will be useless in modelling and should be dropped.'''

_ = df.drop(columns=['brand'])
cols_wo_brands = list(_.columns)

rotation = 45
legend = (1, 1)
figsize = (6, 3.5)

for i in df[cols_wo_brands]:
    fig, ax = plt.subplots(figsize=figsize)
    if df[i].dtype == 'object':
        ax = sns.histplot(x=i, data=df,
                          hue='brand', multiple="dodge",
                          stat='density', shrink=0.8, common_norm=False)
        sns.move_legend(ax, "upper left", bbox_to_anchor=legend)
        plt.xticks(rotation=rotation)
    elif df[i].dtype == 'category':
        ax = sns.histplot(x=i, data=df,
                          hue='brand', multiple="dodge",
                          stat='density', shrink=0.8, common_norm=False)
        sns.move_legend(ax, "upper left", bbox_to_anchor=legend)
        plt.xticks(rotation=rotation)
    else:
        ax = sns.kdeplot(x=i, data=df,
                         hue='brand', common_norm=False)
        sns.move_legend(ax, "upper left", bbox_to_anchor=legend)
        plt.xticks(rotation=rotation)

    plt.tight_layout()
    plt.show()


df.drop(columns='notRepairedDamage', inplace=True)

'''Pivot table displays descriptive statistics divided into brands groups.
Basing on mean, standard deviation and our previous observations we can see that the variables
providing the greatest variance are: gearbox, engine power, total price and price of 1 HP.'''


pivot = pd.pivot_table(df, index='brand', values=['price', 'vehicleType', 'yearOfRegistration',
                                                  'gearbox', 'powerPS', 'fuelType',
                                                  'postalCode', 'kilometer', 'priceOf1PS'],
                       aggfunc=[np.mean, np.median, np.std, min, max])
print(pivot)


'''Transformation of categorical features into numeric labels via defined function.'''

le = LabelEncoder()
encode_list = ['gearbox', 'vehicleType', 'fuelType',
               'kilometer', 'postalCode', 'brand']


def encoding_and_desc(series, encoder):
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
    df[i] = encoding_and_desc(df[i], le)
    print('______________________')


''' On correlations heatmap we can see that that dependent variable has quite much correlation with variable "powerPS".
The strongest correlation (0.82) takes place between total car price and price of 1 HP.'''

cm = df.corr()
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(cm, annot=True)
plt.show()


'''Now we're ready to divide dataset into features (X) and labels (y).'''

X = df.drop(columns='brand')
y = df['brand']

'''Sets splitting - train, validation and test.'''

X_train, X_val, y_train, y_val = train_test_split(X,
                                                  y,
                                                  test_size=0.25,
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

'''Data normalization.'''

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

'''Checking the imbalance between classes is important to set class weights in estimator.'''


counter = Counter(y)
imbalance = counter[0] / counter[1]
print(f'Imbalance of classes equals: {imbalance}.')

'''Modelling - chosen XGBClassifier instance (process of tuning and selection of algorithm - in notebook).
The 1st algorithm we've chosen to solve classification problem is XGBClassifier. It's efficient ensemble classifier,
keeping what's the best from random forest idea and adding gradient boosting, which advantage is improving results 
of learning after each boosting step.
This problem isn't linear and therefore usage of more complicated algorithm could bring more profits.'''

xgb_model = xgb.XGBClassifier(
    max_depth=8,  # depth of the tree estimator
    n_estimators=300,  # number of boosting steps/epochs
    learning_rate=0.3,  # shrinkage rate of the feature weights after each boosting step
    objective='binary:logistic',  # objective - logistic regression for binary classification
    eval_metric='error',  # evaluation metrics
    reg_lambda=0.6,  # L2 regularization rate
    reg_alpha=0.2,  # L1 regularization rate
    scale_pos_weight=imbalance  # balance control rate
)

print("XGBClassifier's training:")
start = time.time()
xgb_model.fit(
    X_train_sc, y_train,
    eval_set=[(X_train_sc, y_train), (X_val_sc, y_val)],
    early_stopping_rounds=15,  # training interruption after 15 steps with no progress
    verbose=10  # tells us about training progress every 10 steps
)

xgb_model_train_pred = xgb_model.predict(X_train_sc)
stop = time.time()
mins = math.floor((stop - start) / 60)
seconds = math.ceil((stop - start) % 60)
print(f"Training time: {mins}:{seconds} mins.")

'''Scores in positive class - premium - are decent.
About 90% of its cases have been detected (as it's indicated by recall in validation and test sets).
The decrease of metrics is nominal in evaluation sets and the recall for true positives is still pretty good.'''


xgb_model_val_pred = xgb_model.predict(X_val_sc)
print(f'Training metrics:\n\n{classification_report(y_train, xgb_model_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, xgb_model_val_pred)}')


'''Confusion matrices plotting.'''

brands = le.classes_


def conf_matrix_show(estimator, prediction, truth, classes):
    # 1. Creating the instance of confusion matrix from estimator and set, and displaying classes to predict.
    ConfusionMatrixDisplay.from_estimator(estimator,
                                          prediction,
                                          truth,
                                          display_labels=classes,
                                          values_format=' ')
    # 2. Disabling grid - it makes confusion matrix much more readable.
    ax.grid(False)
    plt.show()


conf_matrix_show(xgb_model, X_train_sc, y_train, brands)
conf_matrix_show(xgb_model, X_val_sc, y_val, brands)

'''Model stopped making visible learning progress at about 160th estimator.'''

results = xgb_model.evals_result()

plt.plot(results["validation_0"]["error"], label="Training error")
plt.plot(results["validation_1"]["error"], label="Validation error")
plt.axvline(xgb_model.best_iteration, color="gray", label="Optimal tree number")
plt.xlabel("Number of trees")
plt.ylabel("Loss")
plt.legend()


'''Plotting the importance of features.
The biggest importance, due to the model, have engine power. Quite big is also vehicle type's rate.'''


xgb_importance = xgb_model.feature_importances_
xgb_features = X.columns


def plot_feature_importance(importance, names, model_type):
    # 1. Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    # 2. Create a pd.DataFrame using a dict
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_frame = pd.DataFrame(data)
    # 3. Sort the pd.DataFrame in order decreasing feature importance
    fi_frame.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    # 4. Define size of bar plot
    plt.figure(figsize=(8, 6))
    # 5. Plot Seaborn bar chart
    sns.barplot(x=fi_frame['feature_importance'],
                y=fi_frame['feature_names'])
    # 6. Add chart labels
    plt.title(f'Feature importance of {model_type}')
    plt.xlabel('Importance')
    plt.ylabel('Name')
    plt.show()


plot_feature_importance(xgb_importance, xgb_features, xgb_model)


''' RandomForestClassifier is probably the most powerful Scikit-learn ensemble classifier. 
BalancedRFC from Imbalanced-learn module is its variant created especially for imbalanced sets; 
it randomly under-samples each boostrap sample to balance it.
BRFC doesn't need normalized data.'''


brfc_model = BalancedRandomForestClassifier(
    max_depth=18,  # depth of the tree in model
    min_samples_split=4,  # min samples in node
    min_samples_leaf=1,  # min samples in leaf
    n_estimators=100,  # number of trees in forest
    verbose=3
)

print("BalancedRFC's training:")
start = time.time()
brfc_model.fit(X_train, y_train)
brfc_train_pred = brfc_model.predict(X_train)
stop = time.time()
mins = math.floor((stop - start) / 60)
seconds = math.ceil((stop - start) % 60)
print(f"Training time: {mins}:{seconds} mins.")

'''Metrics are also solid. Model has even better recall, when it comes to positive class,
but also has worse precision and that's why F1 score decreases'''

brfc_val_pred = brfc_model.predict(X_val)
print(f'Training metrics:\n\n{classification_report(y_train, brfc_train_pred)}')
print('_____________________________________________________\n')
print(f'Validation metrics:\n\n{classification_report(y_val, brfc_val_pred)}')


'''This algorithm makes more false positive mistakes.'''

conf_matrix_show(brfc_model, X_train, y_train, brands)
conf_matrix_show(brfc_model, X_val, y_val, brands)

'''BRFC features importance plotting.'''

brfc_importance = brfc_model.feature_importances_
brfc_features = X.columns
plot_feature_importance(brfc_importance, brfc_features, brfc_model)

'''As the summary we should display metrics of every model on test set.'''

xgb_model_test_pred = xgb_model.predict(X_test_sc)
print(f'XGB Classifier test metrics:\n\n{classification_report(y_test, xgb_model_test_pred)}')
print('_____________________________________________________\n')
brfc_test_pred = brfc_model.predict(X_test)
print(f'Balanced RFC metrics:\n\n{classification_report(y_test, brfc_test_pred)}')


'''At the end, let's plot confusion matrix of each model's test set prediction.'''

conf_matrix_show(xgb_model, X_test_sc, y_test, brands)
conf_matrix_show(brfc_model, X_test, y_test, brands)
