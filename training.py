
# Author: Caner Burc BASKAYA

import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt




#############################
#############################
####                     ####
####      TODO LIST      ####
####  similarity matrix  ####
#### classifier pipeline ####
####     up-sampling     ####
####                     ####
#############################
#############################


###############################
###############################
####                       ####
####     DATA ANALYSIS     ####
####                       ####
###############################
###############################

# source:
# https://go.apziva.com/project/detail/7/


data = pd.read_csv('term-deposit-marketing-2020.csv')
data.name = 'data'

# (40000, 14)
data.shape


data.columns.tolist()

data.dtypes

data.head()

from sklearn.preprocessing import LabelEncoder
import seaborn as sn
from pandas.api.types import is_string_dtype
def plot_correlation_matrix(df, size=16):
    df = df.copy()
    encoder = LabelEncoder()
    feature_columns_to_be_made_numerical = [c for c in df.columns if is_string_dtype(df[c])]
    for c in feature_columns_to_be_made_numerical:
        df[c] = encoder.fit_transform(df[c])
    corrMatrix = df.corr()
    fig, ax = plt.subplots(figsize=(size,size))  
    sn.heatmap(corrMatrix, annot=True, ax=ax)
    plt.show()

plot_correlation_matrix(data)


# prints the value counts as a shortcut
def vc(feature, df=data):
    print('Value Counts of Feature {} in dataframe "{}"'.format(feature, df.name))
    return df[feature].value_counts()

# describes the given feature as a shortcut
def desc(feature, df=data):
    print('Describe the Feature {} in dataframe "{}"'.format(feature, df.name))
    return df[feature].describe()


# Dataset is highly imbalanced, the minority class is y == 'yes'
vc('y')
desc('age')
desc('job')
vc('job')
vc('marital')
vc('education')
vc('default')
desc('balance')
vc('housing')
vc('loan')
vc('contact')
vc('day')
vc('month')
desc('duration')
desc('campaign')



##############################
##############################
####                      ####
####    PRE-PROCESSING    ####
####                      ####
##############################
##############################

# duration values are too diverse with many minor groups and outliers
# data.duration.hist()

# handle outliers by shrinking and discreatization
data.duration = data.duration.apply(lambda x: round(math.log(x+1)))
def handle_outliers_in_duration_by_binning(x):
    if x <=4:
        return 'short'
    elif x >= 7:
        return 'long'
    else:
        return 'mid'

data.duration = data.duration.apply(handle_outliers_in_duration_by_binning)

# data.duration.hist()

# handle campaign times by conjoining 3 or more inro one single group
data.campaign = data.campaign.apply(lambda x: '3' if x > 2 else str(x))

# data.campaign.hist()

# categorizes ages based on the standard deviation of the sequence
def categorize_age(x):
    low_bound = mean - std
    upper_bound = mean + std
    
    if x < low_bound:
        return 'young'
    elif x >= low_bound and x < mean:
        return 'lower_mid'
    elif x < upper_bound and x >= mean:
        return 'upper_mid'
    else:
        return 'aged'


std = data.age.std()
mean = data.age.mean()
data.age = data.age.apply(categorize_age)

# data.age.hist()




# applying logarithmic function for the balance not the best solution in this case
#data.balance.apply(lambda x: math.log(x - data.balance.min() + 100)).hist()
#data[data.balance < 3000].balance.hist()

# categorizes balance based on the standard deviation of the sequence



def categorize_balance(x):
    low_bound = mean - std / 3
    upper_bound = mean + std  / 3
    
    if x < low_bound:
        return 'low'
    elif x >= low_bound and x < upper_bound:
        return 'mid'
    else:
        return 'high'


std = data.balance.std()
mean = data.balance.mean()
data.balance = data.balance.apply(categorize_balance)

# data.balance.hist()


# to catch seasonal effect, it's better to convert months into seasons
# and that would be wise to merge autumn and winter since they barely provide a reasonable quantity of samples togather to form a group

def convert_month_to_season(x):
    if x in ['mar', 'apr', 'may']:
        return 'spring'
    elif x in ['jun', 'jul', 'aug']:
        return 'summer'
    elif x in ['sep', 'oct', 'nov', 'dec', 'jan', 'feb']:
        return 'autumn_winter'

data.month = data.month.apply(convert_month_to_season)
data = data.rename(columns={'month': 'season'})
# data.season.hist()

data.season.value_counts()


# data.day.hist()
# desc('day')

def categorize_days_of_the_month(x):
    low_bound = mean - std * 0.66
    upper_bound = mean + std  * 0.66
    
    if x < low_bound:
        return 'early'
    elif x >= low_bound and x < upper_bound:
        return 'mid'
    else:
        return 'late'


std = data.day.std()
mean = data.day.mean()
data.day = data.day.apply(categorize_days_of_the_month)

# data.day.hist()


# check null values
data.isnull().sum().sum()



# get dummies to make categorical features numeric
def convert_categorical_features_into_numeric(data):
    # convert the booleans beforehand
    for c in data.columns.tolist():
        if len(data[c].value_counts()) == 2:
            data[c] = data[c].apply(lambda x: 1 if x == 'yes' else 0)
    
    
    feature_columns_to_be_made_numerical = [c for c in data.columns if is_string_dtype(data[c]) and c != 'y']
    new_df = pd.get_dummies(data[feature_columns_to_be_made_numerical])
    not_to_convert_columns = [c for c in data.columns if c not in feature_columns_to_be_made_numerical]
    new_df[not_to_convert_columns] = data[not_to_convert_columns]
    
    return new_df


data = convert_categorical_features_into_numeric(data)


# plot_correlation_matrix(data, 38)







##############################
##############################
####                      ####
####    BEST ALGORITHM    ####
####       PIPELINE       ####
####                      ####
##############################
##############################


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split


def start_experiment_for_best_classifier_algorithm():
    
    X = data.drop(['y'], axis = 1)
    y = data.y
    y.sum()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, shuffle=True)
    
    # Initialize the three models
    random_state = 55
    
    clf_A = KNeighborsClassifier(n_neighbors=3)
    clf_B = GaussianNB()
    clf_C = DecisionTreeClassifier()
    clf_D = RandomForestClassifier(random_state=random_state)
    clf_E = SVC()
    clf_F = XGBClassifier(learnin_rate=0.2, max_depth= 8)
    clf_G = AdaBoostClassifier()
    
    
    # Calculate the number of samples for 1%, 10%, and 100% of the training data
    samples_100 = len(y_train)
    samples_10 = int(len(y_train)/10)
    samples_1 = int(len(y_train)/100)
    
    # Collect results on the learners
    results = {}
    for clf in [clf_A, clf_B, clf_C, clf_D, clf_E, clf_F, clf_G]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([ samples_100]): # samples_1, samples_10
            results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)
    
    return results




from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.model_selection import cross_val_score
from time import time
def train_predict(clf, sample_size, X_train, y_train, X_test, y_test):
    
    X = data.drop(['y'], axis = 1)
    y = data.y

    results = {}
    
    # Fit the classifier to the training data 
    # Gets start time
    start = time()
    clf = clf.fit(X_train[:sample_size], y_train[:sample_size])
    # Gets end time
    end = time()
    
    # Calculates the training time
    results['train_time'] = end - start 
    
    # Get the predictions on the test set
    # Gets start time
    start = time()
    predictions_test = clf.predict(X_test)
    predictions_train = clf.predict(X_train)
    # Gets end time
    end = time()
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
    
    # Compute accuracy on the first 300 training samples 
    results['acc_train'] = accuracy_score(y_train, predictions_train)
    
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # Success
    print("{} trained on {} samples.".format(clf.__class__.__name__, sample_size))
    
    # Return the results
    return results



best_classifier_results = start_experiment_for_best_classifier_algorithm()

best_classifier_results




##############################
##############################
####                      ####
####   PARAMETER TUNING   ####
####                      ####
##############################
##############################


from sklearn.model_selection import GridSearchCV
def ExecuteGridSearchCV(clf = AdaBoostClassifier(), parameters=None):
    
    X = data.drop(['y'], axis = 1)
    y = data.y
    
    if parameters == None:
        parameters = {'n_estimators':[30,50,80,120], 'learning_rate':[0.2,0.5,1.0,1.25,2.0], 'algorithm':['SAMME', 'SAMME.R'], 'random_state': [55]}
        
    cr_clf_in_grid = GridSearchCV(clf, param_grid=parameters)
    cr_clf_in_grid.fit(X, y)
    
    return grid_clf, grid_clf.score(X,y)


grid_clf, best_result = ExecuteGridSearchCV(AdaBoostClassifier())
best_result










from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

X = data.drop(['y'], axis = 1)
y = data.y

stratified_folds = StratifiedKFold(n_splits=5, shuffle=True)

for train_indices, test_indices in stratified_folds.split(X, y): 
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        clf = RandomForestClassifier(n_jobs=-2, class_weight='balanced')
        clf.fit(X_train, y_train)
        predictions = y.copy()
        predictions[test_indices] = clf.predict(X_test)



print("cr_accuracy_score = ", accuracy_score(y, predictions))
cr_precision_score = precision_score(y, predictions)
cr_recall_score =  recall_score(y, predictions)
cr_fbeta_score = fbeta_score(y, predictions, beta=0.5)




##############################################
##############################################
####                                      ####
####            --------------            ####
####            --- RESULT ---            ####
####            --------------            ####
####                                      ####
####      BEST PERFORMING CLASSIFIER      ####
####         ADA BOOST CLASSIFIER         ####
####                                      ####
####       5-FOLD CROSS VALIDATION        ####
####           ACCURACY SCORE:            ####
####               0.98695                ####
####                                      ####
##############################################
##############################################






########################################
########################################
####                                ####
####         ---------------        ####
####         ---  BONUS  ---        ####
####         ---------------        ####
####                                ####
####     FEATURE IMPORTANCE:        ####
####                                ####
####     1) DURATION                ####
####     2) HOUSING                 ####
####     3) SPRING SEASON (MONTH)   ####
####                                ####
####                                ####
########################################
########################################


# Feature Selection with Boruta

from boruta import BorutaPy 


X = data.drop(['y'], axis = 1)
y = data.y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, shuffle=True)
    

clf = RandomForestClassifier(n_jobs=-2)



# define Boruta feature selection method
feat_selector = BorutaPy(clf, n_estimators='auto', verbose=2, alpha=0.20)
 
# find all relevant features
feat_selector.fit(X.values, y.values)
 
# check selected features
feat_selector.support_
 
# check ranking of features
feat_selector.ranking_


pd.Series(data.iloc[:,:-1].columns.tolist())[feat_selector.support_]






