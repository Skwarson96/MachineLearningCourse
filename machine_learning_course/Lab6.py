import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute
import seaborn as sns
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions



def read_data():

    df = datasets.fetch_openml("Titanic", version="1", as_frame=True)

    # print(df.describe())
    # print(df.info())
    # print(df.columns)
    # print(df)

    df_data = df.data
    df_target = df.target

    df_data = df_data.drop(['boat', 'body', 'home.dest'], axis=1)
    df_data = df_data.rename(columns={'pclass': 'TicketClass'})


    return df_data, df_target

def base_solution(y_test):

    y_predicted = np.random.randint(2, size=len(y_test.index))

    label_encoder = preprocessing.LabelEncoder()
    y_test = label_encoder.fit_transform(y_test)

    print("Base random solution:", metrics.classification_report(y_test, y_predicted))

def fill_missing_values(train_data, test_data):

    # Fill enmbarked column
    embarked_impouter = impute.SimpleImputer(missing_values=np.nan,  strategy='most_frequent')

    train_data_filled = train_data.copy()
    test_data_filled = test_data.copy()

    train_data_filled[['embarked']] = embarked_impouter.fit_transform(train_data[['embarked']])
    test_data_filled[['embarked']] = embarked_impouter.fit_transform(test_data[['embarked']])


    # Delete cabin column
    train_data_filled = train_data_filled.drop(['cabin'], axis=1)
    test_data_filled = test_data_filled.drop(['cabin'], axis=1)

    # fare column fill
    fare_imputer  = impute.IterativeImputer(missing_values=np.nan, random_state=42)
    train_data_filled[['fare', 'TicketClass']] = fare_imputer.fit_transform(train_data[[ 'fare', 'TicketClass']])
    test_data_filled[['fare', 'TicketClass']] = fare_imputer.fit_transform(test_data[['fare', 'TicketClass']])

    # Age column fill
    # plt.figure("before fill")
    # plt.hist(train_data_filled['age'], bins=80)

    age_impouter = impute.IterativeImputer(missing_values=np.nan, random_state=42)

    train_data_filled[['age', 'parch', 'sibsp', 'fare', 'TicketClass']] = age_impouter.fit_transform(train_data[['age', 'parch', 'sibsp', 'fare', 'TicketClass']])
    test_data_filled[['age', 'parch', 'sibsp', 'fare', 'TicketClass']] = age_impouter.fit_transform(test_data[['age', 'parch', 'sibsp', 'fare', 'TicketClass']])

    # plt.figure("after fill")
    # plt.hist(train_data_filled['age'], bins=80)
    # plt.show()

    return train_data_filled, test_data_filled

def change_object_values(train_data, test_data):
    list_columns_to_change = ['name', 'sex', 'ticket', 'embarked']

    label_encoder = preprocessing.LabelEncoder()

    for col_name in list_columns_to_change:
        # train data
        label_encoder.fit(train_data[col_name])
        train_data[col_name] = label_encoder.transform(train_data[col_name]).astype(np.float64)
        # test data
        label_encoder.fit(test_data[col_name])
        test_data[col_name] = label_encoder.transform(test_data[col_name]).astype(np.float64)

    return train_data, test_data

def survival_corelation_by_sex(train_data, train_labels):
    train_combined = pd.concat([train_data, train_labels.astype(float)], axis=1)
    df = train_combined[['sex', 'survived']]
    df = df.groupby('sex').mean()
    print(df)

def correlation_and_box_plot(train_data, train_labels):
    X_combined = pd.concat([train_data, train_labels.astype(float)], axis=1)
    print(X_combined.corr().head(5))
    plt.figure()
    c = sns.heatmap(X_combined.corr(), annot=True, cmap='Dark2')

    plt.figure()
    X_combined.boxplot()
    plt.show()

def main():
    X, y = read_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    base_solution(y_test)

    # Find missing data before filling
    # print(X_train.isnull().sum())

    # fill missing values
    X_train_filled, X_test_filled = fill_missing_values(X_train, X_test)

    # change object values to numerical
    X_train_filled, X_test_filled = change_object_values(X_train_filled, X_test_filled)

    clf = svm.SVC()
    clf.fit(X_train_filled, y_train)
    y_predicted = clf.predict(X_test_filled)
    print("SVC", metrics.classification_report(y_test, y_predicted))

    clf_rf = ensemble.RandomForestClassifier()
    clf_rf.fit(X_train_filled, y_train)
    y_predicted_rf = clf_rf.predict(X_test_filled)
    print("RandomForest", metrics.classification_report(y_test, y_predicted_rf))


    # survival_corelation_by_sex(X_train_filled, y_train)

    correlation_and_box_plot(X_train_filled, y_train)




if __name__ == '__main__':

    main()






