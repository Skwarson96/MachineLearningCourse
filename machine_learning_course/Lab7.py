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







def resampling():
    print("resampling")

    rng = pd.date_range('00:00', periods=40, freq='4min')

    df = pd.Series(np.random.randint(0, 50, len(rng)), index=rng)

    mean = df.resample('5min').mean()
    nearest = df.resample('5min').nearest()
    ffill = df.resample('5min').ffill()
    bfill = df.resample('5min').bfill()


    plt.figure()
    plt.plot(df.index, df)
    plt.plot(mean.index, mean)
    plt.plot(nearest.index, nearest)
    plt.plot(ffill.index, ffill)
    plt.plot(bfill.index, bfill)
    plt.legend(['df', 'mean', 'nearest', 'ffill', 'bfill'])
    plt.show()

def main():
    resampling()



if __name__ == '__main__':
    main()