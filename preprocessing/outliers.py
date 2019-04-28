from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
import pandas as pd
import os


def remove_outliers(d_frame):
    """
    detect outliers with Local Outlier Factor
    delete them

    :param d_frame: pandas.core.frame
    :return: data frame without outliers
    """

    lof = LocalOutlierFactor(n_neighbors=5)
    lof.fit(d_frame)
    lof_factor = lof.negative_outlier_factor_

    outlier_factor = 1.8
    cluster_df = d_frame[lof_factor >= -outlier_factor]
    outlier_df = d_frame[lof_factor < -outlier_factor]

    return cluster_df, outlier_df


if __name__ == '__main__':
    df = pd.read_csv(os.path.join('..', 'data', 'combined_data.csv'), index_col=False)
    print(df.shape)

    df, outliers_df = remove_outliers(df)
    print('data without outliers'.format(df.shape))
    print('outliers'.format(outliers_df.shape))

    # we will change class names here: 1 -> class1, 2 -> class2, etc.
    df.loc[df['class'] == 1, 'class'] = 'class1'
    df.loc[df['class'] == 2, 'class'] = 'class2'
    df.loc[df['class'] == 3, 'class'] = 'class3'
    df.loc[df['class'] == 4, 'class'] = 'class4'
    df.loc[df['class'] == 5, 'class'] = 'class5'
    df.loc[df['class'] == 6, 'class'] = 'class6'
    df.loc[df['class'] == 7, 'class'] = 'class7'

    outliers_df.loc[outliers_df['class'] == 1, 'class'] = 'class1'
    outliers_df.loc[outliers_df['class'] == 2, 'class'] = 'class2'
    outliers_df.loc[outliers_df['class'] == 3, 'class'] = 'class3'
    outliers_df.loc[outliers_df['class'] == 4, 'class'] = 'class4'
    outliers_df.loc[outliers_df['class'] == 5, 'class'] = 'class5'
    outliers_df.loc[outliers_df['class'] == 6, 'class'] = 'class6'
    outliers_df.loc[outliers_df['class'] == 7, 'class'] = 'class7'

    # number of class instances for data frames
    classes = df['class']
    outliers_classes = outliers_df['class']

    print('data classes count:')
    print(sorted(Counter(classes).items()))

    print('outliers classes count:')
    print(sorted(Counter(outliers_classes).items()))

    # save
    df.to_csv(os.path.join('..', 'data', 'data_without_outliers.csv'), index=False)
    outliers_df.to_csv(os.path.join('..', 'data', 'outliers_data.csv'), index=False)
