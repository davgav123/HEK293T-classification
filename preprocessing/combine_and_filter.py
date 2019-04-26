import pandas as pd
import os


def transpose_and_set_column_names(df):
    """
    1. data will be transposed
    2. the first columns will be removed
    3. the same column will be placed for column names

    :param df: pandas.core.frame
    :return df: pandas.core.frame
    """

    # transpose
    df = df.T

    # set new column names and remove first column
    hg_names = df.iloc[0, :]
    df = df.iloc[1:, :]
    df.columns = hg_names

    return df


def combine_files(files):
    """
    create pandas data frame for every file
    transpose each one of them and change it's column names
    add class column
    combine them into one file

    :param files: list of paths
    :return: pandas.core.frame
    """

    # there are seven files, transpose and set col names for each
    df061 = pd.read_csv(files[0], index_col=False)
    df061 = transpose_and_set_column_names(df061)

    df065 = pd.read_csv(files[1], index_col=False)
    df065 = transpose_and_set_column_names(df065)

    df066 = pd.read_csv(files[2], index_col=False)
    df066 = transpose_and_set_column_names(df066)

    df067 = pd.read_csv(files[3], index_col=False)
    df067 = transpose_and_set_column_names(df067)

    df068 = pd.read_csv(files[4], index_col=False)
    df068 = transpose_and_set_column_names(df068)

    df073 = pd.read_csv(files[5], index_col=False)
    df073 = transpose_and_set_column_names(df073)

    df074 = pd.read_csv(files[6], index_col=False)
    df074 = transpose_and_set_column_names(df074)

    # also, add class column
    df061['class'] = 'class1'
    df065['class'] = 'class2'
    df066['class'] = 'class3'
    df067['class'] = 'class4'
    df068['class'] = 'class5'
    df073['class'] = 'class6'
    df074['class'] = 'class7'

    # combine files
    df = pd.concat([df061, df065, df066, df067, df068, df073, df074], axis=0, ignore_index=True)

    return df


def filter_zeros(df):
    """
    delete columns that contains nothing else but zeros
    :param df: pandas.core.frame
    :return: filtered pandas.core.frame
    """
    return df.loc[:, (df != 0).any(axis=0)]


if __name__ == '__main__':
    input_files = [
        os.path.join('..', 'data', '061_HEK293T_human_embryonic_kidney_csv.csv'),
        os.path.join('..', 'data', '065_HEK293T_human_embryonic_kidney_csv.csv'),
        os.path.join('..', 'data', '066_HEK293T_human_embryonic_kidney_csv.csv'),
        os.path.join('..', 'data', '067_HEK293T_human_embryonic_kidney_csv.csv'),
        os.path.join('..', 'data', '068_HEK293T_human_embryonic_kidney_csv.csv'),
        os.path.join('..', 'data', '073_HEK293T-human_embryonic_kidney_matcsv.csv'),
        os.path.join('..', 'data', '074_HEK293T-human_embryonic_kidney_csv.csv'),
    ]

    df = combine_files(input_files)
    print(df.shape)

    df = filter_zeros(df)
    print(df.shape)

    df.to_csv(os.path.join('..', 'data', 'combined_data.csv'), index=False)