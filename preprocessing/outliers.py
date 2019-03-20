import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import os


def remove_outliers(input_file, output_file):
    df = pd.read_csv(input_file)
    print(input_file + ' dimensions: ' + str(df.shape))

    lof = LocalOutlierFactor(n_neighbors=20)
    lof.fit(df)
    lof_factor = lof.negative_outlier_factor_

    outlier_factor = 2

    # data frame without outliers
    df = df[lof_factor >= -outlier_factor]
    print(output_file + ' dimensions: ' + str(df.shape))

    # outlier_points = df[lof_factor < -outlier_factor].values
    # print(len(outlier_points))

    df.to_csv(output_file, index=False)
    print('file saved!')


def main():
    input_files = [
        os.path.join('..', 'data_preprocessed', '061_HEK293T_human_embryonic_kidney_transposed_filtered.csv'),
        os.path.join('..', 'data_preprocessed', '065_HEK293T_human_embryonic_kidney_transposed_filtered.csv'),
        os.path.join('..', 'data_preprocessed', '066_HEK293T_human_embryonic_kidney_transposed_filtered.csv'),
        os.path.join('..', 'data_preprocessed', '067_HEK293T_human_embryonic_kidney_transposed_filtered.csv'),
        os.path.join('..', 'data_preprocessed', '068_HEK293T_human_embryonic_kidney_transposed_filtered.csv'),
        os.path.join('..', 'data_preprocessed', '073_HEK293T_human_embryonic_kidney_transposed_filtered.csv'),
        os.path.join('..', 'data_preprocessed', '074_HEK293T_human_embryonic_kidney_transposed_filtered.csv'),
    ]

    output_files = [
        os.path.join('..', 'data_preprocessed', '061_HEK293T_class1.csv'),
        os.path.join('..', 'data_preprocessed', '065_HEK293T_class2.csv'),
        os.path.join('..', 'data_preprocessed', '066_HEK293T_class3.csv'),
        os.path.join('..', 'data_preprocessed', '067_HEK293T_class4.csv'),
        os.path.join('..', 'data_preprocessed', '068_HEK293T_class5.csv'),
        os.path.join('..', 'data_preprocessed', '073_HEK293T_class6.csv'),
        os.path.join('..', 'data_preprocessed', '074_HEK293T_class7.csv'),
    ]

    # I have limited memory, so I will call garbage collector
    # after elimination for every table, just in case
    import gc
    remove_outliers(input_files[0], output_files[0])
    gc.collect()

    remove_outliers(input_files[1], output_files[1])
    gc.collect()

    remove_outliers(input_files[2], output_files[2])
    gc.collect()

    remove_outliers(input_files[3], output_files[3])
    gc.collect()

    remove_outliers(input_files[4], output_files[4])
    gc.collect()

    remove_outliers(input_files[5], output_files[5])
    gc.collect()

    remove_outliers(input_files[6], output_files[6])
    gc.collect()


if __name__ == '__main__':
    main()
