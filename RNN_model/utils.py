import numpy as np


def subsample_sequence(df, length):
    """
    Given the initial dataframe `df`, return a shorter dataframe sequence of length `length`.
    This shorter sequence should be selected at random.
    """

    last_possible = df.shape[0] - length

    random_start = np.random.randint(0, last_possible)
    df_sample = df[random_start:random_start + length]

    return df_sample


def compute_means(X, df_mean):
    '''utils'''
    # Compute means of X
    means = X.mean()

    # Case if ALL values of at least one feature of X are NaN, then reaplace with the whole df_mean
    if means.isna().sum() != 0:
        means.fillna(df_mean, inplace=True)

    return means


def split_subsample_sequence(df,
                             length,
                             df_mean=None,
                             target_name='volume_gross'):
    '''Return one single random sample (X_sample, y_sample) containing one sequence each of length `length`'''
    # Trick to save time during potential recursive calls
    if df_mean is None:
        df_mean = df.mean()

    df_subsample = subsample_sequence(df, length)

    y_sample = df_subsample.iloc[length - 1][target_name]
    # Case y_sample is NaN: redraw !
    if y_sample != y_sample:  # A value is not equal to itself only for NaN
        X_sample, y_sample = split_subsample_sequence(
            df, length, df_mean)  # Recursive call !!!
        return np.array(X_sample), np.array(y_sample)

    X_sample = df_subsample[0:length - 1]
    # Case X_sample has some NaNs
    if X_sample.isna().sum().sum() != 0:
        X_sample = X_sample.fillna(compute_means(X_sample, df_mean))
        X_sample = X_sample.values

    return np.array(X_sample), np.array(y_sample)


def get_X_y(df, n_sequences, length, target_name):
    '''Return a list of samples (X, y)'''
    # $CHALLENGIFY_BEGIN
    X, y = [], []

    for i in range(n_sequences):
        (xi, yi) = split_subsample_sequence(df, length,target_name)
        X.append(xi)
        y.append(yi)

    X = np.array(X)
    y = np.array(y)
    # $CHALLENGIFY_END
    return X, y
