import pandas as pd
import numpy as np

def preprocess_data(df):
    '''
    INPUT:
    df (pd.DataFrame) - raw dataset
    OUTPUT:
    x (pd.DataFrame) - preprocessed dataset of features
    y (pd.Series) - preprocessed vector of labels
    
    The function does the following
    1. handling of missing data
    2. splitting dataframe into x (features) and y (labels)
    3. one hot encoding of categorical variables
    4. 0-1 scaling of numerical variables
    '''

    # columns with a high amount of missing data are dropped, 
    df = df.drop(columns=['company','agent','reservation_status','reservation_status_date'],axis=1)
    # columns with a lower amount of missing entries are not dropped entirely, rather rows are dropped where the missing entries are
    df = df.dropna()

    # the DataFrame is split into x and y (features and target)
    x = df.drop(columns=['is_canceled'],axis=1)
    y = df['is_canceled']

    # column names containing categorical and numerical values are separated, since they are handled separately
    cat_vars = x.select_dtypes(include=['object']).copy().columns
    num_vars = x.select_dtypes(include=['float64','int64']).copy().columns

    # checking if all the columns in 'x' were assigned to the categoriacal- and numerical groups
    if len(num_vars)+len(cat_vars) == x.shape[1]:
        print('All columns handled.')
    else:
        print('Error!')

    # One-Hot Encoding of categorical variables
    for var in cat_vars:
        # for each category add dummy var, drop original column
        x = pd.concat([x.drop(var, axis=1), pd.get_dummies(x[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)

    # normalization of numerical values
    for var in num_vars:
        norm = (x[var] - x[var].min()) / (x[var].max() - x[var].min())
        x = pd.concat([x.drop(var, axis=1), norm], axis=1)
    print(x.shape, y.shape)
    
    return x,y


def coef_weights(lm_model, X_train):
    '''
    INPUT:
    lm_model (model) - trained linear model
    X_train (pd.DataFrame) - the training data, so the column names can be used
    OUTPUT:
    coefs_df (pd.DataFrame) - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''

    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_[0]
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_[0])
    #the dataframe returned is sorted by the absolute value of each coefficient, descending
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df
