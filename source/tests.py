
# Libraries

import os

if 'source' not in os.listdir():
    os.chdir('..')

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import ClassifierChain

import lightgbm as lgb

# Import Data

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Preprocessing

target_vars = ['EC1', 'EC2']
exclude_features = ['id', 'EC3', 'EC4', 'EC5', 'EC6']
categorical_features = ['fr_COO', 'fr_COO2']
numerical_features = [col for col in train_data.columns if col not in target_vars + exclude_features + categorical_features]

train_data['y'] = train_data.apply(
    lambda x:
    '-'.join([str(int(x[target_var])) for target_var in target_vars]),
    axis = 1
)

# # Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    train_data[numerical_features + categorical_features],
    train_data['y'],
    random_state = 23,
    stratify = train_data['y']
)

# Model

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 23)

lgbm_estimator = Pipeline(
    [
        (
            'column_transformer',
            ColumnTransformer(
                [
                    (
                        'numeric',
                        Pipeline(
                            [
                                ('impute', SimpleImputer(strategy = 'mean')),
                                ('scale', StandardScaler())
                            ]
                        ),
                        numerical_features
                    ),
                    (
                        'categoric',
                        Pipeline(
                            [
                                ('impute', SimpleImputer(strategy = 'most_frequent')),
                                (
                                    'encode', OrdinalEncoder(
                                        handle_unknown = 'use_encoded_value',
                                        unknown_value = -1
                                    )
                                )
                            ]
                        ),
                        categorical_features
                    )
                ]
            )
        ),
        (
            'lgbm',
            lgb.LGBMClassifier(random_state = 23, is_unbalance = True)
        )
    ]
)

lgbm_estimator.fit(X_train, y_train)
e
