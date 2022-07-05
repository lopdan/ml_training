#!/usr/bin/env python
# coding: utf-8

import os
import tarfile
import pandas as pd
import numpy as np
import typing
import matplotlib.pyplot as plt

from zlib import crc32
from six.moves import urllib
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "notebooks/datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # Adds the aditional data next to the existing data (row like)
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]  

def main():
    """
    Main function
    """
    fetch_housing_data()
    housing = load_housing_data()
    training_test_set(housing)
    clean_training_set(housing)

def fetch_housing_data(housing_url: str = HOUSING_URL, housing_path: str = HOUSING_PATH):
    """Fetch data
    Args:
        housing_url (str, optional): house data URL. Defaults to HOUSING_URL.
        housing_path (str, optional): house path data storage. Defaults to HOUSING_PATH.
    """
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path: str = HOUSING_PATH) -> pd.DataFrame:
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def test_set_check(identifier, test_ratio):
    # Check if the test hash if lower or equal to 20% of maxium hash value
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data: pd.DataFrame, test_ratio: int, id_column: str) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    ids: pd.Series = data[id_column]
    in_test_set: pd.Series = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def training_test_set(housing: pd.DataFrame):
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

# Clean the training set:
def clean_training_set(housing: pd.DataFrame):
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"] = housing["population"]/housing["households"]

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)

    housing_cat = housing[["ocean_proximity"]]

    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    housing_cat_1hot.toarray()

# Transformation pipelines
def get_transformer_pipeline():
    # Imput missing values (Transform 1) -> Add relevant new attributes (Transform 2) 
    # -> Scale the data (Transform 3)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    housing_num_tr = num_pipeline.fit_transform(housing_num)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    # Transform numerical columns and categorical columns at the same time
    full_pipeline = ColumnTransformer([
        ("numerical_columns", num_pipeline, num_attribs),
        ("categorical_columns", OneHotEncoder(), cat_attribs),
    ])
    housing_prepared = full_pipeline.fit_transform(housing)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def forest_regressor_train():
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, 
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores)


# Grid Search
def grid_search_train():
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

def get_correlation_features():
    feature_importances = grid_search.best_estimator_.feature_importances_

    extra_attribs = ["rooms_per_household", "population_per_household", "bedrooms_per_room"] #Extra atribs added previously
    cat_encoder = full_pipeline.named_transformers_["categorical_columns"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    sorted(zip(feature_importances, attributes), reverse=True)

def grid_search_model():
    final_model = grid_search.best_estimator_

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    final_rmse

    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    np.sqrt(stats.t.interval(confidence, len(squared_errors)-1, loc=squared_errors.mean(), scale=stats.sem(squared_errors)))


if __name__ == '__main__':
    main()