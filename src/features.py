"""Feature definitions and custom sklearn steps used by both the notebook and the app.

They live in this file (not the notebook) so the saved model can be loaded by the app.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

TARGET = "sale_price"
SUBURBS = {"Blacktown", "Parramatta", "Mosman"}

# Variables as collected, before any feature engineering.
ORIGINAL_NUMERIC = [
    "bedrooms", "bathrooms", "car_spaces", "land_size_sqm", "building_size_sqm",
    "distance_to_sydney_gpo_km", "distance_to_nearest_train_station_km",
    "distance_to_nearest_bus_route_m", "regular_bus_routes_within_250m",
    "regular_bus_routes_within_500m", "regular_bus_routes_within_1km",
]
ORIGINAL_CATEGORICAL = ["suburb", "property_type"]

# Columns in the CSV that were already derived from other columns.
ENGINEERED_IN_CSV = [
    "sale_year", "sale_month", "sale_quarter", "days_since_first_sale", "total_rooms_basic",
    "bathrooms_per_bedroom", "has_parking", "car_spaces_missing", "land_size_sqm_missing",
    "building_size_sqm_missing",
]

# Raw columns the final model uses.
MODEL_NUMERIC = [
    "bedrooms", "bathrooms", "car_spaces", "land_size_sqm", "building_size_sqm",
    "distance_to_nearest_train_station_km", "distance_to_nearest_bus_route_m",
    "regular_bus_routes_within_500m",
]
MODEL_CATEGORICAL = ["suburb", "property_type"]
MODEL_INPUTS = MODEL_NUMERIC + MODEL_CATEGORICAL

# Numeric columns after HousingFeatures has run.
ENGINEERED_NUMERIC = [
    "bedrooms", "bathrooms", "car_spaces", "private_land_sqm", "building_size_sqm",
    "distance_to_nearest_train_station_km", "distance_to_nearest_bus_route_m",
    "regular_bus_routes_within_500m", "bathrooms_per_bedroom",
]
FLAGS = ["land_unknown", "building_unknown", "car_unknown"]


class HousingFeatures(BaseEstimator, TransformerMixin):
    """Adds engineered columns row by row. Nothing is learned from the data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        is_unit = X["property_type"] == "Unit"

        # A unit has no private land parcel, so its land is 0 rather than "unknown".
        X["land_unknown"] = (X["land_size_sqm"].isna() & ~is_unit).astype(int)
        X["private_land_sqm"] = np.where(is_unit, 0.0, X["land_size_sqm"])

        X["building_unknown"] = X["building_size_sqm"].isna().astype(int)
        X["car_unknown"] = X["car_spaces"].isna().astype(int)
        X["bathrooms_per_bedroom"] = X["bathrooms"] / X["bedrooms"].clip(lower=1)
        return X.drop(columns=["land_size_sqm"])


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fills missing values with the median for the same property type.

    Medians come from the training data only. If a type has fewer than
    `min_group` known values (e.g. townhouses), the overall median is used.
    """

    def __init__(self, columns, group_col="property_type", min_group=5, land_col="private_land_sqm"):
        self.columns = columns
        self.group_col = group_col
        self.min_group = min_group
        self.land_col = land_col

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.global_ = X[self.columns].median()
        if self.land_col in self.columns:
            # Units all have land = 0, so the overall land median should only use houses and townhouses.
            not_unit = X[self.group_col] != "Unit"
            self.global_[self.land_col] = X.loc[not_unit, self.land_col].median()

        self.group_ = {}
        for group, rows in X.groupby(self.group_col):
            counts = rows[self.columns].count()
            medians = rows[self.columns].median()
            self.group_[group] = medians.where(counts >= self.min_group, self.global_)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for group in X[self.group_col].unique():
            fill_values = self.group_.get(group, self.global_)
            in_group = X[self.group_col] == group
            X.loc[in_group, self.columns] = X.loc[in_group, self.columns].fillna(fill_values)
        X[self.columns] = X[self.columns].fillna(self.global_)
        return X


class GroupMedianRegressor(BaseEstimator, RegressorMixin):
    """Baseline: predicts the median training price for the same suburb and property type.

    Falls back to the suburb median (then the overall median) if a group has fewer than `min_group` sales.
    """

    def __init__(self, min_group=3):
        self.min_group = min_group

    def fit(self, X, y):
        df = pd.DataFrame(X)[["suburb", "property_type"]].copy()
        df["price"] = np.asarray(y)
        self.overall_ = df["price"].median()
        self.suburb_ = df.groupby("suburb")["price"].median().to_dict()

        stats = df.groupby(["suburb", "property_type"])["price"].agg(["median", "size"])
        big_enough = stats[stats["size"] >= self.min_group]
        self.group_ = big_enough["median"].to_dict()
        return self

    def predict(self, X):
        X = pd.DataFrame(X)
        predictions = []
        for suburb, prop_type in zip(X["suburb"], X["property_type"]):
            fallback = self.suburb_.get(suburb, self.overall_)
            predictions.append(self.group_.get((suburb, prop_type), fallback))
        return np.array(predictions)
