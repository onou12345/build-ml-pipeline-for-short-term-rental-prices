import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data):

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


########################################################
# Implement here test_row_count and test_price_range   #
def test_row_count(data):
    """
    Test if the number of rows in the dataset falls within the acceptable range.

    This function asserts that the number of rows in the input DataFrame `data` 
    is greater than 15,000 and less than 1,000,000. If the condition is not met, 
    the function will raise an AssertionError.

    Parameters:
    data (pandas.DataFrame): The input DataFrame to be checked.

    Raises:
    AssertionError: If the number of rows is not within the range (15,000, 1,000,000).
    """
    assert 15000 < data.shape[0] < 1000000


def test_price_range(data, min_price, max_price):
    """
    Test if all prices in the dataset fall within a specified range.

    This function asserts that all values in the 'price' column of the input DataFrame `data`
    are between `min_price` and `max_price`. If any price is outside this range, 
    an AssertionError will be raised.

    Parameters:
    data (pandas.DataFrame): The input DataFrame containing the 'price' column.
    min_price (float or int): The minimum acceptable price.
    max_price (float or int): The maximum acceptable price.

    Raises:
    AssertionError: If any price is outside the range [min_price, max_price].
    """
    assert data['price'].between(min_price, max_price).all()
