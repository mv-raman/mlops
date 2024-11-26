"""Test initialization of the GenerateLagFeatures class."""

from pkg_demo.generate_lag_features import GenerateLagFeatures


def test_init():
    """_summary_"""
    features = GenerateLagFeatures(
        target_column="sales", lags=[1, 2, 4], imputation_value=0
    )

    # check if target column, lags and imputation value are correct
    assert features.target_column == "sales"
    assert features.lags == [1, 2, 4]
    assert features.imputation_value == 0

    # check if feature list is correct
    assert features.features_list == [
        "sales_lag_1",
        "sales_lag_2",
        "sales_lag_4",
    ]
