import numpy as np
import pandas as pd

from pkg_demo.generate_lag_features import GenerateLagFeatures


def test_lag_outputs():
    """Test the outputs of the fit_transform function of GenerateLagFeatures class"""
    features = GenerateLagFeatures(
        target_column="sales", lags=[1, 2, 4], imputation_value=0
    )

    data = pd.DataFrame({"sales": np.random.randint(low=1, high=100, size=12)})

    data_w_lags = features.fit_transform(data=data)

    # check data shapes and column names
    assert data_w_lags.shape == (12, 1 + 3)

    # column names
    for lag in [1, 2, 4]:
        assert f"sales_lag_{lag}" in data_w_lags.columns

    # no nulls since imputation value is given
    for col in data_w_lags:
        assert data_w_lags[col].isnull().sum() == 0
