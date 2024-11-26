import pandas as pd


class GenerateLagFeatures(object):
    """Generate lag features on an input dataframe.

    Arguments
    ---------
    target_column: str
        the target column to take lags of in input data

    lags: list
        list of integers defining the lags to take

    imputation_value: float
        the value to use for replacing the null values

    Attributes
    ----------
    features_list: list
        list of strings denoting the lag columns of the format
        {target_column}_lag_{lag value}
    """

    def __init__(self, target_column, lags=[1], imputation_value=0):
        """Init and add attributes. Also create additional attrs."""
        # assume lags and target_column has been checked for a correct value
        self.target_column = target_column
        self.lags = lags
        self.imputation_value = imputation_value
        self.features_list = [
            f"{target_column}_lag_{lag}" for lag in self.lags
        ]

    def fit_transform(self, data):
        """Create lags in input data and return the updated dataframe.

        Arguments
        ----------
        data: pd.DataFrame
            the input data

        Returns
        -------
        data: pd.DataFrame
            the output data
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be pandas dataframe, got type {type(data)}"
            )

        data = data.copy()

        for index, lag in enumerate(self.lags):
            data.loc[:, self.features_list[index]] = data.loc[
                :, self.target_column
            ].shift(periods=lag)

            data[self.features_list[index]].fillna(
                self.imputation_value, inplace=True
            )

        return data

    def fit(self, data):
        """Alias function for fit_transform."""
        return self.fit_transform(data)

    def transform(self, data):
        """Alias function for fit_transform."""
        return self.fit_transform(data)
