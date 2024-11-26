"""Linear Regression model that uses lags as features."""
from sklearn.linear_models import LinearRegression
from pkg_demp.generate_lag_features import GenerateLagFeatures


class LinearRegressionAR(object):
    """Linear Regression model that uses lags as the features."""

    def __init__(self, target_column, lags=1):
        """Initialize the attributes and the lage generator."""
        self.target_column = self._check_target_column(target_column)

        self.lags = self._correctly_assign_lags(lags)

        self.lags_generator = GenerateLagFeatures(
            target_column=self.target_column,
            lags=self.lags,
            imputation_value=0,
        )
        self.features_list = self.lags_generator.features_list

        self.model = LinearRegression()

    def _correctly_assign_lags(self, lags):
        """_summary_

        Args:
            lags (_type_): _description_
        Returns:
            _type_: _description_
        """
        if not isinstance(lags, [list, int]):
            raise TypeError(f"lags must be an int or list, got {lags} instead")

        if isinstance(lags, int) and not lags > 0:
            raise ValueError(f"if integer, lags must be positive, got {lags}")

        if isinstance(lags, int):
            lags = list(range(1, lags + 1))

        if isinstance(lags, list) and not lags:
            raise ValueError("lags must not be an empty list")

        if isinstance(lags, list):
            for lag in lags:
                if not isinstance(lag, int):
                    raise ValueError(
                        f"lags must only contain integers, got {lag}"
                    )

        return lags

    def _check_target_column(self, target_column):
        """_summary_

        Args:
            target_column (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not isinstance(target_column, str):
            raise TypeError(
                f"target_column must be a string, got {target_column}"
            )

        return target_column

    def fit(self, data):
        """Fits the model initialized in init.

        This function does not return Anything.
        """
        data = data.copy()

        transformed_data = self.lags_generator.fit_transform(data)
        self.model = LinearRegression()
        self.model.fit(
            transformed_data.loc[:, self.features_list].to_numpy(),
            transformed_data.loc[:, self.target_column].to_numpy(),
        )

    def predict(self, data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        data = data.copy()

        transformed_data = self.lags_generator.fit_transform(data)
        outputs = self.model.predict(
            transformed_data.loc[:, self.features_list].to_numpy()
        )

        return outputs
