import pandas as pd

class ToxicCommentData:

    def __init__(self,data: pd.DataFrame) -> None:
        self.data = data

    def shape(self) -> tuple[int, int]:
        """Return the number of rows and columns."""
        return self.data.shape

    def missing_values(self) -> pd.Series:
        """Return the number of missing values in each column."""
        return self.data.isnull().sum()

    def class_distribution(self, target_columns: list[str]) -> None:
        """Print the class distribution for each target column."""
        for column in target_columns:
            print(f"\n{column}")
            print(self.data[column].value_counts())
            print(self.data[column].value_counts(normalize=True) * 100)

    def duplicates(self) -> int:
        """Return the number of duplicate rows."""

        return self.data.duplicated().sum()

    def data_types(self) -> pd.Series:
        """Return the data type of each column."""

        return self.data.dtypes
