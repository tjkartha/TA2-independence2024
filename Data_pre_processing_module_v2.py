import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class DataCleaning:
    def __init__(self):
        pass

    def check_unique(self, df: pd.DataFrame) -> dict:
        """
        Takes in a dataframe and returns the number of unique values in each column as a dictionary.
        """
        unique_values = {col: df[col].nunique() for col in df.columns}
        return unique_values

    def drop_redcols(self, df: pd.DataFrame, unique_vals: dict) -> pd.DataFrame:
        """
        Takes the dataframe and the dictionary returned by check_unique and drops columns with only one unique value.
        Returns the resultant dataframe.
        """
        columns_to_drop = [col for col, unique_count in unique_vals.items() if unique_count == 1]
        return df.drop(columns=columns_to_drop)

    def check_null(self, df: pd.DataFrame) -> dict:
        """
        Takes in a dataframe and returns the number of null values in each column as a dictionary.
        """
        null_values = {col: df[col].isnull().sum() for col in df.columns}
        return null_values
 


class OutlierManagement:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def percentile_way(self, column: str) -> pd.DataFrame:
        """
        Takes a dataframe and column name as arguments and removes all the rows in which the given column value 
        exceeds the 99th percentile of the column's values, ignoring null values.
        Returns the resultant dataframe.
        """
        threshold = self.df[column].dropna().quantile(0.99)
        result_df = self.df[self.df[column] <= threshold]
        return result_df

    def IQR_way(self, column: str) -> pd.DataFrame:
        """
        Takes a dataframe and column name as arguments and removes all the rows in which the given column value 
        exceeds the interquartile range (IQR).
        Returns the resultant dataframe.
        """
        Q1 = self.df[column].dropna().quantile(0.25)
        Q3 = self.df[column].dropna().quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        result_df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return result_df
    

class Imputor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def num_imputor(self, column: str) -> pd.DataFrame:
        """
        Takes in the dataframe and a numerical column as arguments and imputes all the null values in the column
        with the median of the column's values. Returns the resultant dataframe.
        """
        median_value = self.df[column].median()
        self.df[column].fillna(median_value, inplace=True)
        return self.df

    def cat_imputor(self, column: str) -> pd.DataFrame:
        """
        Takes in the dataframe and a categorical column as arguments and imputes all the null values in the column
        with the mode of the column's values. Returns the resultant dataframe.
        """
        mode_value = self.df[column].mode()[0]
        self.df[column].fillna(mode_value, inplace=True)
        return self.df


class DataTransformation:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def normalize_data(self, column: str, threshold: float) -> pd.DataFrame:
        """
        Takes a threshold value, the dataframe, and a column name. If the data distribution of the column 
        (max value - min value) is greater than the threshold value, it normalizes the data 
        (value-min)/(max-min) and returns the resultant dataframe.
        """
        col_min = self.df[column].min()
        col_max = self.df[column].max()
        if col_max - col_min > threshold:
            self.df[column] = (self.df[column] - col_min) / (col_max - col_min)
        return self.df

    def standardize_data(self, column: str, threshold: float) -> pd.DataFrame:
        """
        Takes a threshold value, the dataframe, and a column name. If the data distribution of the column 
        (max value - min value) is greater than the threshold value, it standardizes the data using mean 
        and standard deviation, and returns the resultant dataframe.
        """
        col_min = self.df[column].min()
        col_max = self.df[column].max()
        if col_max - col_min > threshold:
            col_mean = self.df[column].mean()
            col_std = self.df[column].std()
            self.df[column] = (self.df[column] - col_mean) / col_std
        return self.df

    def encoding_cat(self, cat_cols):
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = encoder.fit_transform(self.df[cat_cols])

        # Convert encoded data to DataFrame with appropriate column names
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols), index=self.df.index)

        # Drop original categorical columns
        self.df.drop(cat_cols, axis=1, inplace=True)

        # Concatenate encoded columns back to the original DataFrame
        self.df = pd.concat([self.df, encoded_df], axis=1)

        return self.df
    

class DataPreProcessor:
    def __init__(self, df: pd.DataFrame, threshold: int = 15):
        self.df = df
        self.threshold = threshold
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

    def data_pre_processed(self):
        # Step 1: Get unique values
        data_cleaning = DataCleaning()
        unique_list = data_cleaning.check_unique(self.df)

        # Step 2: Move numerical columns with unique values < threshold to categorical columns
        for col in self.numerical_cols[:]:
            if unique_list[col] < self.threshold:
                self.numerical_cols.remove(col)
                self.categorical_cols.append(col)

        # Step 3: Remove redundant columns
        self.df = data_cleaning.drop_redcols(self.df, unique_list)

        # Step 4: Check and display null values
        null_vals = data_cleaning.check_null(self.df)
        print("Null values:", null_vals)

        # Step 5: Handle outliers
        outlier_management = OutlierManagement(self.df)
        for col in self.numerical_cols:
            self.df = outlier_management.IQR_way(col)

        # Step 6: Impute missing values
        imputor = Imputor(self.df)
        for col in self.numerical_cols:
            self.df = imputor.num_imputor(col)
        for col in self.categorical_cols:
            self.df = imputor.cat_imputor(col)

        # Step 7: Normalize numerical columns
        data_transformation = DataTransformation(self.df)
        for col in self.numerical_cols:
            self.df = data_transformation.normalize_data(col, threshold=500)

        return (self.df,self.categorical_cols)
