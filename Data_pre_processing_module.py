import pandas as pd
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

    def encoding_cat(self,cat_cols) -> pd.DataFrame:
        """
        Takes the dataframe and hot encodes all the categorical columns using OneHotEncoder from sklearn,
        returning the resultant dataframe.
        """
        # Select categorical columns
        #categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_cols = cat_cols
        # Create an instance of OneHotEncoder
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        
        # Apply OneHotEncoder to categorical columns
        encoded_data = encoder.fit_transform(self.df[categorical_cols])
        
        # Create a DataFrame with the encoded data
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
        
        # Drop the original categorical columns from the dataframe
        self.df = self.df.drop(categorical_cols, axis=1)
        
        # Concatenate the original dataframe with the encoded dataframe
        self.df = pd.concat([self.df, encoded_df], axis=1)
        return self.df
    


