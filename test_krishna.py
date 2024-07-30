from Data_pre_processing_module import DataCleaning, OutlierManagement, Imputor, DataTransformation
import pandas as pd
# Sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [1, 1, 1, 1, 1],
    'C': [1, None, 3, None, 5],
    'D': ['a', 'b', 'c', 'd', 'e'],
    'E': [10, 12, 13, 1000, 15],
    'F': [None, 'dog', 'cat', 'dog', None]
}
df = pd.DataFrame(data)

# DataCleaning usage
data_cleaning = DataCleaning()
unique_vals = data_cleaning.check_unique(df)
print("Unique values:", unique_vals)

df_cleaned = data_cleaning.drop_redcols(df, unique_vals)
print("DataFrame after dropping redundant columns:\n", df_cleaned)

null_vals = data_cleaning.check_null(df)
print("Null values:", null_vals)

# OutlierManagement usage
outlier_management = OutlierManagement(df)
df_no_outliers_percentile = outlier_management.percentile_way('E')
print("DataFrame after removing outliers using percentile way:\n", df_no_outliers_percentile)

df_no_outliers_IQR = outlier_management.IQR_way('E')
print("DataFrame after removing outliers using IQR way:\n", df_no_outliers_IQR)

# Imputor usage
imputor = Imputor(df)
df_imputed_num = imputor.num_imputor('C')
print("DataFrame after imputing numerical column 'C':\n", df_imputed_num)

df_imputed_cat = imputor.cat_imputor('F')
print("DataFrame after imputing categorical column 'F':\n", df_imputed_cat)

# DataTransformation usage
data_transformation = DataTransformation(df)
df_normalized = data_transformation.normalize_data('E', threshold=500)
print("DataFrame after normalizing column 'E':\n", df_normalized)

df_standardized = data_transformation.standardize_data('E', threshold=500)
print("DataFrame after standardizing column 'E':\n", df_standardized)

df_encoded = data_transformation.encoding_cat()
print("DataFrame after encoding categorical columns:\n", df_encoded)
