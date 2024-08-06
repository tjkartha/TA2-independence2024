import pandas as pd
import numpy as np
from Data_pre_processing_module import *

# loading the data frame
df = pd.read_csv("./Medical_insurance.csv")
print(df.head())

# threshold value to term the column as categorical
threshold = 15

# getting the unique values
test = DataCleaning()
unique_list = test.check_unique(df)
print(unique_list)

# Separate numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Separate categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# Process to move numerical columns with unique values less than threshold to categorical columns
for col in numerical_cols[:]:                # Iterate over a shallow copy of the list to avoid modifying it while iterating
    if unique_list[col] < threshold:
        numerical_cols.remove(col)
        categorical_cols.append(col)

print("Updated numerical columns:", numerical_cols)
print("Updated categorical columns:", categorical_cols)

# removing reduntant columns
df = test.drop_redcols(df, unique_list)
print("DataFrame after dropping redundant columns:\n", df)

# checkiing null values
null_vals = test.check_null(df)
print("Null values:", null_vals)

# OutlierManagement usage
# Initialize OutlierManagement with the DataFrame
outlier_management = OutlierManagement(df)

# Loop through numerical columns and handle outliers using the specified method
for col in numerical_cols:
    df = outlier_management.percentile_way(col)

# Imputor usage
imputor = Imputor(df)

# imputing numerical columns
for col in numerical_cols:
    df = imputor.num_imputor(col)

# imputing categorical columns
for col in categorical_cols:
    df = imputor.cat_imputor(col)

print(df.head())

# DataTransformation usage
data_transformation = DataTransformation(df)
# normalizing data
for col in numerical_cols:
    df = data_transformation.normalize_data(col, threshold=500)

df = data_transformation.encoding_cat(cat_cols=categorical_cols)

print(df.head())