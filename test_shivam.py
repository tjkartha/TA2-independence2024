import pandas as pd
from Data_pre_processing_module import DataCleaning,OutlierManagement,Imputor,DataTransformation

def main():
    # Sample data
    data = {
        'Category1': ['A', 'B', 'A', 'C', 'B', 'C'],
        'Category2': ['X', 'Y', 'X', 'Z', 'Y', 'Z'],
        'Numerical1': [1, 2, 3, 4, 5, 6],
        'Numerical2': [7, 8, 9, 10, 11, 12],
        'Numerical3': [1.5, 2.5, 3.5, None, 4.5, 5.5]
    }
    df = pd.DataFrame(data)

    # Testing DataCleaning class
    data_cleaning = DataCleaning()
    unique_vals = data_cleaning.check_unique(df)
    print("Unique Values:", unique_vals)
    
    df_cleaned = data_cleaning.drop_redcols(df, unique_vals)
    print("DataFrame after dropping columns with only one unique value:")
    print(df_cleaned)
    
    null_vals = data_cleaning.check_null(df)
    print("Null Values:", null_vals)

    # Testing OutlierManagement class
    outlier_management = OutlierManagement(df)
    df_no_outliers_percentile = outlier_management.percentile_way('Numerical1')
    print("DataFrame after removing outliers using 99th percentile method:")
    print(df_no_outliers_percentile)

    df_no_outliers_iqr = outlier_management.IQR_way('Numerical1')
    print("DataFrame after removing outliers using IQR method:")
    print(df_no_outliers_iqr)

    # Testing Imputor class
    imputor = Imputor(df)
    df_imputed_num = imputor.num_imputor('Numerical3')
    print("DataFrame after imputing numerical column:")
    print(df_imputed_num)

    df_imputed_cat = imputor.cat_imputor('Category2')
    print("DataFrame after imputing categorical column:")
    print(df_imputed_cat)

    # Testing DataTransformation class
    data_transformation = DataTransformation(df)
    df_normalized = data_transformation.normalize_data('Numerical1', 4)
    print("DataFrame after normalizing data:")
    print(df_normalized)

    df_standardized = data_transformation.standardize_data('Numerical1', 4)
    print("DataFrame after standardizing data:")
    print(df_standardized)

    df_encoded = data_transformation.encoding_cat()
    print("DataFrame after encoding categorical columns:")
    print(df_encoded)

if __name__ == "__main__":
    main()
