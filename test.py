from main_1 import main
from model_build import load_data

data_path = "C:/Users/prans/OneDrive/Documents/TA_2/penguins (1).csv"
data = load_data(data_path)

# Run the main function from main_1.py with the loaded data
main(data)
    

# from main_1 import main

# # Define the model_eval function
# def model_eval(df):
#     # Run the main function from main_1.py with the preprocessed DataFrame
#     main(df)

# # Example usage:
# # Assuming the preprocessing script or team provides the DataFrame 'df',
# # you would directly call model_eval(df) with the DataFrame.

# # For demonstration, let's assume df is loaded directly here.
# # In practice, df would be passed to this script by the preprocessing team or another component.
# if _name_ == "_main_":
#     import pandas as pd

#     # Load the preprocessed DataFrame directly from the file or variable
#     data_path = "./processed_data.csv"
#     df = pd.read_csv(data_path)

#     # Call model_eval with the preprocessed DataFrame
#     model_eval(df)