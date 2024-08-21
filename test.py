from model_selection import *

# if __name__ == "__main__":
    # Load your data here
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("C:/Users/prans/Downloads/heart.csv")

# Now pass the DataFrame to the model_select function
model_select(df)
