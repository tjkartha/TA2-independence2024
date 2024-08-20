import pandas as pd
import json
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import concurrent.futures
import os

class DataHandler:
    @staticmethod
    def load_data(file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        try:
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension in ['.xls', '.xlsx']:
                return pd.read_excel(file_path)
            elif file_extension == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif file_extension == '.xml':
                tree = ET.parse(file_path)
                return tree.getroot()
            elif file_extension in ['.h5', '.hdf5']:
                return pd.read_hdf(file_path)
            elif file_extension == '.feather':
                return pd.read_feather(file_path)
            elif file_extension == '.txt':
                with open(file_path, 'r') as f:
                    return f.read()
            elif file_extension in ['.jpg', '.jpeg']:
                return Image.open(file_path)
            elif file_extension in ['.mp4', '.avi', '.mkv']:
                return cv2.VideoCapture(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    @staticmethod
    def read_multiple_files(file_paths):
        loaded_data = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(DataHandler.load_data, file_path): file_path for file_path in file_paths}
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    data = future.result()
                    if data is not None:
                        loaded_data.append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        return loaded_data

    @staticmethod
    def merge_files_side_by_side(file_paths):
        dfs = DataHandler.read_multiple_files(file_paths)
        dfs = [df for df in dfs if isinstance(df, pd.DataFrame)]
        if len(dfs) == 0:
            return None
        elif len(dfs) == 1:
            return dfs[0]
        merged_df = pd.concat(dfs, axis=1)
        return merged_df

    @staticmethod
    def save_to_csv(dataframe, file_path):
        if isinstance(dataframe, pd.DataFrame):
            try:
                dataframe.to_csv(file_path, index=False)
                print(f"File saved to {file_path}")
            except Exception as e:
                print(f"Error saving file {file_path}: {e}")
        else:
            print("Provided data is not a DataFrame.")

    @staticmethod
    def get_file_paths_from_input():
        file_paths = input("Enter file paths separated by commas: ").split(',')
        file_paths = [file_path.strip() for file_path in file_paths if file_path.strip()]
        return file_paths

    @staticmethod
    def get_output_file_path():
        return input("Enter the output file path (e.g., output.csv or output.xlsx): ")

    def handle_files(self):
        file_paths = self.get_file_paths_from_input()
        if not file_paths:
            print("No files selected.")
            return

        if len(file_paths) > 1:
            if_yes = input("Do you want to merge the files side-by-side? (yes/no): ")
            if if_yes.lower() == "yes":
                merged_df = self.merge_files_side_by_side(file_paths)
                if merged_df is not None:
                    output_file_path = self.get_output_file_path()
                    if output_file_path.endswith('.csv'):
                        self.save_to_csv(merged_df, output_file_path)
                    elif output_file_path.endswith('.xlsx'):
                        merged_df.to_excel(output_file_path, index=False)
                        print(f"Merged file saved to {output_file_path}")
                    else:
                        print("Unsupported output file format. Please use .csv or .xlsx.")
                    # Code to automatically download the file (if running in a web environment)
                    try:
                        from IPython.display import FileLink
                        display(FileLink(output_file_path))
                    except ImportError:
                        pass
                else:
                    print("No DataFrames to merge.")
            elif if_yes.lower() == "no":
                loaded_files = self.read_multiple_files(file_paths)
                for idx, file_data in enumerate(loaded_files):
                    variable_name = f"file_{idx + 1}"
                    globals()[variable_name] = file_data
                    print(f"{variable_name}:\n{file_data}\n")
        else:
            single_file = self.load_data(file_paths[0])
            print(single_file)
            if isinstance(single_file, pd.DataFrame):
                print(single_file.head())
