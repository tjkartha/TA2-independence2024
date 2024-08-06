import pandas as pd
import json
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import concurrent.futures

# DataLoader class definition
class DataLoader:
    @staticmethod
    def load_data(file_path):
        file_extension = file_path.split('.')[-1].lower()
        try:
            if file_extension == 'csv':
                return pd.read_csv(file_path)
            elif file_extension in ['xls', 'xlsx']:
                return pd.read_excel(file_path)
            elif file_extension == 'json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif file_extension == 'xml':
                tree = ET.parse(file_path)
                return tree.getroot()
            elif file_extension in ['h5', 'hdf5']:
                return pd.read_hdf(file_path)
            elif file_extension == 'feather':
                return pd.read_feather(file_path)
            elif file_extension == 'txt':
                with open(file_path, 'r') as f:
                    return f.read()
            elif file_extension in ['jpg', 'jpeg']:
                return Image.open(file_path)
            elif file_extension in ['mp4', 'avi', 'mkv']:
                return cv2.VideoCapture(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

# FileReader class definition
class FileReader:
    @staticmethod
    def read_multiple_files(file_paths):
        loaded_data = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(DataLoader.load_data, file_path): file_path for file_path in file_paths}
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    data = future.result()
                    if data is not None:
                        loaded_data.append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        return loaded_data

# DataMerger class definition
class DataMerger:
    @staticmethod
    def merge_files_side_by_side(file_paths):
        dfs = FileReader.read_multiple_files(file_paths)
        dfs = [df for df in dfs if isinstance(df, pd.DataFrame)]
        if len(dfs) == 0:
            return None
        elif len(dfs) == 1:
            return dfs[0]
        merged_df = pd.concat(dfs, axis=1)
        return merged_df

# UserInputHandler class definition
class UserInputHandler:
    @staticmethod
    def get_file_paths_from_input():
        file_paths = input("Enter file paths separated by commas: ").split(',')
        file_paths = [file_path.strip() for file_path in file_paths]
        return file_paths
