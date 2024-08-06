import pandas as pd
import json
import xml.etree.ElementTree as ET
from PIL import Image
import cv2

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
