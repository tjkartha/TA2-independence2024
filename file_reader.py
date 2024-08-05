import concurrent.futures
from data_loader import DataLoader

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
