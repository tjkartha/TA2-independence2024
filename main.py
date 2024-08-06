import pandas as pd
from data_loader import DataLoader, FileReader, DataMerger, UserInputHandler

def main():
    file_paths = UserInputHandler.get_file_paths_from_input()
    print(type(file_paths))

    if len(file_paths) > 1:
        if_yes = input("Do you want to merge the files side-by-side? (yes/no): ")
        if if_yes.lower() == "yes":
            merged_df = DataMerger.merge_files_side_by_side(file_paths)
            if merged_df is not None:
                print(merged_df.head())
            else:
                print("No DataFrames to merge.")
        elif if_yes.lower() == "no":
            loaded_files = FileReader.read_multiple_files(file_paths)
            for idx, file_data in enumerate(loaded_files):
                variable_name = f"file_{idx + 1}"
                globals()[variable_name] = file_data
                print(f"{variable_name}:\n{file_data}\n")
    else:
        single_file = DataLoader.load_data(file_paths[0])
        print(single_file)
        if isinstance(single_file, pd.DataFrame):
            print(single_file.head())

if __name__ == "__main__":
    main()
