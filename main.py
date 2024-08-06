from data_loader import DataLoader, FileReader, DataMerger, UserInputHandler
import pandas as pd
def main():
    file_paths = UserInputHandler.get_file_paths_from_input()
    if not file_paths:
        print("No files selected.")
        return

    if len(file_paths) > 1:
        if_yes = input("Do you want to merge the files side-by-side? (yes/no): ")
        if if_yes.lower() == "yes":
            merged_df = DataMerger.merge_files_side_by_side(file_paths)
            if merged_df is not None:
                output_file_path = UserInputHandler.get_output_file_path()
                if output_file_path.endswith('.csv'):
                    DataMerger.save_to_csv(merged_df, output_file_path)
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

