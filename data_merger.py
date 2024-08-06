import pandas as pd
from file_reader import FileReader

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
