import os
import warnings
warnings.filterwarnings("ignore")

from data_loader import DataHandler
from csv_analysis import CSVAnalysis
from Data_pre_processing_module_v2 import *
from test import *

def main():
    
    # input----
    handler = DataHandler()
    df = handler.handle_files()

    # eda-viz----
    csv = CSVAnalysis()
    csv.perform_eda(df)

    # preproc----
    preprocessor = DataPreProcessor(df)
    preprocessed_ = preprocessor.data_pre_processed()
    df = preprocessed_[0]

    # eda-viz----
    plots_storage = "./viz/"
    # if(os.path.isdir('new_folder')==False)
    # os.mkdir(plots_storage)
    # csv.interactive_plot_selection(df, plots_storage)

    # preproc----
    transform_ = DataTransformation(df)
    df = transform_.encoding_cat(cat_cols=preprocessed_[1])
    # print(df)

    # model-eval----
    model_eval(df)

if __name__ == "__main__":
    main()
