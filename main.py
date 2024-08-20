import os
import warnings
warnings.filterwarnings("ignore")

from data_loader import DataHandler
from csv_analysis import CSVAnalysis
from Data_pre_processing_module_v2 import *


def main():
    
    # input----
    handler = DataHandler()
    df = handler.handle_files()
    # df = handler.single_file
    # print ("--------------")
    # print (df)

    # eda-viz----
    csv = CSVAnalysis()
    csv.perform_eda(df)

    # preproc----
    preprocessor = DataPreProcessor(df)
    # print ("~~~~~")
    preprocessed_ = preprocessor.data_pre_processed()
    # print ("-----")
    df = preprocessed_[0]
    # print (df.info())

    # eda-viz----
    plots_storage = "./viz/"
    os.mkdir(plots_storage)
    csv.interactive_plot_selection(df, plots_storage)

if __name__ == "__main__":
    main()
