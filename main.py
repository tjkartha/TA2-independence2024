import os
import warnings
warnings.filterwarnings("ignore")

from data_loader import DataHandler
from csv_analysis import CSVAnalysis
from Data_pre_processing_module_v2 import *
from model_selection import *
from post_model_viz import *

def main():
    
    # input----
    print("\n--------------- input --------------------------")
    handler = DataHandler()
    df = handler.handle_files()

    # eda-viz----
    print("\n--------------- eda-viz ------------------------")
    csv = CSVAnalysis()
    csv.perform_eda(df)

    # preproc----
    print("\n--------------- preproc------------------------")
    preprocessor = DataPreProcessor(df)
    preprocessed_ = preprocessor.data_pre_processed()
    df = preprocessed_[0]

    # eda-viz----
    print("\n--------------- eda-viz ------------------------")
    plots_storage = "./viz/"
    if(os.path.isdir('new_folder')==False)
        os.mkdir(plots_storage)
        csv.interactive_plot_selection(df, plots_storage)
    else:
        csv.interactive_plot_selection(df, plots_storage)

    # preproc----
    print("\n--------------- preproc ------------------------")
    print("**********************")
    print("NaN values: \n", df.isnull().sum())
    print("**********************\n")
    transform_ = DataTransformation(df)
    df = transform_.encoding_cat(cat_cols=preprocessed_[1])
    print("\nPrinting the columns:")
    print(df.info())
    print("\n------------\n")
    print("NaN values: \n", df.isnull().sum())

    # model-eval----
    print("\n--------------- model-eval ---------------------")
    model_select(df)

    # eda-viz----
    print("\n--------------- eda-viz ------------------------")
    pmv_ = PostModelViz()
    interactive_plot_selection(plots_storage, results_dict)

if __name__ == "__main__":
    main()
