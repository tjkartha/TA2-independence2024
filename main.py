from data_loader import DataHandler
from csv_analysis import CSVAnalysis

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
    
    

if __name__ == "__main__":
    main()
