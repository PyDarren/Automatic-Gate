from FCS import Fcs
from FCS import StainFcs
from FCS import Write
from tkinter import filedialog
import pandas as pd
import os, random



def renameFcsFile(Fpath):
    for file in [file for file in os.listdir(Fpath) if os.path.splitext(file)[1] == ".fcs"]:
        filename = Fpath + '/' + file
        fcs = Fcs(filename)
    Fcs.panel_rename()



def fcsToCSV(path):
    Fpath = path + '/Marker_rename'
    for file in [file for file in os.listdir(Fpath) if os.path.splitext(file)[1] == ".fcs"]:
        filename = Fpath + '/' + file
        stain_fcs = StainFcs(filename)
        stain_fcs.fcs2excel(folder_name="Stain_excel")



def csvFileMerge(Fpath):
    path = Fpath + '/Marker_rename/Stain_excel'
    all_df = pd.DataFrame()
    for file in os.listdir(path):
        df = pd.read_csv(path+'/'+file)
        all_df = all_df.append(df)
        print("File %s has finished." % file)
    all_df.to_csv(path + '/' + path.split('/')[-3] + '.csv')


if __name__ == "__main__":
    # 选择路径
    Fpath = filedialog.askdirectory()

    renameFcsFile(Fpath)
    print('\n' + 'Rename Procession has Finished.' + '\n')
    fcsToCSV(Fpath)
    print('\n' + 'FCS to CSV Procession has Finished.' + '\n')
    csvFileMerge(Fpath)
    print('\n' + 'CSV Merge Procession has Finished.' + '\n')
    print('All Procession has Finished.')





