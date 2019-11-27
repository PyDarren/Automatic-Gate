from FCS import Fcs
from tkinter import filedialog
import pandas as pd
import os
import re




if __name__ == '__main__':
    # 选择路径
    Fpath = filedialog.askdirectory()
    os.makedirs(Fpath+"/WriteFcs/")

    # 读取panel表信息
    panel_file = Fpath + "/panel.xlsx"
    panel_tuple = Fcs.export_panel_tuple(panel_file)
    print(panel_tuple)

    for filename in [filename for filename in os.listdir(Fpath) if os.path.splitext(filename)[1] == ".fcs"]:
        file = Fpath + '/' + filename
        fcs = Fcs(file)

        # 重写marker-name
        pars = fcs.marker_rename(fcs.pars, *panel_tuple)
        # 导出指定通道写出
        # export_channel_tuple = ("Event_length", 115, 140, 144, 145, 148, 149, 153, 155, 158, 160, 163, 165, 170, 174, 191,
        #                         193, 194, 197, 198)
        # pars = fcs.export_channel(pars, *export_channel_tuple)
        # print(len(export_channel_tuple), len(pars))

        new_filename = re.sub('-', '', filename)
        new_filename = re.sub('^.+?_', 'PLT_', new_filename)
        new_filename = re.sub(r'fcs$', 'csv', new_filename)
        # print(new_filename)
        new_file = Fpath + "/WriteFcs/" + new_filename

        fcs.write_to(new_file, pars, to="csv")



    path = Fpath + '/WriteFcs'
    all_df = pd.DataFrame()
    for file in os.listdir(path):
        df = pd.read_csv(path+'/'+file)
        all_df = all_df.append(df)
        print("File %s has finished." % file)
    all_df.to_csv(path + '/' + path.split('/')[-2] + '.csv', index=False)
