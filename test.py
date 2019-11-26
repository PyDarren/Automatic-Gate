# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/8


import pandas as pd
import numpy as np


if __name__ == "__main__":
    df_more = pd.read_csv('E:/cd/Automatic_Gate_Data/Rawdata/marker_42/Singlets.csv')
    df_less = pd.read_csv('E:/cd/Automatic_Gate_Data/Rawdata/marker_42/Viable.csv')

    df = df_more.append(df_less)
    df = df.append(df_less)
    df_no_less = df.drop_duplicates(subset=list(df.columns), keep=False)

    df_no_less.to_csv('E:/cd/Automatic_Gate_Data/Rawdata/marker_42/noViable.csv', index=False)
