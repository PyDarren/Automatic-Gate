# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/27

from FCS import Fcs
from tkinter import filedialog
import pandas as pd
import numpy as np
import os, re, sys, warnings, copy
from marker_ratio_calculation import markerRatioCalculation
from immuneAgeSubsets import subsetsRatioCalculation, subsetsRatioCalculation_real
from confidenceCalculation import confidence_calculation
from immuneAgeCalculation import predict_age
from immuneImpairment import immuneImpairmentMatrix


warnings.filterwarnings('ignore')



def normalization(df, feature):
    '''
    1、Calculated the mean and s.d. of frequency values between the 10th and 90th percentiles;
    2、Normalize cellular frequencies by subtraction of the mean and division by s.d.
    :param feature:  The feature to normalized.
    :return:
    '''
    f_list = list(df[feature])
    f_list = [i for i in f_list if i != 0]              # NA values are not computed
    quantile_10 = np.quantile(f_list, 0.1)
    quantile_90 = np.quantile(f_list, 0.9)
    nums_to_calcu = [i for i in f_list if i >= quantile_10 and i <= quantile_90]
    f_mean = np.mean(nums_to_calcu)
    f_std = np.std(nums_to_calcu)
    df[feature] = df[feature].apply(lambda x : (x - f_mean) / f_std)


def scaling(df):
    for subset in df.columns[:-4]:
        normalization(df, subset)
        print('Cell subset "%s" has finished.' % subset)





if __name__ == '__main__':
    ###################################################
    ####    1.读取singlets导出的FCS文件并转换成CSV    ####
    ###################################################
    # 选择路径
    Fpath = filedialog.askdirectory()
    os.makedirs(Fpath+"/WriteFcs/")
    os.makedirs(Fpath+"/Output/")
    csv_path = Fpath + '/WriteFcs/'
    output_path = Fpath + '/Output/'

    # 读取panel表信息
    panel_file = Fpath + "/panel.xlsx"
    panel_tuple = Fcs.export_panel_tuple(panel_file)
    print(panel_tuple)

    for filename in [filename for filename in os.listdir(Fpath) if os.path.splitext(filename)[1] == ".fcs"]:
        file = Fpath + '/' + filename
        fcs = Fcs(file)
        pars = fcs.marker_rename(fcs.pars, *panel_tuple)
        stain_channel_index = fcs.get_stain_channels(pars)

        # 添加event_length, 191, 193, 194, 140
        add_channel = ["Event_length", "Ir191Di", "Ir193Di", "Pt194Di", "Ce140Di"]
        add_index = [i + 1 for i in range(0, len(pars)) if pars[i].par_short_name in add_channel]
        stain_channel_index.extend(add_index)
        pars = [pars[i] for i in range(0, len(pars)) if i + 1 in stain_channel_index]
        # 根据当前的filename去查找新的name
        new_filename = re.sub("-", "", filename)
        new_filename = re.sub("^.+?_", "", new_filename)
        # new_filename = re.sub("^.+?_", "", new_filename)
        new_file = Fpath + "/WriteFcs/" + new_filename

        # # 重写marker-name
        # pars = fcs.marker_rename(fcs.pars, *panel_tuple)
        # new_filename = re.sub('-', '', filename)
        # new_filename = re.sub('^.+?_', 'PLT_', new_filename)
        # new_filename = re.sub(r'fcs$', 'csv', new_filename)
        # new_file = Fpath + "/WriteFcs/" + new_filename
        fcs.write_to(new_file, pars, to="csv")
    print('所有样本FCS文件均已转换为CSV文件', '\n','-'*100,'\n')


    ###################################################
    ####           2.计算        ####
    ###################################################
    file_list = os.listdir(csv_path)
    label_frequency_all = pd.DataFrame()
    ratio_all = pd.DataFrame()
    real_all = pd.DataFrame()
    confidence_all = pd.DataFrame()
    immune_age_all = pd.DataFrame()
    ratio34_all = pd.DataFrame()
    impair_all = pd.DataFrame()

    for info in file_list:
        sample_id = info[:9]
        os.makedirs(output_path + '/%s/'%info[:9])
        sample_path = output_path + '/%s/'%info[:9]
        sample_df = pd.read_csv(csv_path+info)

        sample_df.columns = ['length', 'CD57', 'CD3', 'CD68', 'beads',
                            'CD56', 'gdTCR', 'CCR6', 'CD14 ', 'IgD', 'CD123(IL-3R)',
                            'CD85J', 'CD19', 'CD25', 'CD274(PD-L1)', 'CD278(ICOS)',
                            'CD39', 'CD27', 'CD24', 'CD45RA', 'CD86', 'CD28',
                            'CD197(CCR7)', 'CD11c ', 'CD33', 'CD152(CTLA-4)', 'FoxP3',
                            'CD161', 'CXCR5', 'CD66b', 'CD183(CXCR3)', 'CD94', 'T-bet',
                            'Ki-67', 'CD127(IL-7Ra)', 'CD279(PD-1)', 'CD38', 'Granzyme B',
                            'CD20', 'CD16', 'HLA-DR', 'DNA1', 'DNA2', 'cisplatin',
                            'CD4', 'CD8a', 'CD11b']

        # 1. 计算各个marker的标签矩阵
        label_df = markerRatioCalculation(sample_df)
        label_frequency = np.sum(label_df)/label_df.shape[0]
        label_frequency_df = pd.DataFrame(label_frequency).T
        label_frequency_df.index = [sample_id]
        label_frequency_all = label_frequency_all.append(label_frequency_df)
        print('Marker ratio calculation has finished!', '\n')

        # 2. 计算特定亚群的比率
        ratio_merge_df = subsetsRatioCalculation(label_df)
        ratio_df = ratio_merge_df.T
        ratio_df['subset'] = list(ratio_df.index)
        ratio_df.columns = ['frequency', 'subset']
        ratio_df.index = [i for i in range(ratio_df.shape[0])]
        ratio_df = ratio_df[['subset', 'frequency']]
        ratio_df['frequency'] = ratio_df['frequency'].apply(lambda x: x*100)
        print('Subset ratio calculation has finished!', '\n')
        ratio_df.to_excel(sample_path+'subset_ratio.xlsx', index=False)
        ratio_merge_df.index = [sample_id]
        ratio_all = ratio_all.append(ratio_merge_df)
        ##############################################
        real_merge_df = subsetsRatioCalculation_real(label_df)
        real_df = real_merge_df.T
        real_df['subset'] = list(real_df.index)
        real_df.columns = ['frequency', 'subset']
        real_df.index = [i for i in range(real_df.shape[0])]
        real_df = real_df[['subset', 'frequency']]
        real_df['frequency'] = real_df['frequency'].apply(lambda x: x*100)
        real_merge_df.index = [sample_id]
        real_all = real_all.append(real_merge_df)


        # 3. 计算置信区间相对值
        confidence_df = confidence_calculation(real_df)
        confidence_df.to_excel(sample_path+'confidence.xlsx', index=False)
        print('Confidence calculation has finished!', '\n')
        confidence_merge_df = confidence_df.T
        confidence_merge_df.columns = list(confidence_df['subset'].values)
        confidence_merge_df = confidence_merge_df.iloc[1:, :]
        confidence_merge_df.index = [sample_id]
        confidence_all = confidence_all.append(confidence_merge_df)

        # 4. 计算免疫年龄
        age_df, ratio34_df = predict_age(ratio_df)
        frequency_list = [info, 'x']
        frequency_list.extend(list(ratio34_df['frequency'].values))
        col_names = ['Patient.ID', 'age', 'B cells', 'CD161+CD4+ T cells', 'CD161+CD8+ T cells', 'CD161-CD45RA+ Tregs', 'CD161-CD45RA- Tregs', 'CD20- CD3- lymphocytes', 'CD28+CD8+ T cells', 'CD4+ T cells', 'CD4+CD28+ T cells', 'CD8+ T cells', 'CD85j+CD8+ T cells', 'IgD+CD27+ B cells', 'IgD+CD27- B cells', 'IgD-CD27+ B cells', 'IgD-CD27- B cells', 'NK cells', 'NKT cells', 'T cells', 'Tregs', 'central memory CD4+ T cells', 'central memory CD8+ T cells', 'effector CD4+ T cells', 'effector CD8+ T cells', 'effector memory CD4+ T cells', 'effector memory CD8+ T cells', 'gamma-delta T cells', 'lymphocytes', 'memory B cells', 'monocytes', 'naive B cells', 'naive CD4+ T cells', 'naive CD8+ T cells', 'transitional B cells', 'viable/singlets']
        ratio34_df = pd.DataFrame(frequency_list).T
        ratio34_df.columns = col_names
        age_df.index = [sample_id]
        age_df.columns = ['immune age']
        age_df.to_excel(sample_path+'immune_age.xlsx', index=False)
        ratio34_df.to_excel(sample_path+'subset34_ratio.xlsx', index=False)
        print('Immune age calculation has finished!', '\n')
        immune_age_all = immune_age_all.append(age_df)
        ratio34_all = ratio34_all.append(ratio34_df)

        # 5. 提取免疫损伤矩阵
        impair_df = immuneImpairmentMatrix(ratio_df, sample_id)
        print('Immune impairment matrix has finished!', '\n')
        impair_all = impair_all.append(impair_df)


    ratio_all.to_excel(output_path+'ratio_all.xlsx')
    real_all.to_excel(output_path+'real_all.xlsx')
    confidence_all.to_excel(output_path+'confidence_all.xlsx')
    immune_age_all.to_excel(output_path+'immune_age_all.xlsx')
    ratio34_all.to_excel(output_path+'ratio34_all.xlsx', index=False)
    impair_all.to_excel(output_path+'impairment_all.xlsx', index=False)
    label_frequency_all.to_excel(output_path+'label_frequency.xlsx')

    # 提取置信区间66亚群比率
    select_subsets_df = pd.read_excel('C:/Users/pc/OneDrive/PLTTECH/Project/00_immune_age_project/Rawdata/置信区间选择.xlsx')
    select_subsets = list(select_subsets_df['subset'].values)
    confidence_66_ratio = real_all.loc[:, select_subsets].T
    confidence_66_ratio = confidence_66_ratio.multiply(100)
    confidence_66_ratio.to_excel(output_path+'confidence_66_ratio.xlsx')


    ###################################################
    ####           3. 免疫损伤预处理                ####
    ###################################################
    os.makedirs(output_path+'immune_impairment/')
    os.makedirs(output_path+'immune_impairment/per_sample_data/')
    os.makedirs(output_path+'immune_impairment/stage2_data/')
    os.makedirs(output_path+'immune_impairment/final_score/')

    raw_df = pd.read_csv('C:/Users/pc/OneDrive/PLTTECH/Project/00_immune_age_project/Rawdata/diffusion_original.csv')
    cell_subsets = ['B.cells', 'CD161negCD45RApos.Tregs', 'CD161pos.NK.cells', 'CD28negCD8pos.T.cells',
                    'CD57posCD8pos.T.cells', 'CD57pos.NK.cells', 'effector.CD8pos.T.cells',
                    'effector.memory.CD4pos.T.cells', 'effector.memory.CD8pos.T.cells',
                    'HLADRnegCD38posCD4pos.T.cells', 'naive.CD4pos.T.cells', 'naive.CD8pos.T.cells',
                    'PD1posCD8pos.T.cells', 'T.cells', 'CXCR5+CD4pos.T.cells', 'CXCR5+CD8pos.T.cells',
                    'Th17 CXCR5-CD4pos.T.cells', 'Tregs']
    cell_subsets.extend(['subject id', 'year', 'visit number', 'age'])
    raw_df = raw_df.loc[:, cell_subsets]

    for i in range(impair_all.shape[0]):
        sample_id = impair_all.iloc[i, :]['subject id']
        sample_df = raw_df.append(impair_all.iloc[i, :])
        sample_df = sample_df.fillna(0)
        scaling(sample_df)
        year_list = [2012, 2013, 2014, 2015, 2019]
        sample_df = sample_df[sample_df['year'].isin(year_list)]
        sample_df.to_csv(output_path+'immune_impairment/per_sample_data/%s.csv' % sample_id,
                         index=False)

