import pandas as pd
import numpy as np
from time import time

def data_preparation(column_name):
    # get raw data
    ADMET_path = r'./ADMET.xlsx'
    descriptior_path = r'./Molecular_Descriptor.xlsx'

    ADMET = pd.read_excel(ADMET_path, sheet_name=None)
    descriptior = pd.read_excel(descriptior_path, sheet_name=None)

    ADMET = ADMET['training']
    descriptior = descriptior['training']

    target_c = ADMET.loc[:, column_name]
    factors = descriptior.loc[:, 'nAcid':]

    return factors.values, target_c.values

def data_preparation_5c(column_name):
    # get raw data
    ADMET = pd.read_excel(r'./ADMET.xlsx', sheet_name=None)
    main_var20 = pd.read_excel(r'./main_variable20.xlsx', index_col=False, header=0)

    ADMET = ADMET['training']

    target_c = ADMET.loc[:, column_name]

    return main_var20.values,target_c.values

def get_mean_std():
    new = pd.DataFrame(columns=['item','mean','mean0.7','std','std0.7'])
    main_var20 = pd.read_excel(r'./main_variable20.xlsx', index_col=False, header=0)
    a = main_var20.sample(1380)

    print(main_var20.shape)
    for name in main_var20.columns.values:
        new = new.append({'item':name,
                          'mean':np.nanmean(main_var20[name].values),
                          'std':np.nanstd(main_var20[name].values),
                          'mean0.7': np.nanmean(a[name].values),
                          'std0.7': np.nanstd(a[name].values),
                          },ignore_index=True)
    print(new)
    new.to_excel(r'./mean_std.xlsx', index=False, header=True)


if __name__ == '__main__':
    #factors, target_c = data_preparation_5c('Caco-2')
    #print(factors.shape,target_c,target_c)
    get_mean_std()