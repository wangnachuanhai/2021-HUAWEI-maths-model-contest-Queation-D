import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def initial_processing():
    #get raw data
    ER_alpha_path = r'./ERα_activity.xlsx'
    descriptior_path = r'./Molecular_Descriptor.xlsx'
    ER_alpha = pd.read_excel(ER_alpha_path, sheet_name=None)
    descriptior = pd.read_excel(descriptior_path, sheet_name=None)
    ER_alpha = ER_alpha['training']
    descriptior = descriptior['training']
    return ER_alpha,descriptior

def get_imf_rank(ER_alpha,descriptior):

    target_c = ER_alpha.loc[:,'pIC50']
    factors = descriptior.loc[:,'nAcid':]
    print(target_c.shape,factors.shape)

    factors_name = factors.columns.values

    x = factors.values
    y = target_c.values

    clf = RandomForestRegressor(n_estimators=300, random_state=(0), max_features=729, n_jobs=(2))
    clf.fit(x, y)

    #统计20个主要变量
    importances = clf.feature_importances_
    imf_list = []
    for index,imf in enumerate(importances):
        imf_list.append([factors_name[index],imf])

    df_tmp = pd.DataFrame(imf_list, columns=['features','importance'], dtype=float)
    df_tmp = df_tmp.sort_values(by="importance", inplace=False, ascending=False)

    print(df_tmp[:20])

    main_variable = factors.loc[:, df_tmp['features'][:20] ]
    #main_variable.to_excel(r'./main_variable20.xlsx', index=False, header=True)
    #target_c.to_excel(r'./target.xlsx', index=False, header=True)

if __name__ == '__main__':
    ER_alpha,descriptior = initial_processing()
    get_imf_rank(ER_alpha, descriptior)
