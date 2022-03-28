# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:34:10 2021

@author: Administrator
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel(r'.\main_variable20.xlsx')
df = data.iloc[:, :] #取前20列数据
name = df.columns.values

result2 = np.corrcoef(df, rowvar=False)
print(result2.shape)

new = pd.DataFrame(columns=['l_id','c_id','cor','description'])
for l_id,item in enumerate(result2):
    count = 0
    for c_id,cor in enumerate(item):
        if 0.99 > cor >= 0.6 :
            if c_id in new['l_id']: continue
            new = new.append({'l_id': l_id,'c_id':c_id,'cor':cor,'description':(name[l_id],name[c_id])}, ignore_index=True)
print(new)

#pd.plotting.scatter_matrix(df, figsize=(12,12),range_padding=0.5)
#figure, ax = plt.subplots(figsize=(12, 12))

#sns.heatmap(df.corr(), square=True, annot=True, ax=ax)