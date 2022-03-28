import pandas as pd
import numpy as np
from ML_classifier import run_one

def pso_data_preaparation():
     factors = pd.read_excel(r'./main_variable20.xlsx',index_col=False,header=0)
     min_list = []
     max_list = []
     for col in factors.columns.values:
          tmp = factors[col].values
          min_list.append(np.min(tmp))
          max_list.append(np.max(tmp))
     return min_list,max_list,factors.columns.values

#min_list,max_list = pso_data_preaparation()

def get_five_classifier():
    c_list = ['Caco-2','CYP3A4','hERG','HOB','MN']
    c_model = {}
    c_scaler = {}
    for task_name in c_list:
        model, scaler =run_one(task_name=task_name, net='rfc')
        c_model[task_name] = model
        c_scaler[task_name] = scaler
    return c_model,c_scaler


