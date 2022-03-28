from ML_regressor_8 import run_one
from pso_prepare import pso_data_preaparation,get_five_classifier
from sko.PSO import PSO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#regressing
model,scaler = run_one()
min_list,max_list,var_list= pso_data_preaparation()

#normalize range
min_list = scaler.transform(np.array(min_list).reshape(1,-1))
max_list = scaler.transform(np.array(max_list).reshape(1,-1))

#classify preparation
c_model,c_scaler = get_five_classifier()

def judge(x,c_model):
    c_list = ['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
    predict = []
    answer = np.array([1,1,0,1,0])
    for model_name in c_list:
        model = c_model[model_name]
        result = model.predict(x)
        predict.append(result[0])
    predict = np.array(predict)
    #print(predict)
    return sum((predict == answer)),predict

def demo_func0(x):
    x = [x]
    #scaler.transform(x)
    pred = model.predict(x)
    print(pred)
    return -pred

def demo_func1(x):
    x = [x]
    #scaler.transform(x)
    pred = model.predict(x)
    amount,pre_label = judge(x,c_model)
    print('回归预测值:{},分类预测标签:{}，ADMET性质个数:{}'.format(pred,pre_label,amount))
    if amount >= 3:
        return -pred
    else:
        return pred

def run_pso():
    pso = PSO(demo_func1, dim=20, pop=50, max_iter=100, lb=min_list[0], ub=max_list[0], w=0.8, c1=2, c2=2)
    pso.run()
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    plt.plot(pso.gbest_y_hist)
    plt.show()
    return pso.gbest_x

def get_best():
    new = pd.DataFrame(columns=['var_name','x_value'])
    new['var_name'] = var_list
    new['x_value'] = run_pso()
    print(new)
    new.to_excel(r'./bestanswer_3limit.xlsx',index=False,header=True)

def valid():
    best_limit3 = np.array([pd.read_excel(r'./bestanswer_3limit.xlsx',index_col=False,header=0)['x_value']]).reshape(1,-1)
    print(judge(best_limit3,c_model))
    print(model.predict(best_limit3))

run_pso()
#valid()