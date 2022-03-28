import pandas as pd
from time import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



#read:index = 0/false and header = 0/false:第0行或者第0列为行号/列号，如果为false则默认加上一行/列作为索引
#save：index = true/false and header = true/false，当前状态下是否要把行标列标保存进去
class ExcelProcessing():
    def __init__(self):
        pass

    def get_raw(self,path):
        return pd.read_excel(path)

    def get_sheet(self,path):
        data_xls = pd.ExcelFile(path)
        print(data_xls.sheet_names)
        data = {}
        for name in data_xls.sheet_names:
            df = data_xls.parse(sheetname=name, header=None)
            data[name] = df
            #print(df)
            #print(name)
        return data


    def data_formating(self,df,type):
        #type :all/any
        #delete invalid row and column
        df =  df.dropna(axis=0, how=type, inplace = False)
        return df.dropna(axis=1, how=type, inplace = False)

    def date_filter(self,raw_data,thresh,column_name):
        #keep row if it's length more than thresh
        row_polishing = raw_data.dropna(axis=0, thresh=thresh, inplace=False)
        # keep column if it's length more than thresh
        column_polishing = raw_data.dropna(axis=1, thresh=thresh, inplace=False)
        #delete the row whose specified column inlude nan value,axis must be zero if searching by column
        specific_polishing = raw_data.dropna(axis=0, subset=[column_name], inplace=False)
        return row_polishing,column_polishing,specific_polishing

    def get_column_id(self,df):
        return df.keys()

    def delete_specified_column(self,df,column_list):
        return df.drop(column_list, axis=1)

    def delete_specified_row(self,df,row_list):
        #type of element in rowlist must be int,cause index always be int type
        #row_list=[6,7,8]
        return df.drop(row_list,inplace = False)

    def slice(self,df, rs,re, cs,ce):
        #page 304[1:2,3:4]
        # ['nama1':'nama2','nama3':'nama4']
        # [['nama1','nama2'],['nama3','nama4']] 特定行列
        # [:,['nama3','nama4']] 特定行/特定列
        #特定个需要列表，切片只需要最外层括号
        return df.iloc[rs:re, cs:ce]

    def query(self,df,bool_str):
        #bool_str='IF1>100 and IF5 < 20'
        return df.query(bool_str)

    def get_singal_value(self,df,column_name,line_position):
        #line_position start at 0
        return df.column_name.loc[line_position]

    def get_imf_rank(self,df,targer_id_c):

        target_c = df.loc[:,[targer_id_c]]  # target column
        factors = self.delete_specified_column(df=df,column_list=['FE1','FE2','FE10','FE11'])

        factors.to_excel(r'./x1_data.xlsx')
        target_c.to_excel(r'./y1_data.xlsx')

        factors = pd.read_excel(r'./x1_data.xlsx',index=False,header=None)
        target = pd.read_excel(r'./y1_data.xlsx', index=False,header=None)

        t0 = time()

        clf = RandomForestRegressor(n_estimators=500, random_state=(0), max_features=100, n_jobs=(2))
        x_train, x_test, y_train, y_test = train_test_split(factors, target, test_size=0.3, shuffle=True, random_state=0)
        clf.fit(factors, target)

        print("done in %0.3fs" % (time() - t0))

        importances = clf.feature_importances_
        df1 = pd.DataFrame(importances, columns=['importance'], dtype=float)
        df1.sort_values(by="importance", inplace=True, ascending=False)
        print(df1)

#---------------------------------------------------------------------------------------------------------
# ---notice augments 'inplace = True /False':TRUE need no new variable name but False must get new one,
# meanwhile concentrate whether return a new object or changed in original memory




