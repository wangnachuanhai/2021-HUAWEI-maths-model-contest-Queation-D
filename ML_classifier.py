import pandas as pd
import numpy as np
#regress method-----------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
#-------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,recall_score, precision_score,confusion_matrix,f1_score,roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import label_binarize
from classify_data_pre import data_preparation,data_preparation_5c
from openpyxl import load_workbook
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
from time import time
kernel = 1.0 * RBF(np.ones(20))

class MLClassifier():
    def __init__(self,nor_flag,test_size,task_name,c5):
        tmp_x, tmp_y = data_preparation_5c(task_name)
        print('20variable:{}'.format(c5))
        if not c5 : tmp_x, tmp_y = data_preparation(task_name)
        # splite
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            tmp_x, tmp_y, test_size=test_size, shuffle=True, random_state=0)


        #normalization
        self.scaler = StandardScaler()
        if nor_flag:
            self.scaler.fit(self.x_train)
            self.x_train = self.scaler.transform(self.x_train)
            self.x_test = self.scaler.transform(self.x_test)
        self.model = None

    def calPerformance(self,y_tre, y_pre,y_score):
        #idf average == None return a metric list of each label
        accuracy = accuracy_score(y_tre, y_pre)  # 计算准确度
        precision = precision_score(y_tre, y_pre, average='macro')  # 计算精度
        recall = recall_score(y_tre, y_pre, average='macro')  # 计算召回率
        F1_score = f1_score(y_tre, y_pre, average='macro')  # 计算F1得分
        #AUC
        fpr, tpr, thresholds = roc_curve(y_tre, y_score[:,1], pos_label=1)
        auc_r = auc(fpr, tpr)
        return accuracy,precision,recall,F1_score,auc_r

    def print_confusion_matrix_1(self,y_tre, y_pre,title):
        sns.set()
        f, ax = plt.subplots()
        C2 = confusion_matrix(y_tre, y_pre, labels=[0, 1])
        print(C2)  # 打印出来看看
        label = ['negative', 'positive']
        sns.heatmap(C2, annot=True, ax=ax, fmt='.20g', cmap='Blues', xticklabels=label, yticklabels=label)  # 画热力图
        ax.xaxis.tick_top()
        ax.set_title(title)  # 标题
        ax.set_xlabel('predict')  # x轴
        ax.set_ylabel('true')  # y轴
        plt.show()

    def plot_curve(self,y_tre,y_pre):
        fig, ax = plt.subplots()  # 创建图实例
        # x = np.linspace(0,2,100) # 创建x的取值范围
        ax.plot(y_tre, 'b.-', label='true')  # 作y1 = x 图，并标记此线名为linear
        ax.plot(y_pre, 'm.-', label='predict')  # 作y2 = x^2 图，并标记此线名为quadratic
        ax.set_xlabel('sample—id')  # 设置x轴名称 x label
        ax.set_ylabel('value')  # 设置y轴名称 y label
        ax.set_title('model predict')  # 设置图名为Simple Plot
        ax.legend()  # 自动检测要在图例中显示的元素，并且显示
        plt.show()

    def get_model(self,model_name):
        if model_name == 'lgr':
            #hidden layer:100-50
            self.model = LogisticRegression(C=100.0,random_state=1)
        if model_name == 'svc':
            self.model = SVC(kernel='rbf',C=100,random_state=1,probability=True)
        if model_name == 'dtc':
            self.model = DecisionTreeClassifier(max_depth=5,max_features=20,random_state=1)
        if model_name == 'rfc':
            self.model = RandomForestClassifier(n_estimators=100,random_state=(0),n_jobs=2)
        if model_name == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=2)
        if model_name == 'bag':
            self.model = BaggingClassifier()
        if model_name == 'gbdt':
            self.model = GradientBoostingClassifier(n_estimators=100)
        if model_name == 'guss':
            self.model = GaussianProcessClassifier(kernel)

    def train(self):

        self.model.fit(self.x_train, self.y_train)

        y_pred = self.model.predict(self.x_train)
        y_score = self.model.predict_proba(self.x_train)
        accuracy,precision,recall,F1_score,auc_r = self.calPerformance(self.y_train, y_pred,y_score)
        scores = self.model.score(self.x_train, self.y_train)
        print('training model score: {:.4f}'.format(scores))

        y_pred = self.model.predict(self.x_test)
        y_score1 = self.model.predict_proba(self.x_test)
        accuracy1,precision1,recall1,F1_score1,auc1 = self.calPerformance(self.y_test, y_pred,y_score1)
        scores1 = self.model.score(self.x_test, self.y_test)
        print('testing model score: {:.4f}'.format(scores))

        return accuracy,precision,recall,F1_score,auc_r,scores,accuracy1,precision1,recall1,F1_score1,auc1,scores1

    def pretict(self,target_item,sheetname,c5):

        ADMET_path = r'./ADMET.xlsx'
        descriptior_path = r'./Molecular_Descriptor.xlsx'

        ADMET = pd.read_excel(ADMET_path, sheet_name=None)
        descriptior = pd.read_excel(descriptior_path, sheet_name=None)

        ADMET = ADMET['test']
        descriptior = descriptior['test']

        #variable temp:
        columns = pd.read_excel(r'./main_variable20.xlsx',index_col=False,header=0).columns.values
        x = descriptior.loc[:, columns]

        if not c5 :x = descriptior.loc[:, 'nAcid':]

        y = self.model.predict(x)
        print(y)
        ADMET[target_item] = y

        book = load_workbook(r'./ADMET.xlsx')
        writer = pd.ExcelWriter(r'./ADMET.xlsx', engine='openpyxl')
        writer.book = book
        ADMET.to_excel(writer, index=False, header=True, sheet_name='answer{}'.format(sheetname))
        writer.save()

def run_all(task_name):
    test = MLClassifier(nor_flag=True,test_size=0.3,task_name=task_name,c5=True)
    name_list = ['lgr','svc','dtc','rfc','knn','bag','gbdt']
    tmp_list = []
    for method in name_list:
        test.get_model(method)
        accuracy,precision,recall,F1_score,auc,scores,accuracy1,precision1,recall1,F1_score1,auc1,scores1 = test.train()
        tmp_list.append([method,accuracy,precision,recall,F1_score,auc,scores,accuracy1,precision1,recall1,F1_score1,auc1,scores1])
    df = pd.DataFrame(tmp_list,
                      columns=['method', 'train_acc','train_pre','train_rec','train_f1','train_auc','train_score',
                               'test_acc','test_pre','test_rec','test_f1','test_auc','test_score'])
    print(df.sort_values(by='test_auc',ascending=False))

def run_one(task_name,net):
    print('op_target:{}...'.format(task_name))
    sample = MLClassifier(nor_flag=True, test_size=0.3,task_name=task_name,c5=True)
    sample.get_model(net)
    sample.model.fit(sample.x_train, sample.y_train)
    y_pred = sample.model.predict(sample.x_test)
    y_score = sample.model.predict_proba(sample.x_test)
    accuracy,precision,recall,F1_score,auc = sample.calPerformance(sample.y_test,y_pred,y_score)
    print('accuracy:{:4f},precision:{:4f},recall:{:4f},F1_score:{:4f},auc:{:4f}'.format(
        accuracy, precision, recall, F1_score, auc))
    #sample.print_confusion_matrix_1(sample.y_test,y_pred,title=task_name)
    #sample.pretict(target_item=task_name,sheetname=task_name+net+'change_dir',c5=True)
    #-------------------------------------------
    return sample.model, sample.scaler
    #return accuracy,precision,recall,F1_score,auc


if __name__ == '__main__':
    T0 = time()
    c_list = ['Caco-2','CYP3A4','hERG','HOB','MN']
    run_one(task_name = 'MN',net='rfc')
    '''
    new = pd.DataFrame(columns=['metrics','accuracy','precision','recall','F1_score','auc'])
    for name in c_list:
        #run_all(task_name=name)
        accuracy,precision,recall,F1_score,auc_r = run_one(task_name = name,net='rfc')
        new = new.append({'metrics':name,
                          'accuracy':accuracy,
                          'precision':precision,
                          'recall':recall,
                          'F1_score':F1_score,
                          'auc':auc_r},ignore_index=True)
    new = new.round(decimals=4)
    new.to_excel(r'./question5classifier.xlsx', index=False, header=True)
    '''



