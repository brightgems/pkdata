# coding: utf-8
import pandas as pd
import pprint
import tableprint as tp
import numpy as np
import itertools as it

import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers
from keras import backend as K
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve, confusion_matrix

font0 = FontProperties()
alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
# Show family options

families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']

font1 = font0.copy()
font1.set_size('large')
plt.style.use('ggplot')
pp = pprint.PrettyPrinter(indent=4)

# 读取X和y，均为未标准化数据集, y为log后的数据
X = pd.read_csv('tmp/X.csv',memory_map=True)
X=X.fillna(0)
y = pd.read_csv('tmp/y.csv',header=None)
y_clf = y.values.ravel()>0
y_log = np.log(y.values.ravel()+1)

df_variable = pd.read_excel('data/tap4fun.xlsx',sheet_name='Sheet1')
df_variable = df_variable.iloc[:,range(3)].rename(columns={'字段名':'column','字段解释':'desc','数据时间':'cycle'})
df_variable['cat'] = df_variable.column.str.split('_').apply(lambda x: x[0])


# 因子合成
# 查询消费类因子
from toolz import filter
pay_cols= list(filter(lambda x: 'pay' in x, X.columns))

X['avg_online_minutes_log10'] = np.log10(X['avg_online_minutes']+1)

active_cols= list(filter(lambda x: 'avg_online' in x, X.columns)) 
achievement_cols = ['army_add','resource_add','acceleration_add']
for col in achievement_cols:
    X[col+'_log'] = np.log(X[col]+1)

X['achievement'] = np.log(X['army_add']+X['resource_add']+1)
X['achievement_per_hour'] = X['achievement']/X['avg_online_minutes']/60
X['acceleration_per_hour'] = X['acceleration_add']/X['avg_online_minutes']/60    
X.loc[X.avg_online_minutes==0,'achievement_per_hour']=0
X.loc[X.avg_online_minutes==0,'acceleration_per_hour']=0
X['acceleration_per_hour_log']= np.log(X['acceleration_per_hour']+1)
X['acceleration_reduce_per_hour_log']= np.log(X['acceleration_per_hour']*X['acceleration_reduce%']+1)
X['resource_reduce_log']= np.log(X['resource_add']*X['resource_reduce%']+1)
X['army_reduce_log']= np.log(X['army_add']*X['resource_reduce%']+1)
X['achievement_per_hour_log']= np.log(X['achievement_per_hour']+1)
X['sr_main_score_per_hour_log']= np.log(X['sr_main_score']/X['avg_online_minutes']/60  +1)
X['sr_level_score_per_hour']= X['sr_level_score']/X['avg_online_minutes']/60
X['army_reduce_per_battle'] = (X['army_add']*X['army_reduce%']+1)/X['battle_count']
X.loc[X.avg_online_minutes==0,'sr_main_score_per_hour_log']=0
X.loc[X.avg_online_minutes==0,'sr_level_score_per_hour']=0
X.loc[X.battle_count==0,'army_reduce_per_battle']=0
X['sr_level_score_per_hour_log']= np.log(X['sr_level_score_per_hour'] +1)
X['bd_resource_ratio'] = X['resource_reduce%']*X['resource_add']/X['bd_main_score']
X.loc[X.bd_main_score==0,'bd_resource_ratio']=0
X['army_reduce_per_battle_log']= np.log(X['army_reduce_per_battle'] +1)
X['minutes_per_battle'] = X['avg_online_minutes']/X['battle_count']
X.loc[X.battle_count==0,'minutes_per_battle']=0
X = X.fillna(0)
war_cols = ['battle_count','army_reduce_per_battle_log','minutes_per_battle','bd_resource_ratio','battle_win%','pvp_launch%','pvp_win%']
achievement_cols= ['sr_main_score','bd_main_score']#'resource_reduce_log','achievement','achievement_per_hour_log','acceleration_per_hour_log','acceleration_reduce_per_hour_log']
achievement_cols += war_cols

from sklearn.linear_model import LassoCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from os import cpu_count
X_data = X.loc[:,achievement_cols]
lsvc = LassoCV(n_jobs = cpu_count(),fit_intercept=True,normalize=True).fit(X_data, y_log)

model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X_data)
achievement_cols=list(X_data.columns[lsvc.coef_!=0])

from sklearn.decomposition import PCA
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler

# 特征合成: 集成
others_col = list(filter(lambda x: x not in pay_cols+active_cols,X.columns))
mapper_tp = DataFrameMapper([
    (pay_cols, PCA(n_components=10),{'alias':'pay_pca_'}),
    ]+list(map(lambda x: (x, None),active_cols+achievement_cols))
    , df_out=True)

X_transform = mapper_tp.fit_transform(X)

# 划分训练测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_transform, y_clf, test_size=.2,stratify =y_clf)

# 预测7日后付费会员
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler_X = MinMaxScaler()

from os import cpu_count
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(verbose=True, n_jobs=cpu_count(), 
                             class_weight={False:1, True:3}, min_samples_leaf = 2,
                             n_estimators=10, max_depth =8)

pipeline_clf = make_pipeline(scaler_X, clf)

pipeline_clf.fit(X_train, y_train)
var_importance = pipeline_clf.steps[1][1].feature_importances_
df_variable_importance =pd.DataFrame({'column':X_train.columns, 'importance':var_importance})
df_variable_importance = pd.merge(df_variable_importance,df_variable,on='column',how='left')
sel_vars = df_variable_importance[df_variable_importance.importance>0.001].sort_values('importance', ascending = False).column




data = {'train':[X_train, y_train],'test':[X_test,y_test]}
y_stacked = {'train':{},'test':{}}

for stage in data.keys():
    X_in, y_in = data.get(stage)
    y_predict = pipeline_clf.predict(X_in)
    f1 = f1_score(y_predict, y_in)
    accuracy = accuracy_score(y_predict, y_in)
    precision= precision_score(y_predict, y_in)
    recall = recall_score(y_predict, y_in)
    print("{0}: f1={1},accuracy={2},precision={3},recall={4}".format(stage, f1,accuracy,precision,recall))
    y_stacked[stage]['random_forest']=y_predict
confusion_matrix(y_predict, y_in)

# Apply Keras


def auc(y_true, y_pred):   
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)    
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)    
    return TP/P

def make_model():
    model = Sequential([
        Dense(16, input_shape=(len(sel_vars),)),
        Activation('relu'),
        Dense(8),
        Activation('relu'),
        Dense(1),
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='Adadelta',
                  metrics=[auc,'accuracy'])

    return model
pipeline_classfier = make_pipeline(scaler_X,KerasClassifier(build_fn = make_model, batch_size=200000,
                                                            class_weight={False:1,True:5}))
pipeline_classfier.fit(X_train.loc[:,sel_vars],y_train)

data = {'train':[X_train, y_train],'test':[X_test,y_test]}

for stage in data.keys():
    X_in, y_in = data.get(stage)
    y_predict = pipeline_classfier.predict(X_in.loc[:,sel_vars])
    f1 = f1_score(y_predict, y_in)
    accuracy = accuracy_score(y_predict, y_in)
    precision= precision_score(y_predict, y_in)
    recall = recall_score(y_predict, y_in)
    print("{0}: f1={1},accuracy={2},precision={3},recall={4}".format(stage, f1,accuracy,precision,recall))
print(confusion_matrix(y_predict, y_in))