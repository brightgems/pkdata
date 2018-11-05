import os
import pickle
import shutil

import numpy as np
import pandas as pd
import requests
from attrdict import AttrDict
from toolz.curried import pipe, map, filter, get
from deepsense import neptune
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_squared_error, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from .utils import compress_dtypes


def read_variable():
    """
        read feature desc table
    """
    X = pd.read_excel('./data/tap4fun.xlsx')

    X = X.iloc[:,range(3)].rename(columns={'字段名':'column','字段解释':'desc','数据时间':'cycle'})
    X['cat'] = X.column.str.split('_').apply(lambda x: x[0])
    X.loc[X.column.str.find('acceleration')>0,'cat'] = 'acceleration'
    X.loc[2:11,'cat'] = 'resource'
    X.loc[12:23,'cat'] = 'army'
    return X

def prepare_dataset():
    """
        generate train/test hdf file by feature engineer on raw data
    """
    df_variable = read_variable()
    df_train = pd.read_csv('data/tap_fun_train.csv', parse_dates=['register_time'], infer_datetime_format=True)
    df_train=df_train.rename(columns={'\ufeffuser_id':'user_id'})
    df_test = pd.read_csv('data/tap_fun_test.csv', parse_dates=['register_time'], infer_datetime_format=True)
    df_test['prediction_pay_price'] = np.nan
    df_all = df_train.append(df_test)
    df_all.reset_index()

    # =========make numerical features=========
    # payment features
    df_all['is_active_user'] =df_all['avg_online_minutes']>2
    df_all['avg_pay'] = df_all['pay_price']/df_all['pay_count']
    df_all =df_all.assign(pay_price_per_hour = lambda df_all: df_all.pay_price/df_all.avg_online_minutes/60)
    df_all.loc[df_all.avg_online_minutes==0,'pay_price_per_hour'] = 0
    # date features
    df_all['register_date']=df_all.register_time.apply(lambda df_all:pd.to_datetime(df_all).strftime('%Y-%m-%d'))
    df_all['register_date'] = df_all['register_date'].astype(np.datetime64)
    df_all = df_all.assign(dayofweek = lambda df_all: df_all['register_date'].dt.dayofweek)
    df_all = df_all.assign(day = lambda df_all: df_all['register_date'].dt.day)
    df_all = df_all.assign(week = lambda df_all: df_all['register_date'].dt.week)
    # day#
    days = sorted(df_all.register_date.unique())
    days = dict(zip(days, range(len(days))))
    df_all['day_num'] = df_all.register_date.map(days)
    # research features
    sr_var = list(df_variable[df_variable.cat=='sr'].column.values)
    df_srdesc = df_all.loc[:,sr_var].sample(frac=.1).describe().transpose()
    cols_sr_level = df_srdesc[df_srdesc['max']==1].index
    cols_sr_main = df_srdesc[df_srdesc['max']>1].index
    df_all['sr_level_score'] = df_all.filter(cols_sr_level).sum(axis=1)
    df_all['sr_main_score']=df_all.filter(cols_sr_main).sum(axis=1)

    for function_name in ['mean','var','min','max']:
        df_all['sr_main_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
            df_all[cols_sr_main], axis=1)
        df_all['sr_level_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
            df_all[cols_sr_level], axis=1)
    # building features
    bd_var = list(df_variable[df_variable.cat=='bd'].column.values)
    df_all['bd_main_score']=df_all.loc[:,bd_var].sum(axis=1)
    for function_name in ['mean','var','min','max']:
        df_all['bd_main_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
            df_all.loc[:,bd_var], axis=1)
    acct_value = df_all[['bd_main_score','sr_level_score','sr_main_score']].sum(axis=1)
    scaler =MinMaxScaler()
    acct_value_zs = np.sum(scaler.fit_transform(df_all[['bd_main_score','sr_level_score','sr_main_score']]),axis=1)
    df_all['acct_value'] =acct_value
    df_all['acct_value_zs'] =acct_value_zs
    cat_set = df_variable[df_variable.column.str.find('reduce')>0].cat.unique()

    # net features
    def make_net_features(cat):
        """
            gain/loss features
        """
        vars = df_variable[df_variable.cat==cat].column
        vars_add = pipe(vars, filter(lambda df_all: df_all.find('add_')>0),list)
        vars_rdc = pipe(vars, filter(lambda df_all: df_all.find('reduce_')>0),list)
        df_all[cat+'_add']= df_all[vars_add].sum(axis=1)
        df_all[cat+'_reduce']= df_all[vars_rdc].sum(axis=1)
        subcats = np.unique([str.join('_',a.split('_')[:-2]) for a in vars])
        for each in subcats:
            try:
                # df_all[each+'_reduce%']= df_all[each+'_reduce_value']/(df_all[each+'_add_value']+1)
                df_all[each+'_net']= df_all[each+'_add_value'] - df_all[each+'_reduce_value']
            except:
                pass
    for cat in cat_set:
        make_net_features(cat)
    # war features
    df_all['pve_launch%'] = df_all['pve_lanch_count']/(1+df_all['pve_battle_count'])
    df_all['pve_win%'] = df_all['pve_win_count']/(1+df_all['pve_battle_count'])
    df_all['pvp_launch%'] = df_all['pvp_lanch_count']/(1+df_all['pvp_battle_count'])
    df_all['pvp_win%'] = df_all['pvp_win_count']/(1+df_all['pvp_battle_count'])
    df_all['battle_count'] = df_all['pvp_win_count']+df_all['pve_win_count']
    df_all['battle_win'] = df_all['pve_battle_count']+df_all['pvp_battle_count']
    df_all['battle_launch%'] = (df_all['pve_battle_count']+df_all['pvp_battle_count'])/(1+df_all['battle_count'] )
    df_all['battle_win%'] = (df_all['pve_win_count']+df_all['pvp_win_count'])/(1+df_all['battle_count'])

    # achievement features
    df_all['acceleration_army_reduce_ratio'] = df_all['acceleration_reduce']/(df_all['army_reduce']+1)
    df_all['resource_army_reduce_ratio'] = df_all['resource_reduce']/(df_all['army_reduce']+1)
    df_all['acceleration_resource_reduce_ratio'] = df_all['acceleration_reduce']/(df_all['resource_reduce']+1)
    df_all['acceleration_army_add_ratio'] = df_all['acceleration_add']/(df_all['army_add']+1)
    df_all['resource_army_add_ratio'] = df_all['resource_add']/(df_all['army_add']+1)
    df_all['acceleration_resource_add_ratio'] = df_all['acceleration_add']/(df_all['resource_add']+1)
    df_all['acceleration_resource_cross_ratio'] = df_all['resource_add']/(df_all['acceleration_reduce']+1)
    df_all['acceleration_army_cross_ratio'] = df_all['army_add']/(df_all['acceleration_reduce']+1)
    df_all['bd_sr_product']= df_all['bd_main_score']*df_all['sr_main_score']
    df_all['bd_sr_ratio']= df_all['bd_main_score']/(df_all['sr_main_score']+1)
    df_all['acceleration_per_hour'] = df_all['acceleration_reduce']/(1+df_all['avg_online_minutes'])/60 
    df_all['sr_level_score_per_hour']= df_all['sr_level_score']/(1+df_all['avg_online_minutes'])/60
    df_all['army_reduce_per_battle'] = df_all['army_reduce']/(df_all['battle_count']+1)
    df_all['sr_army_prod'] = df_all['sr_level_score']*df_all['army_add']
    df_all['army_per_hour'] = df_all['army_reduce']/(1+df_all['avg_online_minutes'])/60
    df_all['bd_resource_ratio'] = df_all['resource_reduce']/(1+df_all['bd_main_score'])
    df_all = compress_dtypes(df_all)
    # ========export dataset=========
    df_train = df_all[~df_all.prediction_pay_price.isnull()]
    df_test = df_all[df_all.prediction_pay_price.isnull()]
    # df_train.to_parquet('data/tap_fun_train.parquet')
    # df_test.to_parquet('data/tap_fun_test.parquet')
    df_train.to_hdf('data/tap_fun_train.h5','/data')
    df_test.to_hdf('data/tap_fun_test.h5','/data')

