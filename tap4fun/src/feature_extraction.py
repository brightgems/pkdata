from copy import deepcopy

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from steppy.base import BaseTransformer
from steppy.utils import get_logger
from sklearn.preprocessing import MinMaxScaler
from toolz.curried import pipe, map, filter, get
from .utils import compress_dtypes

logger = get_logger()


class FeatureJoiner(BaseTransformer):
    def transform(self, numerical_feature_list, categorical_feature_list, **kwargs):
        features = numerical_feature_list + categorical_feature_list
        for feature in features:
            feature.reset_index(drop=True, inplace=True)
        outputs = dict()
        outputs['features'] = pd.concat(features, axis=1).astype(np.float32)
        outputs['feature_names'] = self._get_feature_names(features)
        outputs['categorical_features'] = self._get_feature_names(categorical_feature_list)
        return outputs

    def _get_feature_names(self, dataframes):
        feature_names = []
        for dataframe in dataframes:
            try:
                feature_names.extend(list(dataframe.columns))
            except Exception as e:
                print(e)
                feature_names.append(dataframe.name)

        return feature_names


class CategoricalEncoder(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()
        self.categorical_columns = kwargs['categorical_columns']
        params = deepcopy(kwargs)
        params.pop('categorical_columns', None)
        self.params = params
        self.encoder_class = ce.OrdinalEncoder
        self.categorical_encoder = None

    def fit(self, X, y, **kwargs):
        X_ = X[self.categorical_columns]
        self.categorical_encoder = self.encoder_class(cols=self.categorical_columns, **self.params)
        self.categorical_encoder.fit(X_, y)
        return self

    def transform(self, X, **kwargs):
        X_ = X[self.categorical_columns]
        X_ = self.categorical_encoder.transform(X_)
        return {'categorical_features': X_}

    def load(self, filepath):
        self.categorical_encoder = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.categorical_encoder, filepath)


class GroupbyAggregate(BaseTransformer):
    def __init__(self, groupby_aggregations):
        super().__init__()
        self.groupby_aggregations = groupby_aggregations
        self.features = []
        self.feature_names = []

    def fit(self, main_table, **kwargs):
        for groupby_cols, specs in self.groupby_aggregations:
            group_object = main_table.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)

                group_features = group_object[select].agg(agg).reset_index() \
                    .rename(index=str,
                            columns={select: groupby_aggregate_name})[groupby_cols + [groupby_aggregate_name]]

                self.features.append((groupby_cols, group_features))
                self.feature_names.append(groupby_aggregate_name)
        return self

    def transform(self, main_table, **kwargs):
        for groupby_cols, groupby_features in self.features:
            main_table = main_table.merge(groupby_features,
                                          on=groupby_cols,
                                          how='left')

        return {'numerical_features': main_table[self.feature_names].astype(np.float32)}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.features = params['features']
        self.feature_names = params['feature_names']
        return self

    def persist(self, filepath):
        params = {'features': self.features,
                  'feature_names': self.feature_names}
        joblib.dump(params, filepath)

    def _create_colname_from_specs(self, groupby_cols, agg, select):
        return '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)


class GroupbyAggregateMerge(BaseTransformer):
    def __init__(self, table_name, id_columns, groupby_aggregations):
        super().__init__()
        self.table_name = table_name
        self.id_columns = id_columns
        self.groupby_aggregations = groupby_aggregations

    @property
    def feature_names(self):
        feature_names = list(self.features.columns)
        feature_names.remove(self.id_columns[0])
        return feature_names

    def fit(self, main_table, side_table, **kwargs):
        features = pd.DataFrame({self.id_columns[0]: side_table[self.id_columns[0]].unique()})

        for groupby_cols, specs in self.groupby_aggregations:
            group_object = side_table.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)
                features = features.merge(group_object[select]
                                          .agg(agg)
                                          .reset_index()
                                          .rename(index=str,
                                                  columns={select: groupby_aggregate_name})
                                          [groupby_cols + [groupby_aggregate_name]],
                                          on=groupby_cols,
                                          how='left')
        self.features = features
        return self

    def transform(self, main_table, side_table, **kwargs):
        main_table = main_table.merge(self.features,
                                      left_on=[self.id_columns[0]],
                                      right_on=[self.id_columns[1]],
                                      how='left',
                                      validate='one_to_one')

        return {'numerical_features': main_table[self.feature_names].astype(np.float32)}

    def load(self, filepath):
        self.features = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.features, filepath)

    def _create_colname_from_specs(self, groupby_cols, select, agg):
        return '{}_{}_{}_{}'.format(self.table_name, '_'.join(groupby_cols), agg, select)


class VarialbeFeatures(BaseTransformer):

    def transform(self, X, **kwargs):
        X = X.iloc[:,range(3)].rename(columns={'字段名':'column','字段解释':'desc','数据时间':'cycle'})
        X['cat'] = X.column.str.split('_').apply(lambda x: x[0])
        X.loc[X.column.str.find('acceleration')>0,'cat'] = 'acceleration'
        X.loc[2:11,'cat'] = 'resource'
        X.loc[12:23,'cat'] = 'army'
        return {'X':X}

class Tap4funFeatures(BaseTransformer):
    def __init__(self, categorical_columns, numerical_columns):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

    def select_columns(self, X):
        """
            adapt to feature selection by configuration
        """
        cat_features = X.select_dtypes('category').columns
        num_features = X.select_dtypes(np.number).columns
        if not self.categorical_columns:
            self.categorical_columns = cat_features
        else:
            self.categorical_columns = \
                pipe(self.categorical_columns, filter(lambda x: x in cat_features), list)
        
        
        if not self.numerical_columns:
            self.numerical_columns = num_features
        else:
            self.numerical_columns = \
                pipe(self.numerical_columns, filter(lambda x: x in num_features), list)
        
        selection = self.numerical_columns + self.categorical_columns
        unused_features = pipe(X.columns, filter(lambda x: x not in selection),list)
        logger.info("Useless features:\n"+str(unused_features))

    def transform(self, X,df_variable, **kwargs):
        # share var
        scaler =MinMaxScaler()
        # payment features
        X['pay_price_group']=pd.cut( X['pay_price'],[-0.01,0,1,10,100,1000,5000,10000])
        X['is_active_user'] =X['avg_online_minutes']>2
        X['avg_pay'] = X['pay_price']/X['pay_count']
        X['avg_pay_group'] = pd.cut(X['avg_pay'],[-0.01,0,1,5,10,20,50,100])
        X['pay_count_group']=pd.cut( X['pay_count'],[-1,0,1,2,3,4,200])
        X['avg_online_minutes_group'] = pd.cut(X['avg_online_minutes'], [-0.001,0,2,5,20,60,120,300,1200,2400],include_lowest=True)
        X =X.assign(pay_price_per_hour = lambda x: x.pay_price/x.avg_online_minutes/60)
        X.loc[X.avg_online_minutes==0,'pay_price_per_hour'] = 0
        # date features
        X['register_date']=X.register_time.apply(lambda x:pd.to_datetime(x).strftime('%Y-%m-%d'))
        X['register_date'] = X['register_date'].astype(np.datetime64)
        X = X.assign(dayofweek = lambda x: x['register_date'].dt.dayofweek)
        X = X.assign(day = lambda x: x['register_date'].dt.day)
        X = X.assign(week = lambda x: x['register_date'].dt.week)
        # day#
        days = sorted(X.register_date.unique())
        days = dict(zip(days, range(len(days))))
        X['day_num'] = X.register_date.map(days)
        # time class
        def map_to_timecls(x):
            if x<=7:
                return 'night'
            elif x<=12:
                return 'morning'
            elif x<=18:
                return 'afternoon'
            else:
                return 'evening'
        X['register_timecls'] = X.assign(
                register_timecls = lambda x: x['register_time'].dt.hour
            )['register_timecls'].map(map_to_timecls)
        # research features
        sr_var = list(df_variable[df_variable.cat=='sr'].column.values)
        df_srdesc = X.loc[:,sr_var].sample(frac=.1).describe().transpose()
        cols_sr_level = df_srdesc[df_srdesc['max']==1].index
        cols_sr_main = df_srdesc[df_srdesc['max']>1].index
        X['sr_level_score'] = X.filter(cols_sr_level).sum(axis=1)
        X['sr_main_score']=X.filter(cols_sr_main).sum(axis=1)

        for function_name in ['mean','var','min','max']:
            X['sr_main_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
                X[cols_sr_main], axis=1)
        # building features
        bd_var = list(df_variable[df_variable.cat=='bd'].column.values)
        X['bd_main_score']=X.loc[:,bd_var].sum(axis=1)
        for function_name in ['mean','var','min','max']:
            X['bd_main_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
                X.loc[:,bd_var], axis=1)
        acct_value = X[['bd_main_score','sr_level_score','sr_main_score']].sum(axis=1)
        
        acct_value_zs = np.sum(scaler.fit_transform(X[['bd_main_score','sr_level_score','sr_main_score']]),axis=1)
        X['acct_value'] =acct_value
        X['acct_value_zs'] =acct_value_zs
        
        # gain/loss features
        def merge_add_reduce(cat):
            vars = df_variable[df_variable.cat==cat].column
            vars_add = pipe(vars, filter(lambda x: x.find('add_')>0),list)
            vars_rdc = pipe(vars, filter(lambda x: x.find('reduce_')>0),list)
            X[cat+'_add']= X[vars_add].sum(axis=1)
            X[cat+'_reduce']= X[vars_rdc].sum(axis=1)
            subcats = np.unique([str.join('_',a.split('_')[:-2]) for a in vars])
            for each in subcats:
                try:
                    X[each+'_reduce%']= X[each+'_reduce_value']/(X[each+'_add_value']+1)
                    X[each+'_net']= X[each+'_add_value'] - X[each+'_reduce_value']
                except:
                    pass
            
        cat_set = df_variable[df_variable.column.str.find('reduce')>0].cat.unique()
        for cat in cat_set:
            merge_add_reduce(cat)
        # war features
        X['pve_launch%'] = X['pve_lanch_count']/(1+X['pve_battle_count'])
        X['pve_win%'] = X['pve_win_count']/(1+X['pve_battle_count'])
        X['pvp_launch%'] = X['pvp_lanch_count']/(1+X['pvp_battle_count'])
        X['pvp_win%'] = X['pvp_win_count']/(1+X['pvp_battle_count'])
        X['battle_count'] = X['pvp_win_count']+X['pve_win_count']
        X['battle_win'] = X['pve_battle_count']+X['pvp_battle_count']
        X['battle_launch%'] = (X['pve_battle_count']+X['pvp_battle_count'])/(1+X['battle_count'] )
        X['battle_win%'] = (X['pve_win_count']+X['pvp_win_count'])/(1+X['battle_count'])

        # achievement features
        X['register_timecls']=X.assign(register_timecls = lambda x: x['register_time'].dt.hour)['register_timecls'].map(map_to_timecls)
        X['acceleration_army_reduce_ratio'] = X['acceleration_reduce']/(X['army_reduce']+1)
        X['resource_army_reduce_ratio'] = X['resource_reduce']/(X['army_reduce']+1)
        X['acceleration_resource_reduce_ratio'] = X['acceleration_reduce']/(X['resource_reduce']+1)
        X['acceleration_army_add_ratio'] = X['acceleration_add']/(X['army_add']+1)
        X['resource_army_add_ratio'] = X['resource_add']/(X['army_add']+1)
        X['acceleration_add_add_ratio'] = X['acceleration_add']/(X['resource_add']+1)
        X['bd_sr_product']= X['bd_main_score']*X['sr_main_score']
        X['bd_sr_ratio']= X['bd_main_score']/(X['sr_main_score']+1)
        X['acceleration_per_hour'] = X['acceleration_reduce']/(1+X['avg_online_minutes'])/60 
        X['sr_level_score_per_hour']= X['sr_level_score']/(1+X['avg_online_minutes'])/60
        X['army_reduce_per_battle'] = X['army_reduce']/(X['battle_count']+1)
        X['sr_army_prod'] = X['sr_level_score']*X['army_add']
        X['army_per_hour'] = X['army_reduce']/(1+X['avg_online_minutes'])/60
        X['bd_resource_ratio'] = X['resource_reduce']/(1+X['bd_main_score'])
        X = compress_dtypes(X)
        self.select_columns(X)
        return {'numerical_features': X[self.numerical_columns],
                'categorical_features': X[self.categorical_columns],
                'X':X
                }


class ConcatFeatures(BaseTransformer):
    def transform(self, **kwargs):
        features_concat = []
        for _, feature in kwargs.items():
            feature.reset_index(drop=True, inplace=True)
            features_concat.append(feature)
        features_concat = pd.concat(features_concat, axis=1)
        return {'concatenated_features': features_concat}
