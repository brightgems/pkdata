from copy import deepcopy

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from steppy.base import BaseTransformer
from steppy.utils import get_logger
from toolz.curried import pipe, map, filter, get
from .utils import compress_dtypes, get_random_string
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2, f_classif
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from .pipeline_config import DAYS_COLUMN

logger = get_logger()

class SelectKBestFeatures(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()
        self.k = kwargs.get('k', 20)
        self.metric = kwargs.get('metric', f_classif)
    
    def fit(self, X, y, **kwargs):
        self.estimator = SelectKBest(self.metric, k=self.k).fit(X, y)
        
    def transform(self, X, **kwargs):
        k_features = self.estimator.transform(X)
        return {'features': k_features}

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.estimator, filepath)


class SelectFeaturesFromModel(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()
        self.threshold = kwargs.get('threshold', 'median')
    
    def fit(self, X, y, **kwargs):
        clf_lr = LogisticRegression(penalty='l1',class_weight='balanced',solver='liblinear',C=.02,n_jobs=8)
        # clf_lr = RandomForestClassifier()
        self.estimator = make_pipeline(MinMaxScaler(), SelectFromModel(clf_lr, threshold=self.threshold)).fit(X, y)
        
    def transform(self, X, **kwargs):
        k_features = self.estimator.transform(X)
        support = self.estimator.steps[-1][1].get_support()
        feature_names = X.columns[support].values.tolist()
        k_features = pd.DataFrame(k_features, columns=feature_names)
        # print("selected features from model:\r{}".format(str(feature_names)))
        return {'features': k_features, 'feature_names':feature_names}

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.estimator, filepath)

                
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
        X_.fillna(0,inplace=True)
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
        self.feature_names = []

    def __do_aggredate(self, main_table, days_col_fit = False):
        """
            insert aggregate column by groupby_aggregations

            Arguments:
                 days_col_fit(bool): process grouby cols has days_col only
        """
        self.feature_names_temp = deepcopy(self.feature_names)
        self.features_temp = deepcopy(self.features)

        for groupby_cols, specs in self.groupby_aggregations:
            if days_col_fit ^ (DAYS_COLUMN in groupby_cols):
                # aggregate should delay until transform when groupby_cols has days 
                continue
            group_object = main_table.groupby(groupby_cols)
            for select, agg in specs:
                if str(type(agg)) == "<class 'function'>":
                    groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, get_random_string())
                    group_features = group_object[select].apply(agg).reset_index()
                else:
                    groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)
                    group_features = group_object[select].agg(agg).reset_index()
                group_features = group_features.rename(index=str,
                            columns={select: groupby_aggregate_name})[groupby_cols + [groupby_aggregate_name]]
                group_features.fillna(0, inplace=True)
                
                if days_col_fit:
                    self.features_temp.append((groupby_cols, group_features, select))
                    self.feature_names_temp.append(groupby_aggregate_name)
                else:
                    self.features.append((groupby_cols, group_features, select))
                    self.feature_names.append(groupby_aggregate_name)
        

    def fit(self, main_table, **kwargs):
        self.features = []
        self.__do_aggredate(main_table, False)
        return self

    def transform(self, main_table, **kwargs):
        # create aggregate col group by days_col
        self.__do_aggredate(main_table, True)
        neural_feature_names = []

        for index, (groupby_cols, groupby_features, select) in enumerate(self.features_temp):
            feature_name = self.feature_names_temp[index] 
            main_table = main_table.merge(groupby_features,on=groupby_cols, how='left')
            # create neural features
            if not any([feature_name.endswith(func) for func in ['count']]):
                neural_feature_name = feature_name +'_neural'
                try:
                    main_table[neural_feature_name] =  main_table[select]/(main_table[feature_name]+1)
                except:
                    # work around for strange error
                    main_table = main_table.merge(groupby_features,on=groupby_cols, how='left')
                    main_table[neural_feature_name] =  main_table[select]/(main_table[feature_name]+1)
                
                neural_feature_names.append(neural_feature_name)
        self.feature_names_temp.extend(neural_feature_names)
        logger.info("created aggreate cols#: %d"%len(self.feature_names_temp))
        main_table = compress_dtypes(main_table)
        return {'numerical_features': main_table[self.feature_names_temp].astype(np.float32)}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.features = params['features']
        self.feature_names = params['feature_names']
        return self

    def persist(self, filepath):
        params = {'features': self.features,
                  'feature_names': self.feature_names}
        joblib.dump(params, filepath)

    def _create_colname_from_specs(self, groupby_cols, select, agg):
        return '{}_{}_{}'.format('_'.join(groupby_cols), select, agg )


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


class Tap4funFeatures(BaseTransformer):
    def __init__(self, categorical_columns, numerical_columns):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

    def select_columns(self, X):
        """
            adapt to feature selection by configuration
        """
        cat_features = X.select_dtypes('object').columns
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

    def transform(self, X, **kwargs):
        self.select_columns(X)
        X[self.numerical_columns].fillna(0, inplace=True)
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
