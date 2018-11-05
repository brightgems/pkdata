import os
import itertools as it
from attrdict import AttrDict
from deepsense import neptune
from sklearn.metrics import roc_auc_score
from .utils import read_params, parameter_eval, rmse
import numpy as np


ctx = neptune.Context()
params = read_params(ctx, fallback_file='./neptune_reg.yaml')

RANDOM_SEED = 90210
DEV_SAMPLE_SIZE = 50000

ID_COLUMNS = ['user_id']
is_regression_problem = params.is_regression_problem
if is_regression_problem:
    TARGET_COLUMNS = ['prediction_future_pay_price']
    score_function = rmse
    score_name = 'RMSE'
else:
    TARGET_COLUMNS = ['is_future_no_pay']
    score_function = roc_auc_score
    score_name = 'ROC_AUC'

CATEGORICAL_COLUMNS = [
    'pay_price_group',
    'register_timecls',
    'avg_online_minutes_group',
]

DAYS_COLUMN = 'day_num'

NUMERICAL_COLUMNS = [
    # active/payment features
     'day_num',
    'avg_online_minutes', 'pay_price', 'pay_price_per_hour', 'is_active_user',
    'dayofweek', 
    'sr_main_var', 'sr_level_var', 'net_var',
    # war features
    'battle_win%',
    # achievement features
    'resource_add', 'acceleration_add'
    'acct_value_zs', 'resource_reduce', 'army_reduce'
    'sr_main_score',
    # add features
    'stone_add_value', 'ivory_add_value', 'general_acceleration_add_value',
       'training_acceleration_add_value'
    # reduce features
    'wood_reduce_value', 'ivory_reduce_value',
       'general_acceleration_reduce_value',
       'reaserch_acceleration_reduce_value',
    # net features
    'meat_net', 'cavalry_net', 'general_acceleration_net',
       'training_acceleration_net',
    # ratio features
    'acceleration_army_add_ratio', 'resource_army_add_ratio',
       'bd_resource_ratio',
    # mix features
    'sr_army_prod',
    'bd_sr_product',
]

USELESS_COLUMNS = [
    'is_future_no_pay',
    'prediction_pay_price',
    'prediction_future_pay_price',
    'user_id',
    'day', 'week', 'day_num',
]

IMPORTANT_COLUMNS = ['wood_net' 'sr_main_mean' 'battle_count' 'magic_net' 'army_per_hour'
 'resource_reduce' 'bd_resource_ratio' 'resource_add' 'stone_net'
 'sr_main_var' 'bd_main_mean' 'ivory_net' 'reaserch_acceleration_net'
 'bd_main_var' 'building_acceleration_net'
 'acceleration_resource_reduce_ratio' 'acceleration_resource_cross_ratio'
 'general_acceleration_net' 'training_acceleration_net'
 'acceleration_resource_add_ratio']

# ensure USELESS_COLUMNS never appear in NUMERICAL_COLUMNS
NUMERICAL_COLUMNS = list(filter(lambda x: x not in USELESS_COLUMNS, NUMERICAL_COLUMNS))

TAP4FUN_AGGREGATION_RECIPIES = [
    (['day_num'], [
                   ('avg_online_minutes', 'mean'),
                   ('avg_online_minutes', lambda x: x.quantile(0.99)-x.quantile(0.01)),
                   ('avg_online_minutes', lambda x: x.skew()),
                   ('avg_online_minutes', lambda x: x.median()),
                   ('avg_online_minutes', lambda x: x.max()-x.min()),
                   ('pay_price', 'mean'),
                   ('pay_price', 'sum'),
                   ('pay_count', 'mean'),
                   ('pay_count', 'sum'),
                   ('sr_main_var', 'mean'),
                   ('meat_net', 'mean'),
                   ('pay_count', 'mean'),
                   ('acct_value_zs', 'mean'),
                   ('acct_value_zs', lambda x: x.skew()),
                   ('acct_value_zs', lambda x: x.median()),
                   ('acct_value_zs', lambda x: x.max()-x.min()),
                   ]),
    (['avg_online_minutes_group'],[
                     ('avg_online_minutes', 'mean'),
                   ('avg_online_minutes', lambda x: x.skew()),
                   ('avg_online_minutes', lambda x: x.median()),
                   ('avg_online_minutes', lambda x: x.max()-x.min()),
                   ('pay_price', 'mean'),
                   ('pay_price', 'sum'),
                   ('pay_count', 'mean'),
                   ('pay_count', 'sum'),
                   ('meat_net', 'mean'),
                   ('general_acceleration_net', 'mean'),
                   ('acct_value_zs', 'mean'),
                   ('acct_value_zs', lambda x: x.skew()),
                   ('acct_value_zs', lambda x: x.median()),
                   ('acct_value_zs', lambda x: x.max()-x.min()),
                    ]),
    (['register_timecls'], [
                            ('pay_price_per_hour','mean'),
                            ('pay_price_per_hour','var'),
                            ('avg_online_minutes', 'mean'),
                            ('pay_count', 'mean'),
                            ('acct_value_zs', 'mean')]),
    (['pay_count_group'], [
                            ('pay_price_per_hour','mean'),
                            ('pay_price_per_hour','var'),
                            ('avg_online_minutes', 'mean'),
                            ('pay_count', 'mean'),
                            ('acct_value_zs', 'mean')]),
    (['pay_price_group'], [
                            ('pay_price_per_hour','mean'),
                            ('pay_price_per_hour','var'),
                            ('avg_online_minutes', 'mean'),
                            ('pay_count', 'mean'),
                            ('acct_value_zs', 'mean')]),
]


SOLUTION_CONFIG = AttrDict({
    'pipeline': {'experiment_directory': params.experiment_directory
                 },

    'preprocessing': {'impute_missing': {'fill_missing': params.fill_missing,
                                         'fill_value': params.fill_value},
                      'categorical_encoder': {'categorical_columns': CATEGORICAL_COLUMNS,
                                              'impute_missing': False
                                              },
                      },

    'tap4fun': {'columns': {'categorical_columns': CATEGORICAL_COLUMNS,
                            'numerical_columns': NUMERICAL_COLUMNS
                            },
                'aggregations': {'groupby_aggregations': TAP4FUN_AGGREGATION_RECIPIES
                                 }
                },

    'light_gbm': {'device': parameter_eval(params.lgbm__device),
                  'boosting_type': parameter_eval(params.lgbm__boosting_type),
                  'objective': parameter_eval(params.lgbm__objective),
                  'metric': parameter_eval(params.lgbm__metric),
                  'scale_pos_weight': parameter_eval(params.lgbm__scale_pos_weight),
                  'learning_rate': parameter_eval(params.lgbm__learning_rate),
                  'max_bin': parameter_eval(params.lgbm__max_bin),
                  'max_depth': parameter_eval(params.lgbm__max_depth),
                  'num_leaves': parameter_eval(params.lgbm__num_leaves),
                  'min_child_samples': parameter_eval(params.lgbm__min_child_samples),
                  'subsample': parameter_eval(params.lgbm__subsample),
                  'colsample_bytree': parameter_eval(params.lgbm__colsample_bytree),
                  'subsample_freq': parameter_eval(params.lgbm__subsample_freq),
                  'min_gain_to_split': parameter_eval(params.lgbm__min_gain_to_split),
                  'reg_lambda': parameter_eval(params.lgbm__reg_lambda),
                  'reg_alpha': parameter_eval(params.lgbm__reg_alpha),
                  'nthread': parameter_eval(params.num_workers),
                  'is_unbalance': parameter_eval(params.lgbm__is_unbalance),
                  'number_boosting_rounds': parameter_eval(params.lgbm__number_boosting_rounds),
                  'early_stopping_rounds': parameter_eval(params.lgbm__early_stopping_rounds),
                  'verbose': parameter_eval(params.verbose),
                  },

    'xgboost': {'booster': parameter_eval(params.xgb__booster),
                'objective': parameter_eval(params.xgb__objective),
                'tree_method': parameter_eval(params.xgb__tree_method),
                'eval_metric': parameter_eval(params.xgb__eval_metric),
                'eta': parameter_eval(params.xgb__eta),
                'max_depth': parameter_eval(params.xgb__max_depth),
                'subsample': parameter_eval(params.xgb__subsample),
                'colsample_bytree': parameter_eval(params.xgb__colsample_bytree),
                'colsample_bylevel': parameter_eval(params.xgb__colsample_bylevel),
                'min_child_weight': parameter_eval(params.xgb__min_child_weight),
                'lambda': parameter_eval(params.xgb__lambda),
                'alpha': parameter_eval(params.xgb__alpha),
                'max_bin': parameter_eval(params.xgb__max_bin),
                'num_leaves': parameter_eval(params.xgb__max_leaves),
                'nthread': parameter_eval(params.num_workers),
                'nrounds': parameter_eval(params.xgb__nrounds),
                'early_stopping_rounds': parameter_eval(params.xgb__early_stopping_rounds),
                'verbose': parameter_eval(params.verbose)
                },

})

if is_regression_problem:
    SOLUTION_CONFIG.update({
        'random_search': {'light_gbm': {'n_runs': params.lgbm_random_search_runs,
                                        'callbacks':
                                            {'neptune_monitor': {'name': 'light_gbm'},
                                             'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                          'random_search_light_gbm.pkl')}
                                             },
                                        },
                          'xgboost': {'n_runs': params.xgb_random_search_runs,
                                      'callbacks':
                                      {'neptune_monitor': {'name': 'xgboost'},
                                       'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                    'random_search_xgboost.pkl')}
                                       },
                                    },
                            'lasso': {'n_runs': params.lasso_random_search_runs,
                                      'callbacks':
                                      {'neptune_monitor': {'name': 'lasso'},
                                       'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                    'random_search_lasso.pkl')}
                                       },
                                    },
                            'ridge': {'n_runs': params.ridge_random_search_runs,
                                      'callbacks':
                                      {'neptune_monitor': {'name': 'ridge'},
                                       'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                    'random_search_ridge.pkl')}
                                       },
                                    },
                          },
        'lasso': {
                'tol': params.lasso__tol
        },
        'ridge': {
                'tol': params.ridge__tol
        },
    })
else:
    SOLUTION_CONFIG.update({
        'random_forest': {'n_estimators': parameter_eval(params.rf__n_estimators),
                          'criterion': parameter_eval(params.rf__criterion),
                          'warm_start': parameter_eval(params.rf__warm_start),
                          'max_features': parameter_eval(params.rf__max_features),
                          'min_samples_split': parameter_eval(params.rf__min_samples_split),
                          'min_samples_leaf': parameter_eval(params.rf__min_samples_leaf),
                          'n_jobs': parameter_eval(params.num_workers),
                          'random_state': RANDOM_SEED,
                          'verbose': parameter_eval(params.verbose),
                          'class_weight': parameter_eval(params.rf__class_weight),
                          },
        'gradient_boost' : {
                            'n_estimators': parameter_eval(params.gb__n_estimators),
                            'learning_rate': parameter_eval(params.gb__learning_rate),
                            'subsample': parameter_eval(params.gb__subsample),
                            'max_features': parameter_eval(params.gb__max_features),
                            'verbose': parameter_eval(params.verbose),
                            # 'n_jobs': parameter_eval(params.num_workers),
                            },
        'logistic_regression': {'penalty': parameter_eval(params.lr__penalty),
                                'tol': parameter_eval(params.lr__tol),
                                'C': parameter_eval(params.lr__C),
                                'fit_intercept': parameter_eval(params.lr__fit_intercept),
                                'class_weight': parameter_eval(params.lr__class_weight),
                                'random_state': RANDOM_SEED,
                                'solver': parameter_eval(params.lr__solver),
                                'max_iter': parameter_eval(params.lr__max_iter),
                                'verbose': parameter_eval(params.verbose),
                                'n_jobs': parameter_eval(params.num_workers),
                                },

        'svc': {# 'kernel': parameter_eval(params.svc__kernel),
                'C': parameter_eval(params.svc__C),
                # 'degree': parameter_eval(params.svc__degree),
                # 'gamma': parameter_eval(params.svc__gamma),
                # 'coef0': parameter_eval(params.svc__coef0),
                # 'probability': parameter_eval(params.svc__probability),
                'tol': parameter_eval(params.svc__tol),
                'max_iter': parameter_eval(params.svc__max_iter),
                'verbose': parameter_eval(params.verbose),
                'random_state': RANDOM_SEED,
                },
        'adaboost': {'random_state': RANDOM_SEED},
        'random_search': {'light_gbm': {'n_runs': params.lgbm_random_search_runs,
                                        'callbacks':
                                            {'neptune_monitor': {'name': 'light_gbm'},
                                             'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                          'random_search_light_gbm.pkl')}
                                             },
                                        },
                          'xgboost': {'n_runs': params.xgb_random_search_runs,
                                      'callbacks':
                                      {'neptune_monitor': {'name': 'xgboost'},
                                       'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                    'random_search_xgboost.pkl')}
                                       },
                                      },
                          'random_forest': {'n_runs': params.rf_random_search_runs,
                                            'callbacks':
                                                {'neptune_monitor': {'name': 'random_forest'},
                                                 'persist_results':
                                                    {'filepath': os.path.join(params.experiment_directory,
                                                                              'random_search_random_forest.pkl')}
                                                 },
                                            },
                          'gradient_boost': {'n_runs': params.gb_random_search_runs,
                                  'callbacks': {'neptune_monitor': {'name': 'gradient_boost'},
                                                'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                             'random_search_gradient_boost.pkl')
                                                                    }
                                            },
                            },
                          'logistic_regression': {'n_runs': params.lr_random_search_runs,
                                                  'callbacks':
                                                  {'neptune_monitor': {'name': 'logistic_regression'},
                                                   'persist_results':
                                                   {'filepath': os.path.join(params.experiment_directory,
                                                                             'random_search_logistic_regression.pkl')}
                                                   },
                                                  },
                          'svc': {'n_runs': params.svc_random_search_runs,
                                  'callbacks': {'neptune_monitor': {'name': 'svc'},
                                                'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                             'random_search_svc.pkl')}
                                                },
                                  },
                          },
    })
