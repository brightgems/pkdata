import os

from attrdict import AttrDict
from deepsense import neptune
from sklearn.metrics import roc_auc_score
from .utils import read_params, parameter_eval, rmse

ctx = neptune.Context()
params = read_params(ctx, fallback_file='./neptune_clf.yaml')

RANDOM_SEED = 90210
DEV_SAMPLE_SIZE = 5000

ID_COLUMNS = ['user_id']
is_regression_problem = params.is_regression_problem
if is_regression_problem:
    TARGET_COLUMNS = ['prediction_future_pay_price']
    score_function = rmse
    score_name = 'RMSE'
else:    
    TARGET_COLUMNS = ['is_future_pay']
    score_function = roc_auc_score
    score_name = 'ROC_AUC'

CATEGORICAL_COLUMNS = [
    'pay_price_group',
    'register_timecls',
    'avg_online_minutes_group',
    'pay_count_group',
    'avg_pay_group',
]

NUMERICAL_COLUMNS = [
    # active/payment features
    'avg_online_minutes', 'pay_price', 'pay_price_per_hour'
    # war features
    'battle_count',
    'battle_win%'
    # achievement features
    'acct_value',
    'acct_value_zs',
    'sr_main_score', 'sr_level_score',
    'bd_main_score', 'army_add', 'resource_reduce',
    # net features
    'ivory_net',
    'magic_net',
    'meat_net',
    'stone_net',
    'wood_net',
    'cavalry_net',
    'infantry_net',
    'shaman_net',
    'wound_cavalry_net',
    'wound_infantry_net',
    'wound_shaman_net',
    'building_acceleration_net',
    'general_acceleration_net',
    'reaserch_acceleration_net',
    'training_acceleration_net',
    # mix features
    'army_reduce_per_battle',
    'acceleration_army_ratio',
    'sr_army_prod',
    'bd_sr_ratio',
    'bd_sr_product',
    'acceleration_army_reduce_ratio', 
    'resource_army_reduce_ratio', 
    'acceleration_resource_reduce_ratio', 
    'acceleration_army_add_ratio', 
    'resource_army_add_ratio', 
    'acceleration_add_add_ratio', 
    'acceleration_per_hour', 
    'sr_level_score_per_hour', 
    'army_per_hour', 
    'bd_resource_ratio'
]

USELESS_COLUMNS = [
    'prediction_pay_price',
    'prediction_future_pay_price',
    'user_id',
]

TAP4FUN_AGGREGATION_RECIPIES = [
    (['pay_price_group'], [('avg_online_minutes', 'mean'),
                           ('pay_count', 'mean'),
                           ('battle_count', 'mean'),
                           ('acct_value', 'mean')]),
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

        'svc': {'kernel': parameter_eval(params.svc__kernel),
                'C': parameter_eval(params.svc__C),
                'degree': parameter_eval(params.svc__degree),
                'gamma': parameter_eval(params.svc__gamma),
                'coef0': parameter_eval(params.svc__coef0),
                'probability': parameter_eval(params.svc__probability),
                'tol': parameter_eval(params.svc__tol),
                'max_iter': parameter_eval(params.svc__max_iter),
                'verbose': parameter_eval(params.verbose),
                'random_state': RANDOM_SEED,
                },

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
