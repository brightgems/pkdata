from functools import partial

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from steppy.adapter import Adapter, E
from steppy.base import Step, make_transformer
from .models import AvgTransformer
from . import pipeline_config as cfg
from .pipeline_blocks import (classifier_light_gbm,
                              classifier_select_best_threshold,
                              classifier_sklearn, classifier_xgb,
                              feature_extraction, preprocessing_fillna)


def get_pipeline(pipeline_name, train_mode):
    if isinstance(PIPELINES[pipeline_name], dict):
        pipeline = PIPELINES[pipeline_name]['train'](config=cfg.SOLUTION_CONFIG)
    else:
        pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=True)
    return pipeline

def ensemble(config, train_mode,suffix="", **kwargs):
    lgb_step = get_pipeline('lightGBM', train_mode)
    logreg_step = get_pipeline('log_reg', train_mode)
    rf_step = get_pipeline('log_reg', train_mode)
    ens_step = Step(name='Ensembler',
                transformer=AvgTransformer(),
                input_steps=[lgb_step,logreg_step, rf_step],
                adapter=Adapter({'y_proba_1': E(lgb_step.name, 'prediction'),
                                 'y_proba_2': E(rf_step.name, 'prediction'),
                                 'y_proba_3': E(logreg_step.name, 'prediction'),
                                }),
                experiment_directory=config.pipeline.experiment_directory,)
    return ens_step


def lightGBM(config, train_mode, suffix=''):
    if train_mode:
        features, features_valid = feature_extraction(config,
                                                      train_mode,
                                                      suffix,
                                                      persist_output=False,
                                                      cache_output=False,
                                                      load_persisted_output=False)
        light_gbm = classifier_light_gbm((features, features_valid),
                                         config,
                                         train_mode, suffix)
    else:
        features = feature_extraction(config,
                                      train_mode,
                                      suffix)
        light_gbm = classifier_light_gbm(features,
                                         config,
                                         train_mode, suffix)
    if cfg.is_regression_problem:
        return light_gbm

    select_threshold = classifier_select_best_threshold(light_gbm, config, train_mode, suffix)
    return select_threshold


def xgboost(config, train_mode, suffix=''):
    if train_mode:
        features, features_valid = feature_extraction(config,
                                                      train_mode,
                                                      suffix,
                                                      persist_output=False,
                                                      cache_output=False,
                                                      load_persisted_output=False)
        xgb = classifier_xgb((features, features_valid),
                             config,
                             train_mode,
                             suffix)
    else:
        features = feature_extraction(config,
                                      train_mode,
                                      suffix,
                                      cache_output=True)
        xgb = classifier_xgb(features,
                             config,
                             train_mode,
                             suffix)
    if cfg.is_regression_problem:
        return xgb
    select_threshold = classifier_select_best_threshold(xgb, config, train_mode, suffix)
    return select_threshold


def sklearn_main(config, ClassifierClass, clf_name, train_mode, suffix='', normalize=False):
    model_params = getattr(config, clf_name)
    random_search_config = getattr(config.random_search, clf_name)
    full_config = (config, model_params, random_search_config)
    if train_mode:
        features, features_valid = feature_extraction(config,
                                                      train_mode,
                                                      suffix,
                                                      persist_output=False,
                                                      cache_output=False,
                                                      load_persisted_output=False)

        sklearn_preproc = preprocessing_fillna((features, features_valid), config, train_mode, suffix)
    else:
        features = feature_extraction(config,
                                      train_mode,
                                      suffix)
        sklearn_preproc = preprocessing_fillna(features, config, train_mode, suffix)

    sklearn_clf = classifier_sklearn(sklearn_preproc,
                                     ClassifierClass,
                                     full_config,
                                     clf_name,
                                     train_mode,
                                     suffix,
                                     normalize)
    return sklearn_clf


PIPELINES = {'lightGBM': lightGBM,
             'XGBoost': xgboost,
             'ensemble': ensemble,
             'random_forest': {'train': partial(sklearn_main,
                                                ClassifierClass=RandomForestClassifier,
                                                clf_name='random_forest',
                                                train_mode=True),
                               'inference': partial(sklearn_main,
                                                    ClassifierClass=RandomForestClassifier,
                                                    clf_name='random_forest',
                                                    train_mode=False)
                               },
             'log_reg': {'train': partial(sklearn_main,
                                          ClassifierClass=LogisticRegression,
                                          clf_name='logistic_regression',
                                          train_mode=True,
                                          normalize=True),
                         'inference': partial(sklearn_main,
                                              ClassifierClass=LogisticRegression,
                                              clf_name='logistic_regression',
                                              train_mode=False,
                                              normalize=True)
                         },
             'gradient_boost': {
                         'train': partial(sklearn_main,
                                          ClassifierClass=GradientBoostingClassifier,
                                          clf_name='gradient_boost',
                                          train_mode=True,
                                          normalize=True),
                         'inference': partial(sklearn_main,
                                              ClassifierClass=GradientBoostingClassifier,
                                              clf_name='gradient_boost',
                                              train_mode=False,
                                              normalize=True)
             },
             'svc': {'train': partial(sklearn_main,
                                      ClassifierClass=SVC,
                                      clf_name='svc',
                                      train_mode=True,
                                      normalize=True),
                     'inference': partial(sklearn_main,
                                          ClassifierClass=SVC,
                                          clf_name='svc',
                                          train_mode=False,
                                          normalize=True)
                     }
             }
