from functools import partial

from steppy.adapter import Adapter, E
from steppy.base import Step, make_transformer
from . import feature_extraction as fe
from . import data_cleaning as dc
from .hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, PersistResults
from .models import get_sklearn_classifier, XGBoost, LightGBM, ThresholdSelection, AvgTransformer
from .pipeline_config import params, score_function


def classifier_select_best_threshold(model_step, config, train_mode,suffix, **kwargs):
    if train_mode:
        select_best_threshold = Step(name='select_best_threshold{}'.format(suffix),
                                transformer=ThresholdSelection(),
                                input_data=['tap4fun'],
                                input_steps=[model_step],
                                is_trainable= True,
                                adapter=Adapter({'prediction': E(model_step.name, 'prediction'),
                                                 'actual': E('tap4fun', 'y')
                                                }),
                                experiment_directory=config.pipeline.experiment_directory,
                                **kwargs)
    else:
        select_best_threshold = Step(name='select_best_threshold{}'.format(suffix),
                                transformer=ThresholdSelection(),
                                input_steps=[model_step],
                                adapter=Adapter({'prediction': E(model_step.name, 'prediction')}),
                                experiment_directory=config.pipeline.experiment_directory,
                                **kwargs)
    return select_best_threshold


def classifier_light_gbm(features, config, train_mode, suffix, **kwargs):
    model_name = 'light_gbm{}'.format(suffix)

    if train_mode:
        features_train, features_valid = features
        if config.random_search.light_gbm.n_runs:
            transformer = RandomSearchOptimizer(TransformerClass=LightGBM,
                                                params=config.light_gbm,
                                                train_input_keys=[],
                                                valid_input_keys=['X_valid', 'y_valid'],
                                                score_func=score_function,
                                                maximize=True,
                                                n_runs=config.random_search.light_gbm.n_runs,
                                                callbacks=[
                                                    NeptuneMonitor(
                                                        **config.random_search.light_gbm.callbacks.neptune_monitor),
                                                    PersistResults(
                                                        **config.random_search.light_gbm.callbacks.persist_results)]
                                                )
        else:
            transformer = LightGBM(name=model_name, **config.light_gbm)

        light_gbm = Step(name=model_name,
                         transformer=transformer,
                         input_data=['tap4fun'],
                         input_steps=[features_train, features_valid],
                         adapter=Adapter({'X': E(features_train.name, 'features'),
                                          'y': E('tap4fun', 'y'),
                                          'feature_names': E(features_train.name, 'feature_names'),
                                          'categorical_features': E(features_train.name, 'categorical_features'),
                                          'X_valid': E(features_valid.name, 'features'),
                                          'y_valid': E('tap4fun', 'y_valid'),
                                          }),
                         experiment_directory=config.pipeline.experiment_directory,
                         is_trainable= True,
                         **kwargs)
    else:
        light_gbm = Step(name=model_name,
                         transformer=LightGBM(name=model_name, **config.light_gbm),
                         input_steps=[features],
                         is_trainable= True,
                         adapter=Adapter({'X': E(features.name, 'features')}),
                         experiment_directory=config.pipeline.experiment_directory,
                         **kwargs)

    
    return light_gbm


def classifier_xgb(features, config, train_mode, suffix, **kwargs):
    if train_mode:
        features_train, features_valid = features
        if config.random_search.xgboost.n_runs:
            transformer = RandomSearchOptimizer(TransformerClass=XGBoost,
                                                params=config.xgboost,
                                                train_input_keys=[],
                                                valid_input_keys=['X_valid', 'y_valid'],
                                                score_func=score_function,
                                                maximize=True,
                                                n_runs=config.random_search.xgboost.n_runs,
                                                callbacks=[
                                                    NeptuneMonitor(
                                                        **config.random_search.xgboost.callbacks.neptune_monitor),
                                                    PersistResults(
                                                        **config.random_search.xgboost.callbacks.persist_results)]
                                                )
        else:
            transformer = XGBoost(**config.xgboost)

        xgboost = Step(name='xgboost{}'.format(suffix),
                       transformer=transformer,
                       is_trainable= True,
                       input_data=['tap4fun'],
                       input_steps=[features_train, features_valid],
                       adapter=Adapter({'X': E(features_train.name, 'features'),
                                        'y': E('tap4fun', 'y'),
                                        'feature_names': E(features_train.name, 'feature_names'),
                                        'X_valid': E(features_valid.name, 'features'),
                                        'y_valid': E('tap4fun', 'y_valid'),
                                        }),
                       experiment_directory=config.pipeline.experiment_directory,
                       **kwargs)
    else:
        xgboost = Step(name='xgboost{}'.format(suffix),
                       transformer=XGBoost(**config.xgboost),
                       is_trainable= True,
                       input_steps=[features],
                       adapter=Adapter({'X': E(features.name, 'features')}),
                       experiment_directory=config.pipeline.experiment_directory,
                       **kwargs)
    return xgboost


def classifier_sklearn(sklearn_features,
                       ClassifierClass,
                       full_config,
                       clf_name,
                       train_mode,
                       suffix,
                       normalize,
                       **kwargs):
    config, model_params, rs_config = full_config
    if train_mode:
        if config.random_search.random_forest.n_runs:
            transformer = RandomSearchOptimizer(
                partial(get_sklearn_classifier,
                        ClassifierClass=ClassifierClass,
                        normalize=normalize),
                model_params,
                train_input_keys=[],
                valid_input_keys=['X_valid', 'y_valid'],
                score_func=score_function,
                maximize=True,
                n_runs=rs_config.n_runs,
                callbacks=[NeptuneMonitor(**rs_config.callbacks.neptune_monitor),
                           PersistResults(**rs_config.callbacks.persist_results)]
            )
        else:
            transformer = get_sklearn_classifier(ClassifierClass, normalize, **model_params)

        sklearn_clf = Step(name='{}{}'.format(clf_name, suffix),
                           transformer=transformer,
                           is_trainable= True,
                           input_data=['tap4fun'],
                           input_steps=[sklearn_features],
                           adapter=Adapter({'X': E(sklearn_features.name, 'X'),
                                            'y': E('tap4fun', 'y'),
                                            'X_valid': E(sklearn_features.name, 'X_valid'),
                                            'y_valid': E('tap4fun', 'y_valid'),
                                            }),
                           experiment_directory=config.pipeline.experiment_directory,
                           **kwargs)
    else:
        sklearn_clf = Step(name='{}{}'.format(clf_name, suffix),
                           transformer=get_sklearn_classifier(ClassifierClass, normalize, **model_params),
                           is_trainable= True,
                           input_steps=[sklearn_features],
                           adapter=Adapter({'X': E(sklearn_features.name, 'X')}),
                           experiment_directory=config.pipeline.experiment_directory,
                           **kwargs)
    return sklearn_clf


def feature_extraction(config, train_mode, suffix, **kwargs):
    if train_mode:
        tap4fun, tap4fun_valid = _tap4fun(config, train_mode, suffix, **kwargs)

        tap4fun_agg, tap4fun_agg_valid = _tap4fun_groupby_agg(config, train_mode, suffix, **kwargs)

        categorical_encoder, categorical_encoder_valid = _categorical_encoders(config, train_mode, suffix, **kwargs)

        feature_combiner, feature_combiner_valid = _join_features(
            numerical_features=[tap4fun,
                                tap4fun_agg,
                                ],
            numerical_features_valid=[tap4fun_valid,
                                      tap4fun_agg_valid,
                                      ],
            categorical_features=[categorical_encoder
                                  ],
            categorical_features_valid=[categorical_encoder_valid
                                        ],
            config=config,
            train_mode=train_mode,
            suffix=suffix,
            **kwargs)
        
        return feature_combiner, feature_combiner_valid
    else:
        tap4fun = _tap4fun(config, train_mode, suffix, **kwargs)

        tap4fun_agg = _tap4fun_groupby_agg(config, train_mode, suffix, **kwargs)
        categorical_encoder = _categorical_encoders(config, train_mode, suffix, **kwargs)
        feature_combiner = _join_features(numerical_features=[tap4fun,
                                                              tap4fun_agg,
                                                              ],
                                          numerical_features_valid=[],
                                          categorical_features=[categorical_encoder
                                                                ],
                                          categorical_features_valid=[],
                                          config=config,
                                          train_mode=train_mode,
                                          suffix=suffix,
                                          **kwargs)

        return feature_combiner


def preprocessing_fillna(features, config, train_mode, suffix, **kwargs):
    """
        impute missing value by condition
    """
    if train_mode:
        features_train, features_valid = features
        fillna = Step(name='fillna{}'.format(suffix),
                      transformer=_fillna(config.preprocessing.impute_missing.fill_value),
                      input_steps=[features_train, features_valid],
                      adapter=Adapter({'X': E(features_train.name, 'features'),
                                       'X_valid': E(features_valid.name, 'features'),
                                       }),
                      experiment_directory=config.pipeline.experiment_directory,
                      **kwargs
                      )
    else:
        fillna = Step(name='fillna{}'.format(suffix),
                      transformer=_fillna(config.preprocessing.impute_missing.fill_value),
                      input_steps=[features],
                      adapter=Adapter({'X': E(features.name, 'features')}),
                      experiment_directory=config.pipeline.experiment_directory,
                      **kwargs
                      )
    return fillna


def _join_features(numerical_features,
                   numerical_features_valid,
                   categorical_features,
                   categorical_features_valid,
                   config, train_mode, suffix,
                   **kwargs):
    if train_mode:
        persist_output = True
        cache_output = True
        load_persisted_output = True
    else:
        persist_output = False
        cache_output = True
        load_persisted_output = False

    feature_joiner = Step(name='feature_joiner{}'.format(suffix),
                          transformer=fe.FeatureJoiner(),
                          input_steps=numerical_features + categorical_features,
                          adapter=Adapter({
                              'numerical_feature_list': [
                                  E(feature.name, 'numerical_features') for feature in numerical_features],
                              'categorical_feature_list': [
                                  E(feature.name, 'categorical_features') for feature in categorical_features],
                          }),
                          experiment_directory=config.pipeline.experiment_directory,
                          persist_output=persist_output,
                          cache_output=cache_output,
                          load_persisted_output=load_persisted_output)
    if train_mode:
        feature_joiner_valid = Step(name='feature_joiner_valid{}'.format(suffix),
                                    transformer=feature_joiner.transformer,
                                    input_steps=numerical_features_valid + categorical_features_valid,
                                    adapter=Adapter({
                                        'numerical_feature_list': [
                                            E(feature.name,
                                              'numerical_features') for feature in numerical_features_valid],
                                        'categorical_feature_list': [
                                            E(feature.name,
                                              'categorical_features') for feature in categorical_features_valid],
                                    }),
                                    experiment_directory=config.pipeline.experiment_directory,
                                    persist_output=persist_output,
                                    cache_output=cache_output,
                                    load_persisted_output=load_persisted_output)

        return feature_joiner, feature_joiner_valid

    else:
        return feature_joiner


def _categorical_encoders(config, train_mode, suffix, **kwargs):
    categorical_encoder = Step(name='categorical_encoder{}'.format(suffix),
                               transformer=fe.CategoricalEncoder(**config.preprocessing.categorical_encoder),
                               input_data=['tap4fun'],
                               is_trainable = True,
                               adapter=Adapter({'X': E('tap4fun', 'X'),
                                                'y': E('tap4fun', 'y')}
                                               ),
                               experiment_directory=config.pipeline.experiment_directory,
                               **kwargs)
    if train_mode:
        categorical_encoder_valid = Step(name='categorical_encoder_valid{}'.format(suffix),
                                         transformer=categorical_encoder.transformer,
                                         input_data=['tap4fun'],
                                         is_trainable = True,
                                         adapter=Adapter(
                                             {'X': E('tap4fun', 'X_valid'),
                                              'y': E('tap4fun', 'y_valid')}
                                         ),
                                         experiment_directory=config.pipeline.experiment_directory,
                                         **kwargs)
        return categorical_encoder, categorical_encoder_valid
    else:
        return categorical_encoder


def _tap4fun_groupby_agg(config, train_mode, suffix, **kwargs):
    tap4fun_groupby_agg = Step(name='tap4fun_groupby_agg{}'.format(suffix),
                                   transformer=fe.GroupbyAggregate(**config.tap4fun.aggregations),
                                   is_trainable=True,
                                   input_data=['tap4fun'],
                                   adapter=Adapter(
                                       {'main_table': E('tap4fun', 'X')}),
                                   experiment_directory=config.pipeline.experiment_directory,
                                   **kwargs)

    if train_mode:

        tap4fun_groupby_agg_valid = Step(name='tap4fun_groupby_agg_valid{}'.format(suffix),
                                             transformer=fe.GroupbyAggregate(**config.tap4fun.aggregations),
                                             input_data=['tap4fun'],
                                             is_trainable=True,
                                             adapter=Adapter(
                                                 {'main_table': E('tap4fun', 'X_valid'),
                                                  }),
                                             experiment_directory=config.pipeline.experiment_directory,
                                             **kwargs)

        return tap4fun_groupby_agg, tap4fun_groupby_agg_valid

    else:
        return tap4fun_groupby_agg


def _tap4fun_cleaning(config, train_mode, suffix, **kwargs):
    tap4fun_cleaning = Step(name='tap4fun_cleaning{}'.format(suffix),
                                transformer=dc.Tap4funCleaning(**config.preprocessing.impute_missing),
                                input_data=['tap4fun'],
                                adapter=Adapter({'X': E('tap4fun', 'X')}),
                                experiment_directory=config.pipeline.experiment_directory,
                                )
    if train_mode:
        tap4fun_cleaning_valid = Step(name='tap4fun_cleaning_valid{}'.format(suffix),
                                          transformer=tap4fun_cleaning.transformer,
                                          input_data=['tap4fun'],
                                          adapter=Adapter({'X': E('tap4fun', 'X_valid')}),
                                          experiment_directory=config.pipeline.experiment_directory,
                                          )
        return tap4fun_cleaning, tap4fun_cleaning_valid
    else:
        return tap4fun_cleaning


def _tap4fun(config, train_mode, suffix, **kwargs):
    if train_mode:
        tap4fun_cleaning, tap4fun_cleaning_valid = _tap4fun_cleaning(config, train_mode, suffix, **kwargs)
    else:
        tap4fun_cleaning = _tap4fun_cleaning(config, train_mode, suffix, **kwargs)

    tap4fun = Step(name='tap4fun_hand_crafted{}'.format(suffix),
                       transformer=fe.Tap4funFeatures(**config.tap4fun.columns),
                       input_steps=[tap4fun_cleaning],
                       adapter=Adapter({'X': E(tap4fun_cleaning.name, 'X')}),
                       experiment_directory=config.pipeline.experiment_directory,
                       **kwargs)
    if train_mode:
        tap4fun_valid = Step(name='tap4fun_hand_crafted_valid{}'.format(suffix),
                                 transformer=tap4fun.transformer,
                                 input_steps=[tap4fun_cleaning_valid],
                                 adapter=Adapter({'X': E(tap4fun_cleaning_valid.name, 'X')}),
                                 experiment_directory=config.pipeline.experiment_directory,
                                 **kwargs)
        return tap4fun, tap4fun_valid
    else:
        return tap4fun


def _fillna(fill_value):
    def _inner_fillna(X, X_valid=None):
        if X_valid is None:
            return {'X': X.fillna(fill_value)}
        else:
            return {'X': X.fillna(fill_value),
                    'X_valid': X_valid.fillna(fill_value)}

    return make_transformer(_inner_fillna)                    