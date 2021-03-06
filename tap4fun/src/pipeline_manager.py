# coding: utf-8
import os
import shutil
import random
from attrdict import AttrDict
import numpy as np
import pandas as pd
from scipy.stats import gmean
from deepsense import neptune
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, recall_score, roc_auc_score, \
    precision_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import requests
from . import pipeline_config as cfg
from .pipeline_config import score_name, score_function
from .pipelines import PIPELINES
from .utils import \
    (compress_dtypes, init_logger, set_seed, make_submission, create_submission, 
     split_train_evaluate, verify_submission, calculate_rank)
from .preprocess import prepare_dataset

set_seed(cfg.RANDOM_SEED)
logger = init_logger()
ctx = neptune.Context()
params = cfg.params

def print_score(pipeline_name, y_true, y_pred):
    score = score_function(y_true, y_pred)
    logger.info('{} score on validation is {}'.format(score_name, score))
    ctx.channel_send(score_name, 0, score)
    if not cfg.is_regression_problem:
        # convert probablity to binary
        y_pred = y_pred>.4
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision= precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        logger.info("{0}: f1={1},accuracy={2},precision={3},recall={4}".format(
            pipeline_name, f1,accuracy,precision,recall))
        ctx.channel_send('f1', 0, f1)
        ctx.channel_send('accuracy', 0, accuracy)
        ctx.channel_send('precision', 0, precision)
        ctx.channel_send('recall', 0, recall)
        print(confusion_matrix(y_true, y_pred))

class PipelineManager():
    def train(self, pipeline_name, dev_mode):
        train(pipeline_name, dev_mode)

    def evaluate(self, pipeline_name, dev_mode, ):
        evaluate(pipeline_name, dev_mode)

    def predict(self, pipeline_name, dev_mode, submit_predictions):
        predict(pipeline_name, dev_mode, submit_predictions)

    def train_evaluate_cv(self, pipeline_name, dev_mode):
        train_evaluate_cv(pipeline_name, dev_mode)

    def train_evaluate_predict_cv(self, pipeline_name, dev_mode, submit_predictions):
        train_evaluate_predict_cv(pipeline_name, dev_mode, submit_predictions)

    def submit_result(self):
        make_submission('','./workdir/submission.csv')

    def prepare_dataset(self):
        prepare_dataset()

def train(pipeline_name, dev_mode):
    logger.info('TRAINING')
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    tables = _read_data(dev_mode, read_train=True, read_test=False)

    logger.info('Shuffling and splitting into train and test...')
    # train_data_split, valid_data_split = train_test_split(tables.train,
    #                                                       test_size=params.validation_size,
    #                                                       random_state=cfg.RANDOM_SEED,
    #                                                       shuffle=params.shuffle)
    train_data_split, valid_data_split = split_train_evaluate(tables.train, 0.78)
    logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMNS].mean()))
    logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMNS].mean()))
    logger.info('Train shape: {}'.format(train_data_split.shape))
    logger.info('Valid shape: {}'.format(valid_data_split.shape))

    train_data = {'tap4fun': {'X': train_data_split.drop(cfg.TARGET_COLUMNS, axis=1),
                                  'y': train_data_split[cfg.TARGET_COLUMNS].values.reshape(-1),
                                  'X_valid': valid_data_split.drop(cfg.TARGET_COLUMNS, axis=1),
                                  'y_valid': valid_data_split[cfg.TARGET_COLUMNS].values.reshape(-1)
                                  },
                  'variables': {'X':tables.variables}
                  }
    if isinstance(PIPELINES[pipeline_name], dict):
        pipeline = PIPELINES[pipeline_name]['train'](config=cfg.SOLUTION_CONFIG)
    else:
        pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=True)
    pipeline.clean_cache()
    logger.info('Start pipeline fit and transform')
    pipeline.fit_transform(train_data)
    pipeline.clean_cache()


def evaluate(pipeline_name, dev_mode):
    logger.info('EVALUATION')
    logger.info('Reading data...')

    tables = _read_data(dev_mode, read_train=True, read_test=False)

    logger.info('Shuffling and splitting to get validation split...')
    # _, valid_data_split = train_test_split(tables.train,
    #                                        test_size=params.validation_size,
    #                                        random_state=cfg.RANDOM_SEED,
    #                                        shuffle=params.shuffle)
    _, valid_data_split = split_train_evaluate(tables.train, 0.78)
    logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMNS].mean()))
    logger.info('Valid shape: {}'.format(valid_data_split.shape))

    y_true = valid_data_split[cfg.TARGET_COLUMNS].values

    eval_data = {'tap4fun': {'X': valid_data_split.drop(cfg.TARGET_COLUMNS, axis=1),
                                 'y': None,
                                 },
                  'variables': {'X':tables.variables}                                 
                 }
    if isinstance(PIPELINES[pipeline_name], dict):
        pipeline = PIPELINES[pipeline_name]['inference'](config=cfg.SOLUTION_CONFIG)
    else:
        pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=False)
    pipeline.clean_cache()
    logger.info('Start pipeline transform')
    output = pipeline.transform(eval_data)
    pipeline.clean_cache()

    y_pred = output['prediction']
    print_score(pipeline_name, y_true, y_pred)
    

def predict(pipeline_name, dev_mode, submit_predictions):
    logger.info('PREDICTION')

    tables = _read_data(dev_mode, read_train=False, read_test=True)

    test_data = {'tap4fun': {'X': tables.test,
                                 'y': None,
                                 },
                  'variables': {'X':tables.variables}                                 
                 }

    if isinstance(PIPELINES[pipeline_name], dict):
        pipeline = PIPELINES[pipeline_name]['inference'](config=cfg.SOLUTION_CONFIG)
    else:
        pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=False)

    pipeline.clean_cache()
    logger.info('Start pipeline transform')
    output = pipeline.transform(test_data)
    pipeline.clean_cache()
    y_pred = output['prediction']

    if not dev_mode:
        logger.info('creating submission file...')
        submission = create_submission(tables.test, y_pred)

        logger.info('verifying submission...')
        sample_submission = pd.read_csv(params.sample_submission_filepath)
        verify_submission(submission, sample_submission)

        submission_filepath = os.path.join(params.experiment_directory, 'submission.csv')
        submission.to_csv(submission_filepath, index=None, encoding='utf-8')
        logger.info('submission persisted to {}'.format(submission_filepath))
        logger.info('submission head \n\n{}'.format(submission.head()))

        if submit_predictions and params.kaggle_api:
            make_submission(pipeline_name, submission_filepath)


def train_evaluate_cv(pipeline_name, dev_mode):
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    tables = _read_data(dev_mode, read_train=True, read_test=False)

    target_values = tables.train[cfg.TARGET_COLUMNS].values.reshape(-1)
    fold_generator = _get_fold_generator(target_values)

    fold_scores = []
    for fold_id, (train_idx, valid_idx) in enumerate(fold_generator):
        (train_data_split,
         valid_data_split) = tables.train.iloc[train_idx], tables.train.iloc[valid_idx]

        logger.info('Started fold {}'.format(fold_id))
        logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Train shape: {}'.format(train_data_split.shape))
        logger.info('Valid shape: {}'.format(valid_data_split.shape))

        score, _, _ = _fold_fit_evaluate_loop(train_data_split, valid_data_split, tables, fold_id, pipeline_name)

        logger.info('Fold {} {} {}'.format(fold_id, score_name, score))
        ctx.channel_send('Fold {}'.format(score_name), fold_id, score)

        fold_scores.append(score)

    score_mean, score_std = np.mean(fold_scores), np.std(fold_scores)

    logger.info('{}: mean {},std {}'.format(score_name, score_mean, score_std))
    ctx.channel_send(score_name, 0, score_mean)
    ctx.channel_send('{} STD'.format(score_name), 0, score_std)


def train_evaluate_predict_cv(pipeline_name, dev_mode, submit_predictions):
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    tables = _read_data(dev_mode, read_train=True, read_test=True)

    target_values = tables.train[cfg.TARGET_COLUMNS].values.reshape(-1)
    fold_generator = _get_fold_generator(target_values)

    fold_scores, out_of_fold_train_predictions, out_of_fold_test_predictions = [], [], []
    for fold_id, (train_idx, valid_idx) in enumerate(fold_generator):
        (train_data_split,
         valid_data_split) = tables.train.iloc[train_idx], tables.train.iloc[valid_idx]

        logger.info('Started fold {}'.format(fold_id))
        logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Train shape: {}'.format(train_data_split.shape))
        logger.info('Valid shape: {}'.format(valid_data_split.shape))

        score, out_of_fold_prediction, test_prediction = _fold_fit_evaluate_predict_loop(train_data_split,
                                                                                         valid_data_split,
                                                                                         tables,
                                                                                         fold_id, pipeline_name)

        logger.info('Fold {} {} {}'.format(fold_id, score_name, score))
        ctx.channel_send('Fold {}'.format(score_name), fold_id, score)

        out_of_fold_train_predictions.append(out_of_fold_prediction)
        out_of_fold_test_predictions.append(test_prediction)
        fold_scores.append(score)

    out_of_fold_train_predictions = pd.concat(out_of_fold_train_predictions, axis=0)
    out_of_fold_test_predictions = pd.concat(out_of_fold_test_predictions, axis=0)

    test_prediction_aggregated = _aggregate_test_prediction(out_of_fold_test_predictions)
    score_mean, score_std = np.mean(fold_scores), np.std(fold_scores)

    logger.info('{}: mean {}, std {}'.format(score_name, score_mean, score_std))
    ctx.channel_send(score_name, 0, score_mean)
    ctx.channel_send('{} STD'.format(score_name), 0, score_std)

    logger.info('Saving predictions')
    out_of_fold_train_predictions.to_csv(os.path.join(params.experiment_directory,
                                                      '{}_out_of_fold_train_predictions.csv'.format(pipeline_name)),
                                         index=None)
    out_of_fold_test_predictions.to_csv(os.path.join(params.experiment_directory,
                                                     '{}_out_of_fold_test_predictions.csv'.format(pipeline_name)),
                                        index=None)
    test_aggregated_file_path = os.path.join(params.experiment_directory,
                                             '{}_test_predictions_{}.csv'.format(pipeline_name,
                                                                                 params.aggregation_method))
    test_prediction_aggregated.to_csv(test_aggregated_file_path, index=None)

    if not dev_mode:
        logger.info('verifying submission...')
        sample_submission = pd.read_csv(params.sample_submission_filepath)
        verify_submission(test_prediction_aggregated, sample_submission)

        if submit_predictions and params.kaggle_api:
            make_submission(pipeline_name, test_aggregated_file_path)


def _read_frame(filepath,nrows=None):
    if os.path.splitext(filepath)[-1]=='.parquet':
        df = pd.read_parquet(filepath)
    elif os.path.splitext(filepath)[-1]=='.h5':
        df = pd.read_hdf(filepath, stop=nrows)
    elif os.path.splitext(filepath)[-1]=='.xlsx':
        df = pd.read_excel(filepath,sheet_name='Sheet1')
    else:
        df = pd.read_csv(filepath, nrows=nrows)
    return df

def _read_data(dev_mode, read_train=True, read_test=False):
    logger.info('Reading data...')
    if dev_mode:
        nrows = cfg.DEV_SAMPLE_SIZE
        logger.info('running in "dev-mode". Sample size is: {}'.format(cfg.DEV_SAMPLE_SIZE))
    else:
        nrows = None

    raw_data = {}

    if read_train:
        df = _read_frame(params.train_filepath, nrows=nrows)
        df = _preprocess_target_feature(df)
        # 只保留付费用户
        # df = df[df['pay_price']>0].reset_index()
        df = df[df['avg_online_minutes']>20].reset_index()
        raw_data['train'] = df
    if read_test:
        df = _read_frame(params.test_filepath, nrows=nrows)
        df = _preprocess_target_feature(df)
        raw_data['test'] = df

    raw_data['variables'] = _read_frame(params.variable_filepath)
    return AttrDict(raw_data)


def _preprocess_target_feature(df):
    logger.info('Preprocess data...')
    if 'prediction_pay_price' in df.columns:
        df['prediction_future_pay_price'] = df['prediction_pay_price'] - df['pay_price']
        df = df.assign(is_future_no_pay= lambda x: x.prediction_future_pay_price<1)
        if cfg.is_regression_problem:
            drop_cols = ['prediction_pay_price','is_future_no_pay']
        else:
            drop_cols = ['prediction_pay_price','prediction_future_pay_price']
    # =========make categical features=========
    df['pay_price_group']=pd.cut( df['pay_price'],[-0.01,0,1,10,100,1000,5000,10000])
    df['pay_count_group']=pd.cut( df['pay_count'],[-1,0,1,2,3,4,200])
    df['avg_online_minutes_group'] = pd.cut(df['avg_online_minutes'], 
            [-0.001,0,2,5,20,60,120,300,1200,2400],include_lowest=True
    )
    # time class
    def map_to_timecls(hour):
        if hour<=7:
            return 'night'
        elif hour<=12:
            return 'morning'
        elif hour<=18:
            return 'afternoon'
        else:
            return 'evening'
    df['register_timecls'] = df.assign(
            register_timecls = lambda df_all: df_all['register_time'].dt.hour
        )['register_timecls'].map(map_to_timecls)
    
    df = compress_dtypes(df)
    return df

def _get_fold_generator(target_values):
    if params.stratified_cv:
        cv = StratifiedKFold(n_splits=params.n_cv_splits, shuffle=True, random_state=cfg.RANDOM_SEED)
        cv.get_n_splits(target_values)
        fold_generator = cv.split(target_values, target_values)
    else:
        cv = KFold(n_splits=params.n_cv_splits, shuffle=True, random_state=cfg.RANDOM_SEED)
        fold_generator = cv.split(target_values)
    return fold_generator


def _fold_fit_evaluate_predict_loop(train_data_split, valid_data_split, tables, fold_id, pipeline_name):
    score, y_valid_pred, pipeline = _fold_fit_evaluate_loop(train_data_split, valid_data_split, tables,
                                                            fold_id, pipeline_name)

    test_data = {'tap4fun': {'X': tables.test,
                                 'y': None,
                                 },
                  'variables': {'X':tables.variables}                                 
                 }

    logger.info('Start pipeline transform on test')
    pipeline.clean_cache()
    output_test = pipeline.transform(test_data)
    pipeline.clean_cache()
    y_test_pred = output_test['prediction']

    train_out_of_fold_prediction_chunk = valid_data_split[cfg.ID_COLUMNS]
    train_out_of_fold_prediction_chunk['fold_id'] = fold_id
    train_out_of_fold_prediction_chunk['{}_prediction'.format(pipeline_name)] = y_valid_pred

    test_out_of_fold_prediction_chunk = tables.test[cfg.ID_COLUMNS]
    test_out_of_fold_prediction_chunk['fold_id'] = fold_id
    test_out_of_fold_prediction_chunk['{}_prediction'.format(pipeline_name)] = y_test_pred

    return score, train_out_of_fold_prediction_chunk, test_out_of_fold_prediction_chunk


def _fold_fit_evaluate_loop(train_data_split, valid_data_split, tables, fold_id, pipeline_name):
    train_data = {'tap4fun': {'X': train_data_split.drop(cfg.TARGET_COLUMNS, axis=1),
                                  'y': train_data_split[cfg.TARGET_COLUMNS].values.reshape(-1),
                                  'X_valid': valid_data_split.drop(cfg.TARGET_COLUMNS, axis=1),
                                  'y_valid': valid_data_split[cfg.TARGET_COLUMNS].values.reshape(-1),
                                  },
                  'variables': {'X':tables.variables}                                  
                  }

    valid_data = {'tap4fun': {'X': valid_data_split.drop(cfg.TARGET_COLUMNS, axis=1),
                                  'y': None,
                                  },
                  'variables': {'X':tables.variables}                                  
                  }
    if isinstance(PIPELINES[pipeline_name], dict):
        pipeline = PIPELINES[pipeline_name]['train'](config=cfg.SOLUTION_CONFIG,
                                            suffix='_fold_{}'.format(fold_id))
    else:
        pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=True,
                                            suffix='_fold_{}'.format(fold_id))

    logger.info('Start pipeline fit and transform on train')
    pipeline.clean_cache()
    pipeline.fit_transform(train_data)
    pipeline.clean_cache()
    
    if isinstance(PIPELINES[pipeline_name], dict):
        pipeline = PIPELINES[pipeline_name]['inference'](config=cfg.SOLUTION_CONFIG,
                                        suffix='_fold_{}'.format(fold_id))
    else:
        pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=False,
                                        suffix='_fold_{}'.format(fold_id))
    logger.info('Start pipeline transform on valid')
    pipeline.clean_cache()
    output_valid = pipeline.transform(valid_data)
    pipeline.clean_cache()

    y_valid_pred = output_valid['prediction']
    y_valid_true = valid_data_split[cfg.TARGET_COLUMNS].values
    score = score_function(y_valid_true, y_valid_pred)

    return score, y_valid_pred, pipeline


def _aggregate_test_prediction(out_of_fold_test_predictions):
    agg_methods = {'mean': np.mean,
                   'gmean': gmean}
    prediction_column = [col for col in out_of_fold_test_predictions.columns if '_prediction' in col][0]
    if params.aggregation_method == 'rank_mean':
        rank_column = prediction_column.replace('_prediction', '_rank')
        test_predictions_with_ranks = []
        for fold_id, fold_df in out_of_fold_test_predictions.groupby('fold_id'):
            fold_df[rank_column] = calculate_rank(fold_df[prediction_column])
            test_predictions_with_ranks.append(fold_df)
        test_predictions_with_ranks = pd.concat(test_predictions_with_ranks, axis=0)

        test_prediction_aggregated = test_predictions_with_ranks.groupby(cfg.ID_COLUMNS)[rank_column].apply(
            np.mean).reset_index()
    else:
        test_prediction_aggregated = out_of_fold_test_predictions.groupby(cfg.ID_COLUMNS)[prediction_column].apply(
            agg_methods[params.aggregation_method]).reset_index()

    test_prediction_aggregated.columns = [cfg.ID_COLUMNS + cfg.TARGET_COLUMNS]

    return test_prediction_aggregated
