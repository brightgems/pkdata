import logging
import os
import sys
import string
import random
import numpy as np
import pandas as pd
import yaml
from attrdict import AttrDict
import itertools as its
import numpy as np
from sklearn.metrics import (
    mean_squared_error, roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
)
import statsmodels.api as sm
import requests
from environs import Env
from featuretools import selection
from matplotlib import pyplot as plt


env = Env()
# Read .env into os.environ
env.read_env()

def make_submission(pipeline_name, submission_filepath):
    logger = logging.getLogger('tap4fun')
    logger.info('making dcjingsai submit...')
    user =env.str('pkbigdata_user', None)
    password = env.str('pkbigdata_password', None)
    if not (user and password):
        raise Exception("Must have pkbigdata_user and pkbigdata_password in enviroment variables")

    with requests.Session() as sess:
        rsp = sess.post("http://www.dcjingsai.com/user/common/login.json",
            data={'username':user,'password':password})
        files = {'file': open(submission_filepath)}
        rsp= sess.post("http://www.dcjingsai.com/user/file/uploadSubmissionFile.json", files =files)
        if rsp.status_code != 200:
            raise Exception("upload submission failed")
        else:
            if rsp.json()['msg'] != '':
                raise Exception("upload submission failed: "+rsp.json()['msg'])
            else:
                # commit result
                fullPath = rsp.json()['data']['fullPath']
                rsp = sess.post('http://www.dcjingsai.com/user/cmpt/commitResult.json',
                    data = {
                            'description': pipeline_name,
                            'stageId': 121,
                            'filePath': fullPath,
                            'name': os.path.split(submission_filepath)[-1],
                            'cmpId': 226}
                )
                if rsp.status_code == 200:
                    result = rsp.json()['data']
                    logger.info("submission done:\r\n"+str(result))
                else:
                    logger.error("failed to commit result")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def create_submission(meta, predictions):
    # calculate prediction pay price in 45 days by add pay_price in first 7 days
    predictions[predictions < 0] = 0
    predictions = predictions + meta['pay_price']
    submission = pd.DataFrame({'user_id': meta['user_id'].tolist(),
                               'prediction_pay_price': predictions
                               })
    return submission


def verify_submission(submission, sample_submission):
    assert str(submission.columns) == str(sample_submission.columns), \
        'Expected submission to have columns {} but got {}'.format(
            sample_submission.columns, submission.columns)

    # for submission_id, correct_id in zip(submission['user_id'].values, sample_submission['user_id'].values):
    #     assert correct_id == submission_id, \
    #         'Wrong id: expected {} but got {}'.format(correct_id, submission_id)


def get_logger():
    return logging.getLogger('tap4fun')


def init_logger():
    logger = logging.getLogger('tap4fun')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger


def read_params(ctx, fallback_file):
    if ctx.params.__class__.__name__ == 'OfflineContextParams':
        neptune_config = read_yaml(fallback_file)
        params = neptune_config.parameters
    else:
        params = ctx.params
    return params


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


def parameter_eval(param):
    try:
        return eval(param)
    except Exception:
        return param


def persist_evaluation_predictions(experiment_directory, y_pred, raw_data, id_column, target_column):
    raw_data.loc[:, 'y_pred'] = y_pred.reshape(-1)
    predictions_df = raw_data.loc[:, [id_column, target_column, 'y_pred']]
    filepath = os.path.join(experiment_directory, 'evaluation_predictions.csv')
    logging.info('evaluation predictions csv shape: {}'.format(
        predictions_df.shape))
    predictions_df.to_csv(filepath, index=None)


def set_seed(seed=90210):
    random.seed(seed)
    np.random.seed(seed)


def calculate_rank(predictions):
    rank = (1 + predictions.rank().values) / (predictions.shape[0] + 1)
    return rank


def compress_dtypes(df):
    """ 
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    # int
    gl_int = df.select_dtypes(include=['int64','uint16','uint32','uint64'])
    int_types = ["int16", "int32", "int64"]
    int_types_max = {}
    for it in int_types:
        int_types_max[it] = np.iinfo(it).max
    int_types_max = sorted(int_types_max.items(), key=lambda x: x[1])

    column_types = {}
    for field, max in gl_int.max().iteritems():
        best_type = list(its.filterfalse(lambda x: max > x[1], int_types_max))

        column_types[field] = best_type[0][0]
    # float
    gl_float = df.select_dtypes(include=['float16', 'float64'])
    float_types = ["float32", "float64"]
    float_types_max = {}
    for it in float_types:
        float_types_max[it] = np.finfo(it).max
    float_types_max = sorted(float_types_max.items(), key=lambda x: x[1])
    for field, max in gl_float.max().iteritems():
        best_type = list(its.filterfalse(
            lambda x: max > x[1], float_types_max))
        column_types[field] = best_type[0][0]
    # category column don't support merge when aggregate step
    for field in df.select_dtypes('category').columns:
        column_types[field] = 'object'
    # apply compressed type
    for c, t in column_types.items():
        df[c] = df[c].astype(t)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def remove_li_features(df):
    """Remove low information features"""
    old_shape = df.shape[1]
    df = selection.remove_low_information_features(df)
    print('Removed features from df: {}'.format(old_shape - df.shape[1]))

    return df


def replace_day_outliers(df):
    """Replace 365243 with np.nan in any columns with DAYS"""
    
    for col in df.columns:
        if "DAYS" in col:
            df[col] = df[col].replace({365243: np.nan})

    return df


def display_roc_curve(y_, oof_preds_, folds_idx_):
    """Plot ROC-AUC curve and calculates AUC"""

    # Plot ROC curves
    plt.figure(figsize=(6,6))
    scores = [] 
    for n_fold, (_, val_idx) in enumerate(folds_idx_):  
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' \
                 % (n_fold + 1, score))
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(fpr, tpr, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()


def get_random_string():
    min_char = 8
    max_char = 12
    allchar = string.ascii_letters + string.digits
    password = "".join(random.choice(allchar) for x in range(random.randint(min_char, max_char)))
    return password

def split_train_evaluate(df, train_ratio=0.8):
    """
        split dataset by register_date
    """
    max_days =  df.day_num.max()
    cutoff_daynum = int(train_ratio*max_days)
    df_train = df[df.day_num<cutoff_daynum]
    df_evaluate = df[df.day_num>=cutoff_daynum]
    return df_train, df_evaluate

# PREPROCESSING BLOCK -------------------------------------------------------------------------------
def reduce_mem_usage(df, skip_cols_pattern='register_time'):
    """ 
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:

        if skip_cols_pattern in col:
            print(f"don't optimize index {col}")

        else:
            col_type = df[col].dtype

            if col_type != object:

                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df    



def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included    