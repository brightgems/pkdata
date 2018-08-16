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
from sklearn.metrics import mean_squared_error
import requests
from environs import Env

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


def compress_dtypes(df_train):
    # int
    gl_int = df_train.select_dtypes(include=['int64'])
    int_types = ["uint16", "uint32", "uint64"]
    int_types_max = {}
    for it in int_types:
        int_types_max[it] = np.iinfo(it).max
    int_types_max = sorted(int_types_max.items(), key=lambda x: x[1])

    column_types = {}
    for field, max in gl_int.max().iteritems():
        best_type = list(its.filterfalse(lambda x: max > x[1], int_types_max))

        column_types[field] = best_type[0][0]
    # float
    gl_float = df_train.select_dtypes(include=['float16', 'float64'])
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
    for field in df_train.select_dtypes('category').columns:
        column_types[field] = 'object'
    # apply compressed type
    for c, t in column_types.items():
        df_train[c] = df_train[c].astype(t)
    return df_train


def get_random_string():
    min_char = 8
    max_char = 12
    allchar = string.ascii_letters + string.digits
    password = "".join(random.choice(allchar) for x in range(random.randint(min_char, max_char)))
    return password