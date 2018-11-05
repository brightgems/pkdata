import os
import random
import shutil

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_squared_error, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from .csv2libsvm import generate_libsvm
from .config import *
from .tfutils import timeit

class PipelineManager():
    @timeit
    def train_evaluate(self, pipeline_name, dev_mode):
        model_params = PIPELINES[pipeline_name]['model_params']
        model_class = PIPELINES[pipeline_name]['model_class']
        if dev_mode:
            train_file = './data/round1_iflyad_train.tiny.txt'
        else:
            train_file = './data/train.pkl'
        model= model_class(**model_params)
        model.train_evaluate(train_file)
        return model

    @timeit
    def predict(self, pipeline_name, dev_mode):
        model = self.train_evaluate(pipeline_name, dev_mode)
        model_params = PIPELINES[pipeline_name]['model_params']
        model_class = PIPELINES[pipeline_name]['model_class']
        test_file = './data/round1_iflyad_test.txt'
        print(model_params)
        model.predict(test_file)
        return model

    @timeit
    def prepare_dataset(self, dataset, dev_mode):
        generate_libsvm(dataset, dev_mode)
