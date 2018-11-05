#!/usr/bin/env python

"""
output file with libsvm format from log txt
"""

import sys
import csv
from .config import CATEGORICAL_COLS, TARGET_LABEL
from functools import cmp_to_key
import pandas as pd
import numpy as np
from itertools import chain


def construct_line(label, line, feat2idx):
    new_line = []

    new_line.append(str(label))

    for i, item in enumerate(line):
        col = line.index[i]
        if col == 'user_tags':
            if item==None or str(item)=='nan':
                continue
            for tag in item.split(','):
                if tag:
                    if not feat2idx.__contains__("%s:%s" % (col, tag)):
                        tag = 'others'
                    new_item = "%s:%s" % (col, tag)
                    new_line.append('%s:1' % feat2idx[new_item])
        elif col in CATEGORICAL_COLS:
            if item == None or str(item)=='nan':
                continue
            elif not feat2idx.__contains__("%s:%s" % (col, item)):
                item = 'others'
            new_item = "%s:%s" % (col, item)
            new_line.append('%s:1' % feat2idx[new_item])
    new_line = " ".join(new_line)
    new_line += "\n"
    return new_line


def generate_libsvm(dataset, dev_mode):
    """
        convert log file into libsvm format

        Arguments:
        -------------------------------------
        dataset: train or test
        dev_mode: true or false. if write tiny file using top k rows
    """
    if dataset == "train":
        input_file = './data/round1_iflyad_train.log.txt'
        if dev_mode:
            output_file = './data/round1_iflyad_train.tiny.txt'
        else:
            output_file = './data/round1_iflyad_train.txt'
    elif dataset == "test":
        input_file = './data/round1_iflyad_test.log.txt'
        if dev_mode:
            output_file = './data/round1_iflyad_test.tiny.txt'
        else:
            output_file = './data/round1_iflyad_test.txt'
    else:
        input_file = './data/round1_iflyad_train.log.txt'

    index_file = './data/featindex.txt'

    categorical_val = set()
    if dev_mode:
        train = pd.read_table(input_file, nrows=200000)
    else:
        train = pd.read_table(input_file)

    if dataset == 'feat':
        #---------------------------------
        # construct feature index
        #---------------------------------
        headers = train.columns

        total = train.shape[0]
        for i, col in enumerate(headers):
            if col in CATEGORICAL_COLS:
                if col == 'user_tags':
                    tags = train.loc[~train.user_tags.isnull(), 'user_tags'].values
                    utags = set(chain.from_iterable(
                        map(lambda item: item.split(','), tags)))
                    for item in utags:
                        if item:
                            ckey = '%s:%s' % (col, item)
                            categorical_val.add(ckey)
                    categorical_val.add('user_tags:others')
                else:
                    cnts = train.groupby(col)[col].count()
                    for item, count in cnts.to_dict().items():
                        if item == None:
                            item = 'NaN'
                        if count/total < 0.001:
                            item = 'others'
                        ckey = '%s:%s' % (col, item)
                        categorical_val.add(ckey)
        categorical_val = sorted(categorical_val)
        feat2idx = dict(
            list(map(lambda v: (v[1], str(v[0])), enumerate(categorical_val))))
        #---------------------------------
        # construct feature index file
        #---------------------------------
        outfile = open(index_file, 'w', encoding='utf-8')
        for feat, index in feat2idx.items():
            new_line = '%s\t%s\n' % (feat, index)
            outfile.write(new_line)
    else:
        #---------------------------------
        # load featindex
        #---------------------------------
        fin = open(index_file, 'r', encoding='utf-8')
        feat2idx = {}
        for line in fin.readlines():
            items = line.strip().split('\t')
            feat2idx[items[0]] = items[1]
        #---------------------------------
        # write libsvm file
        #---------------------------------
        outfile = open(output_file, 'w', encoding='utf-8')

        for index, line in train.iterrows():
            if dataset == 'train':
                label = line[TARGET_LABEL]
            else:
                label = '0'

            new_line = construct_line(label, line, feat2idx)

            outfile.write(new_line)

