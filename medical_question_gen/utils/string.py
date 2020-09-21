# coding=utf-8
import glob
import json
import logging
import math
import os
import os.path as osp
import pdb
import pickle
import random
import re
import subprocess
import sys

import numpy as np


logger = logging.getLogger(__name__)


BERT_SPECIAL_TOKENS = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]

MY_SPECIAL_TOKENS = {
    'Indulgence&Decadence': '[unused1]',
    'Mindful Eating': '[unused2]',
    'Cool Off': '[unused3]',
    'Bonding&Sharing': '[unused4]',
    'Energise': '[unused5]',
}

REVESED_SPECIAL_TOKENS = dict([(item[1], item[0])
                               for item in MY_SPECIAL_TOKENS.items()])


def get_special_tokens():
    my_tokens = list(set(MY_SPECIAL_TOKENS.values()))
    return BERT_SPECIAL_TOKENS+my_tokens


def half2full(b_str):
    """半角转全角
    """
    q_str = ""
    for uchar in b_str:
        if uchar == ',':  # 半角字符（除空格）根据关系转化
            uchar = '，'
        elif uchar == '.':
            uchar = '。'
        elif uchar in '“”':
            uchar = '"'
        q_str += uchar
    return q_str


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def transferStr(ustring):
    """ 输入字符串预处理
    """
    ustring = half2full(ustring)
    ustring = convert_to_unicode(ustring)
    # replace
    for k, v in MY_SPECIAL_TOKENS.items():
        # 忽略大小写
        ustring = re.sub(k, v, ustring, flags=re.IGNORECASE)
    return ustring


def convert_to_original(token_ids, tokenizer):
    """translate token to original text"""
    tokens = [REVESED_SPECIAL_TOKENS.get(
        token, token) for token in tokenizer.convert_ids_to_tokens(token_ids)]
    sample_src = "".join(tokens).split('[CLS]')[1]
    sample_src = "、".join(sample_src.rsplit('[SEP]')[:-1]) + '\n'
    return sample_src.replace('#', '')
