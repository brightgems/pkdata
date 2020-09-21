import glob
import json
import logging
import math
import os
import os.path as osp

import lmdb
import msgpack
import numpy as np
import torch
import torch.functional as F
import torch.nn.init as init
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (BatchSampler, DataLoader, Dataset, RandomSampler,
                              Sampler, SequentialSampler)
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


def data2lmdb(lmdb_path, dataset, write_frequency=5000):
    def dumps_msgpack(obj):
        """
        Serialize an object.
        Returns:
            Implementation-dependent bytes-like object
        """
        return msgpack.dumps(obj)

    isdir = os.path.isdir(lmdb_path)
    print("Generate LMDB to %s" % lmdb_path)
    # It's OK to use super large map_size on Linux, but not on other platforms
    map_size = 1099511627776 * 2 if sys.platform == 'linux' else 128 * 10**6
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=map_size, readonly=False,
                   meminit=False, map_async=True)
    
    txn = db.begin(write=True)
    for idx, data in enumerate(dataset):
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_msgpack(data))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(dataset)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_msgpack(keys))
        txn.put(b'__len__', dumps_msgpack(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()
