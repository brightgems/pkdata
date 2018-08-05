import numpy as np
from steppy.base import BaseTransformer
from steppy.utils import get_logger
from .utils import compress_dtypes

logger = get_logger()


class Tap4funCleaning(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()

    def transform(self, X):
        X = compress_dtypes(X)

        return {'X': X}


