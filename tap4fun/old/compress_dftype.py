import itertools as its
import numpy as np

def compress_dtypes(df_train):
    # int
    gl_int = df_train.select_dtypes(include=['int64'])

    int_types = ["uint16", "uint32","uint64"]
    int_types_max = {}
    for it in int_types:
        int_types_max[it] = np.iinfo(it).max
    int_types_max = sorted(int_types_max.items(),key= lambda x: x[1])

    column_types = {}
    for field,max in gl_int.max().iteritems():
        best_type= list(its.filterfalse(lambda x: max>x[1],int_types_max))
        column_types[field] = best_type[0][0]
    # float
    gl_float = df_train.select_dtypes(include=['float64'])
    float_types = ["float16", "float32","float64"]
    float_types_max = {}
    for it in float_types:
        float_types_max[its] = np.finfo(it).max
    float_types_max = sorted(float_types_max.items(),key= lambda x: x[1])

    
    for field,max in gl_float.max().iteritems():
        best_type= list(its.filterfalse(lambda x: max>x[1],float_types_max))
        column_types[field] = best_type[0][0]
    return df_train