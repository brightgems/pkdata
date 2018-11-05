# coding: utf-8
NUM_SPLITS = 3
RANDOM_SEED = 233

#-------------------------------------------------
# 读取数据，统计基本的信息，field等
#-------------------------------------------------
FIELD_SIZES = {}
with open('./data/featindex.txt') as fin:
    for line in fin:
        line = line.strip().split(':')
        if len(line) > 1:
            f = line[0]
            if FIELD_SIZES.__contains__(f):
                FIELD_SIZES[f] += 1
            else:
                FIELD_SIZES[f] = 1
print('field sizes:', FIELD_SIZES)
FIELD_OFFSETS = [sum(list(FIELD_SIZES.values())[:i]) for i in range(len(FIELD_SIZES))]
INPUT_DIM = sum(FIELD_SIZES.values())
field_sizes = list(FIELD_SIZES.values())
field_offsets = FIELD_OFFSETS
OUTPUT_DIM = 1
STDDEV = 1e-3
MINVAL = -1e-3
MAXVAL = 1e-3
col2idx = dict(list(map(lambda v: (v[1],v[0]+1), enumerate(FIELD_SIZES.keys()))))    

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
'advert_id', #广告主id
'order_id',#订单id
'advert_industry_inner', #广告主行业
'campaign_id', #活动id
'creative_id',#创意id
'creative_type',#创意类型
'creative_tp_dnf',#样式定向id
'creative_has_deeplink',#category 
'creative_is_jump',
'creative_is_download',



#-------------------------------------
# 媒体信息
#-------------------------------------
'app_cate_id',
'f_channel',
'app_id',
'inner_slot_id',
#-------------------------------------
# 用户信息
#-------------------------------------
'user_tags',
#-------------------------------------
# 上下文信息
#-------------------------------------
'city',
'carrier', # 运营商
'province',
'nnt', # 联网数据
'devtype',
'os_name',
'osv',
'os',
'make',
'model',
'ad_id',
]

NUMERIC_COLS = [
'creative_width',
'creative_height',
]

IGNORE_COLS = [
'instance_id',
'app_paid',
'creative_is_js',
'creative_is_voicead',
]

TARGET_LABEL = 'click'

# hyper parameters of pipelines
from .pnn1 import PNN1


PIPELINES = {
    'pnn1':
        {
            'model_class': PNN1,
            'model_params': {
                'field_sizes': field_sizes,
                'embed_size': 10,
                'layer_sizes': [500, 1],
                'layer_acts': ['relu', None],
                'drop_out': [0.2, 0],
                'opt_algo': 'adagrad',
                'learning_rate': 0.1,
                'embed_l2': 0,
                'layer_l2': [0.0001, 0],
                'random_seed': 33
            }
        },
}