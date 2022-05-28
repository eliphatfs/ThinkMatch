from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# IGM model options
__C.IGM = edict()
__C.IGM.FEATURE_CHANNEL = 512
__C.IGM.SK_ITER_NUM = 10
__C.IGM.SK_EPSILON = 1e-10
__C.IGM.SK_TAU = 0.005
__C.IGM.MGM_SK_TAU = 0.005
__C.IGM.GNN_FEAT = [16, 16, 16]
__C.IGM.GNN_LAYER = 3
__C.IGM.GAUSSIAN_SIGMA = 1.
__C.IGM.SIGMA3 = 1.
__C.IGM.WEIGHT2 = 1.
__C.IGM.WEIGHT3 = 1.
__C.IGM.EDGE_FEATURE = 'cat' # 'cat' or 'geo'
__C.IGM.ORDER3_FEATURE = 'none' # 'cat' or 'geo' or 'none'
__C.IGM.FIRST_ORDER = True
__C.IGM.EDGE_EMB = False
__C.IGM.SK_EMB = 1
__C.IGM.GUMBEL_SK = 0 # 0 for no gumbel, other wise for number of gumbel samples
__C.IGM.UNIV_SIZE = -1
__C.IGM.POSITIVE_EDGES = True
