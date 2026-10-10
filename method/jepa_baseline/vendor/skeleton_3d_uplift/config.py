"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

from enum import Enum
from yacs.config import CfgNode as CN


class Config:
	def __init__(self):
		self.cfg = CN({**TrainingConfig().cfg, **AugmentationConfig().cfg, **GeneralConfig().cfg, **ModelConfig().cfg, **DatasetConfig().cfg, **LossConfig().cfg})
		self.cfg.defrost()

class GeneralConfig:
	def __init__(self):
		C = CN()
		C.GET_CFG_FROM_CKPT = None
		C.LOAD_WEIGHTS = None
		C.LOAD_CKPT = True
		C.OUTPUT_DIR = "output/mlp/"
		C.BATCH_SIZE_VALID = None
		C.BATCH_SIZE = 256
		C.NUM_WORKERS = 8
		C.PRECISION = 32
		C.NAME = "body"
		C.SHOW = False
		C.STAGE = "fit"
		C.ALIGNMENT_SCALE = False
		C.defrost()
		self.cfg = C

class AugmentationConfig:
	def __init__(self):
		C = CN()
		C.IMG_AUGM_PROB = 1.0
		C.MOTION_BLUR_LIMIT = None
		C.BLUR_PROB = None
		C.GEOM = CN()
		C.GEOM.RAND = 'U'
		C.GEOM.ROT_PROB = 0.0
		C.GEOM.SHIFT_PROB = 0.0
		C.GEOM.ROT_RNG = 0

		# contrastive augmentation
		C.C_PIXEL_AUGMENTATION = False
		C.C_MOTION_BLUR_LIMIT = None
		C.C_BLUR_PROB = None
		C.C_GEOM = CN()
		C.C_GEOM.RAND = 'U'
		C.C_GEOM.ROT_PROB = 0.0
		C.C_GEOM.SHIFT_PROB = 0.0
		C.C_GEOM.ROT_RNG = 0.0

		C.A2D_NOISE_SCALE = 1e-4
		C.A2D_NOISE_PROB = 0.5
		C.A2D_ROT_PROB = 0.0

		C.A3D_NOISE_SCALE = 1e-4
		C.A3D_NOISE_PROB = 0.5
		C.A3D_ROT_PROB = 0.5
		C.A3D_ROT_DEV = 9

		C.MIN_SHIFT = 0.0
		C.MAX_SHIFT = 0.0
		C.DEFAULT_SHIFT = 0.0
		C.defrost()
		self.cfg = C

class TrainingConfig:
	def __init__(self):
		C = CN()
		C.CHECKPOINT = True
		C.WANDB_LOG = False
		C.WANDB_PROJECT = ''
		C.WANDB_NAME = ''
		C.TRAIN_ITER_SAVE = 10000
		C.PRINT_FREQ = 10
		C.OPTIMIZER = CN()
		C.OPTIMIZER.name = 'AdamW'
		C.OPTIMIZER.lr = 1e-4
		C.OPTIMIZER.weight_decay = 0.05
		C.FILTER_WEIGHT_DECAY = False
		C.MAX_EPOCHS = 50
		C.LOG = True
		C.SCHEDULER = CN()
		C.SCHEDULER.NAME = 'plateau' # cyclic, exp
		C.SCHEDULER.INTERVAL = 'epoch' # step
		C.SCHEDULER.FACTOR = 0.8
		C.SCHEDULER.PATIENCE = 1
		C.SCHEDULER.MAX_LR = 5e-4
		C.SCHEDULER.MIN_LR = 1e-5
		C.SCHEDULER.STEP_UP = 10
		C.SCHEDULER.GAMMA = 0.99
		C.SCHEDULER.THRESHOLD = 1e-3
		C.SCHEDULER.PCT_START = 0.1
		C.SCHEDULER.ADAPTIVE_GAMMA = True
		C.TRAIN_ALL = True
		C.SHUFFLE_VALID = False
		C.USE_EST_FOR_TRAINING = True
		C.SHOW_PROG_BAR = True
		C.defrost()
		self.cfg = C

class ModelConfig:
	class MODEL_INPUT(Enum):
		image = 0; points_2d = 1; points_3d = 2; angles = 3; ik = 4
	class IMAGE_MODEL(Enum):
		resnet = 0; efficientnet = 1

	def __init__(self):
		C = CN()
		C.OPTIMIZE_ANGLES = True
		C.NUM_RESNET_LAYERS = 50
		C.WITH_ARMS = False

		#### Temporal
		C.TEMP_WINDOW_SIZE = 0
		C.IS_N2N = True

		###
		C.IS_INTERSECTIONS_OFFSETS = False
		C.INTERSECTIONS_OFFSETS_WITH_ENVELOPE = False
		C.INTERSECTIONS_OFFSETS_PREDICT_HANDS = False
		C.INTERSECTIONS_OFFSETS_USE_GT_HANDS = False
		###
		#### autoencoder parameters
		C.IS_TEMPORAL_AUTOENCODER = False
		C.AUTOENCODER_ENCODER_PARAMS = []
		C.AUTOENCODER_ENCODER_PARAMS2 = []
		C.AUTOENCODER_DECODER_PARAMS = []

		#### VAE refinement
		C.IS_VAE_TEMPORAL = False
		C.VAE_DIM = None
		C.VAE_ENCODER_PARAMS = []
		C.VAE_DECODER_PARAMS = []

		C.VAE_DIM2 = None
		C.VAE_ENCODER_PARAMS2 = []
		C.VAE_DECODER_PARAMS2 = []

		#### transformer
		C.IS_TEMPORAL_TRANSFORMER = False
		C.NORM_FIRST_TRANSFORMER = True # works better
		C.NUM_TRANSFORMER_HEADS = 0 # 8
		C.ADD_TO_ENCODER_OUT = True
		C.NUM_TRANSFORMER_LAYERS = 6
		C.NUM_DECODER_TRANSFORMER_LAYERS = 2
		C.NUM_TRANSFORMER_HEADS_OFFSETS = 2
		C.NUM_TRANSFORMER_LAYERS_OFFSETS = 2
		C.EMBED_DIM = 512
		C.DROPOUT_P_TRANSFORMER = 0.05
		C.EMBED_DIM_OFFSETS = 32
		C.USE_ENCODER = False
		C.LEARNABLE_PE = False
		C.ENCODER_PROCESS = None
		C.LAYERS_POST_ENCODER = []
		C.USE_SRC_MASK = False

		# swin
		C.SWIN_DROP_RATE = 0.1
		C.SWIN_ATTN_DROP_RATE = 0.1
		C.SWIN_DROP_PATH_RATE = 0.2
		####

		C.WITH_ARMS_3D = False
		C.MLP_LAYERS = []
		C.MLP_LAYERS_OFFSETS = []
		C.MLP_LAYERS_RIGID_ROT = []
		C.WITH_OFFSETS = False
		C.CONSTRAINTS_FILE = ''
		C.LAYER_NORM = None # or batch seems to be better than layer, but no normalization is even better
		C.DROPOUT_P = 0.05
		C.WITH_VIRTUAL = False
		C.DROPOUT_P_OFFSETS = 0.10
		C.MLP_LAYERS_OFFSETS = [0]
		C.MODEL_INPUT = None
		C.IMAGE_MODEL = None
		C.IS_CANONICAL = False
		C.CANONICAL_BONES = None
		C.RIGID_ROT_TYPE = 'svd'
		C.RESNET_NORMALIZE_FEATURES = True
		C.CONTRASTIVE_MLP = None
		C.NEW_ANGULAR = False
		C.BB = []
		C.SCALE_INVARIANCE = False
		C.WO_ROOT_ANGLES = False
		# diffusion
		C.DIFFUSION = False
		C.CONTEXT_PARAMS = None
		C.DIFFUSION_TIMESTEPS = 50
		C.DIFFUSION_UNET1D_DIM = 16
		C.DIFFUSION_UNET_SIZE = None
		C.DIFFUSION_EMBEDDER = None # integer
		C.DIFFUSION_CHANNELS = 1
		C.DIFFUSION_NORM = True
		C.DIFFUSION_USE_WEIGHTS_IN_NORM = True
		C.DIFFUSION_ADD_IK = False
		# occupancy
		C.IS_OCCUPANCY = False
		C.IS_OCCUPANCY_TRANSFORM = False
		C.IS_OCCUPANCY_MLP_INT = False
		C.OCC_MLP_COMPARE_GT = False
		C.OCCUPANCY_TRANSFORM_PARAMS = [256]
		C.OCCUPANCY_DECODER = 'cbatchnorm'
		C.OCCUPANCY_ENCODER = 'pointnet_simple'
		C.OCCUPANCY_ENCODER_LATENT = None
		C.OCCUPANCY_NEW_INT = False
		C.OCCUPANCY_DOUBLE_HANDS = False
		C.OCCUPANCY_ADD_SHIFT_2D = False
		C.OCCUPANCY_WITH_2D_FLIP = 0
		C.OCCUPANCY_DATA_DIM = 3
		C.OCCUPANCY_Z_DIM = 0
		C.OCCUPANCY_C_DIM = 256
		C.OCCUPANCY_INPUT_DIM = 3
		C.OCCUPANCY_THRESHOLD = 0.5
		C.OCCUPANCY_POINTS_SUBSAMPLE = 8192
		C.OCCUPANCY_TRANSLATION_TOLERANCE = 0.03
		C.OCCUPANCY_ORIENTATION_TOLERANCE = 7.0 # degrees
		C.OCCUPANCY_TRANSLATION_HIDDEN_PARAMS = 128
		C.OCCUPANCY_ORIENTATION_HIDDEN_PARAMS = 128
		C.OCCUPANCY_TRANSLATION_WEIGHT = 1.0
		C.OCCUPANCY_ORIENTATION_WEIGHT = 1.0
		C.OCCUPANCY_SELF_INTERSECTION_WEIGHT = 1.0
		C.OCCUPANCY_SAMPLE_WITH_REPLACEMENT = True
		C.OCCUPANCY_USE_CANONICAL_HANDS = False
		C.OCCUPANCY_FOR_LOOP_INTERSECTION_CHECK = False
		C.OCCUPNACY_INFERENCE = CN()
		C.OCCUPNACY_INFERENCE.threshold = 0.2
		C.OCCUPNACY_INFERENCE.resolution_0 = 32
		C.OCCUPNACY_INFERENCE.upsampling_steps = 2
		C.OCCUPNACY_INFERENCE.use_sampling = False
		C.OCCUPNACY_INFERENCE.refinement_step = 30
		C.OCCUPNACY_INFERENCE.simplify_nfaces = 5000
		C.OCCUPANCY_USE_SYMM_ROTATION = False
		# 2d normalization
		C.NORMALIZATION_2D_TYPE = 'L1Med'
		#
		C.IMAGE_MODEL_ADD_RELU_TO_END = True # todo: only temporarly, should be False
		C.defrost()
		self.cfg = C

class DatasetConfig:
	def __init__(self):
		C = CN()
		C.META_EXT = ''
		C.DATA_STORAGE_PATH = '/media/scratch/mi00475/'
		C.DATASET = ''
		C.MAX_TRAIN_META = C.MAX_VALID_META = 1000000
		C.USE_TRAIN_METAS_FOR_VALID = False
		C.GET_ANGLES = True
		C.defrost()
		self.cfg = C

class LossConfig:
	def __init__(self):
		C = CN()
		C.LOSS = CN()
		C.LOSS.AGGREGATION = 'wsum' # 'wsum' (weighted sum), 'sum', or 'laplace'
		C.LOSS.LOSSES = ''
		C.LOSS.LOSS_FNC = ''
		C.LOSS.LOSS_WEIGHTS = []
		C.LOSS.ROOT_JOINT_WEIGHT = 10.0
		C.LOSS.USE_DEFAULT_JOINT_WEIGHT = False
		C.LOSS.JOINT_WEIGHT = []
		C.LOSS.ANGLES_WEIGHT = []
		C.LOSS.LEARNABLE_WEIGHTS = False
		C.LOSS.WEIGHTS_MIN = None # if weights are learnable
		C.defrost()
		self.cfg = C
