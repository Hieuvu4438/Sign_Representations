"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import torch
import os

from config import ModelConfig
from mlp_model import MLPNorm, RigidRotModel, FeatureKeypointAngularModel, ImageModel, ResnetFeatureModel
from normalization import SineNormFnc, Normalization2DL2Med, Normalization2DL1Med, Normalization2DL2


def init_weights(m):
	# https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
	if isinstance(m, torch.nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight);
		if m.bias is not None: m.bias.data.fill_(0.01)

def getModelFromCKPT(fname, eval=False, get_cfg=False, to_cuda=False):
	assert os.path.exists(fname), fname
	ckpt = torch.load(fname, map_location='cpu')
	name_sk1 = name_sk2 = 'self.skeleton'
	if 'self.skeleton_base' in ckpt:
		name_sk1, name_sk2 = 'self.skeleton_base', 'self.skeleton_goal'
	# else: assert name_sk1 in ckpt, ckpt.keys()
	if name_sk1 in ckpt.keys() and name_sk2 in ckpt.keys():
		model = getModelFromConfigSkeleton(ckpt['self.cfg'], ckpt[name_sk1], ckpt[name_sk2])
	else:
		model = getModelFromConfigSkeleton(ckpt['self.cfg'], None, None)
	model_ = torch.nn.Module()
	model_.model = model
	model_.load_state_dict(ckpt['state_dict'])
	if eval: model_.model.eval()
	if to_cuda:
		model_.model = model_.model.cuda()
	if get_cfg:
		return model_.model, ckpt['self.cfg']
	else:
		return model_.model

def getModelFromConfigSkeleton(cfg, skeleton_base, skeleton_target=None):
	if not 'SHUFFLE_VALID' in cfg: cfg['SHUFFLE_VALID'] = False
	if not 'TEMP_WINDOW_SIZE' in cfg: cfg['TEMP_WINDOW_SIZE'] = 0
	if not 'GET_ANGLES' in cfg: cfg['GET_ANGLES'] = False
	if not 'TRAIN_ALL' in cfg: cfg['TRAIN_ALL'] = True
	if not 'IS_VAE_TEMPORAL' in cfg: cfg['IS_VAE_TEMPORAL'] = False
	if not 'IS_TEMPORAL_AUTOENCODER' in cfg: cfg['IS_TEMPORAL_AUTOENCODER'] = False
	if not 'IS_TEMPORAL_TRANSFORMER' in cfg: cfg['IS_TEMPORAL_TRANSFORMER'] = False
	if not 'OCCUPANCY_DOUBLE_HANDS' in cfg: cfg['OCCUPANCY_DOUBLE_HANDS'] = False
	if not 'OCCUPANCY_NEW_INT' in cfg: cfg['OCCUPANCY_NEW_INT'] = False
	if not 'IS_INTERSECTIONS_OFFSETS' in cfg: cfg['IS_INTERSECTIONS_OFFSETS'] = False

	skeleton_target = skeleton_base if skeleton_target is None else skeleton_target

	features_norm = None
	if cfg.MODEL_INPUT == ModelConfig.MODEL_INPUT.image:
		if cfg.IMAGE_MODEL == ModelConfig.IMAGE_MODEL.resnet:
			if 'IMAGE_MODEL_ADD_RELU_TO_END' not in cfg: cfg['IMAGE_MODEL_ADD_RELU_TO_END'] = True
			image_model = ResnetFeatureModel(cfg.NUM_RESNET_LAYERS, cfg.RESNET_NORMALIZE_FEATURES, cfg.CONTRASTIVE_MLP, add_relu_to_end=cfg.IMAGE_MODEL_ADD_RELU_TO_END)
		else: raise NotImplementedError('image model ' + str(cfg.IMAGE_MODEL))
		num_features = image_model.getNumberOfFeature()
	elif cfg.MODEL_INPUT == ModelConfig.MODEL_INPUT.points_2d:
		num_features = 2*skeleton_base.num_non_virtual_nodes

		if cfg.NORMALIZATION_2D_TYPE == 'L2':
			features_norm = Normalization2DL2(None)
		elif cfg.NORMALIZATION_2D_TYPE == 'L1Med':
			features_norm = Normalization2DL1Med(None)
		elif cfg.NORMALIZATION_2D_TYPE == 'L2Med':
			features_norm = Normalization2DL2Med(None)
		else: raise NotImplementedError

	else: raise NotImplementedError('model input '+str(cfg.MODEL_INPUT))

	angular_norm = SineNormFnc(min_max_values=skeleton_target.angles_constraints[~skeleton_target.root_angles_mask] if (len(cfg.MLP_LAYERS_RIGID_ROT) > 0 or cfg.IS_CANONICAL) else skeleton_target.angles_constraints)
	params_rigid_rot = {}
	rigid_rot_instance = None

	angular_model = MLPNorm(INPUT_SHAPE=(num_features,1), NUM_OUTPUT=(2 if cfg.OCCUPANCY_DOUBLE_HANDS else 1)*(skeleton_target.getNumberOfAngles()-int(len(cfg.MLP_LAYERS_RIGID_ROT)>0 or cfg.IS_CANONICAL)*skeleton_target.num_root_angles), parameters= cfg.MLP_LAYERS, dropout_p= cfg.DROPOUT_P, layer_norm= cfg.LAYER_NORM, normalization=angular_norm, split_output=cfg.OCCUPANCY_DOUBLE_HANDS)

	if len(cfg.MLP_LAYERS_RIGID_ROT) > 0:
		assert len(skeleton_target.roots) == 1
		params_rigid_rot = {'INPUT_SHAPE': num_features, 'parameters': cfg.MLP_LAYERS_RIGID_ROT, 'dropout_p': cfg.DROPOUT_P, 'layer_norm': cfg.LAYER_NORM, 'rot_type': cfg.RIGID_ROT_TYPE, 'num_rot': 2 if cfg.OCCUPANCY_DOUBLE_HANDS else 1}
		rigid_rot_instance = RigidRotModel

	offsets_norm = SineNormFnc(min_max_values=skeleton_target.offsets_constraints)
	offsets_model = MLPNorm(INPUT_SHAPE=(num_features,1), NUM_OUTPUT=skeleton_target.num_offsets, parameters= cfg.MLP_LAYERS_OFFSETS, dropout_p= cfg.DROPOUT_P, layer_norm=cfg.LAYER_NORM, normalization=offsets_norm)

	keypoint_model = FeatureKeypointAngularModel(skeleton_base, angular_model, offsets_model, rigid_rot_instance, params_rigid_rot, features_norm=features_norm, double_input=cfg.OCCUPANCY_DOUBLE_HANDS)

	if cfg.MODEL_INPUT == ModelConfig.MODEL_INPUT.image:
		model = ImageModel(image_model, keypoint_model)
	elif cfg.MODEL_INPUT == ModelConfig.MODEL_INPUT.points_2d:
		model = keypoint_model
	else: raise NotImplementedError

	model.apply(init_weights)
	return model

