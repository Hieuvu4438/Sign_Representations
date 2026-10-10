"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import torch
from torch import nn
from forward_kinematics import ForwardK
from maths_utils import symmetric_orthogonalization

"""
Linear : N x M
Input can be ANYTHING x N
Output: ANYTHING x M
"""

class MLP(nn.Module):
	def __init__(self, INPUT_SHAPE, NUM_OUTPUT, parameters, dropout_p, layer_norm, add_sigmoid=True, input_dropout=None):
		if dropout_p is not None: assert dropout_p >= 0.0 and dropout_p <= 1.0
		super().__init__()
		if isinstance(INPUT_SHAPE, list) or isinstance(INPUT_SHAPE, tuple):
			assert len(INPUT_SHAPE) == 2
			INPUT_SHAPE = INPUT_SHAPE[0]*INPUT_SHAPE[1]
		self.layers = nn.Sequential()
		self.layers.add_module('flatten', nn.Flatten())
		# input_dropout = 0.5
		if input_dropout is not None:
			self.layers.add_module('input_dropout', nn.Dropout(p=input_dropout))
		for i in range(len(parameters) + 1):
			if i == len(parameters):
				self.layers.add_module('linear_' + str(i), nn.Linear(parameters[i - 1], NUM_OUTPUT))
			else:
				if i == 0:
					shape = (INPUT_SHAPE, parameters[0])
				else:
					shape = (parameters[i - 1], parameters[i])
				# https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
				self.layers.add_module('linear_' + str(i), nn.Linear(shape[0], shape[1]))
				if layer_norm is not None:
					if layer_norm == "batch":
						self.layers.add_module(layer_norm+"_norm"+str(i), nn.BatchNorm1d(num_features=shape[1]))
					elif layer_norm == "layer":
						self.layers.add_module(layer_norm+"_norm"+str(i), nn.LayerNorm(normalized_shape=shape[1]))
					else: raise NotImplementedError
				# dropout should be placed before activation function, but in case of relu, it can be before so it will be more efficient
				# https://sebastianraschka.com/faq/docs/dropout-activation.html
				if dropout_p is not None and dropout_p > 0: self.layers.add_module('dropout_'+str(i), nn.Dropout(p=dropout_p))
				# if dropout_p is not None and i == 0: self.layers.add_module('dropout_'+str(i), nn.Dropout(p=dropout_p))
				self.layers.add_module('relu_' + str(i), nn.ReLU())
		if add_sigmoid: self.layers.add_module('sigmoid', nn.Sigmoid())  # sigmoid output range [0,1]

class FeatureKeypointAngularModel(torch.nn.Module):
	def __init__(self, skeleton, model_angular, model_offsets, camera_rot_model=None, camera_rot_params=None, features_norm=None, double_input=False):
		super(FeatureKeypointAngularModel, self).__init__()
		self.skeleton = skeleton
		self.model_angular = model_angular
		self.model_offsets = model_offsets
		self.model_cam_rot = camera_rot_model(**camera_rot_params) if camera_rot_model is not None else None
		self.features_norm = features_norm
		self.double_input = double_input

	def forward(self, features, **kwargs):
		if self.features_norm is not None: features = self.features_norm(features)

		angles = self.model_angular(features)
		kwargs_offs = kwargs['offsets'] if 'offsets' in kwargs else {}
		offsets = self.model_offsets(features, **kwargs_offs)
		if self.double_input:
			offsets = offsets[:,None].repeat(1,2,1).reshape(-1, offsets.shape[-1])

		out = {'offsets': offsets}

		if self.model_cam_rot is None:
			out['pose_3d'] = self.skeleton.subSamplePose(ForwardK.forwardPoseSimple(self.skeleton, angles, offsets, kwargs['origin'] if 'origin' in kwargs else None))
		else:
			if self.skeleton.root.idx == 0:
				angles = torch.cat((torch.zeros((len(angles), 3), device=features.device), angles),-1)
			else:
				angles = torch.cat((angles[...,:self.skeleton.root_angles_idxs[0]], torch.zeros((len(angles), 3), device=features.device), angles[...,self.skeleton.root_angles_idxs[0]:]),-1)

			pose_3d = self.skeleton.subSamplePose(ForwardK.forwardPoseSimple(self.skeleton, angles, offsets))
			out['pose_3d_can'] = pose_3d
			out['cam_rot'] = self.model_cam_rot(features)
			if self.double_input: out['cam_rot'] = out['cam_rot'].reshape(-1,3,3)
			out['pose_3d'] = (out['cam_rot'] @ pose_3d.transpose(-1,-2)).transpose(-1,-2)
			if 'origin' in kwargs and kwargs['origin'] is not None:
				out['pose_3d'] = out['pose_3d'] + kwargs['origin']

		out['angles'] = angles
		return out

class ImageFeatureModel(torch.nn.Module):
	def __init__(self, num_features, normalize_features, mlp_layers, add_relu_to_end):
		super(ImageFeatureModel, self).__init__()
		self.normalize_features = normalize_features
		if mlp_layers is not None:
			self.num_features = mlp_layers[-1]
			self.mlp_layers = torch.nn.Sequential()
			for i, layer in enumerate(mlp_layers):
				if i == 0:
					self.mlp_layers.add_module('linear_' + str(i), nn.Linear(num_features, mlp_layers[i]))
				else:
					self.mlp_layers.add_module('linear_' + str(i), nn.Linear(mlp_layers[i-1], mlp_layers[i]))
				if i != len(mlp_layers)-1 or add_relu_to_end:
					if i == len(mlp_layers)-1:
						print('\033[93mWARNING!!! In new release remove last ReLU! \033[0m')
					self.mlp_layers.add_module("relu_"+str(i), nn.ReLU())
		else:
			self.mlp_layers = None
			self.num_features = num_features

	def getNumberOfFeature(self):
		return self.num_features

	def forward(self, img):
		features = self.feature_model(img)
		if self.mlp_layers is not None:
			features = self.mlp_layers(features)
		if self.normalize_features: # after mlp
			features = features / (1e-9 + torch.linalg.norm(features, dim=-1))[...,None]
		return features

class ResnetFeatureModel(ImageFeatureModel):
	def __init__(self, num_layers, normalize_features=True, mlp_layers=None, add_relu_to_end=True):
		feature_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet'+str(num_layers), pretrained=True)
		super(ResnetFeatureModel, self).__init__(feature_model.fc.in_features, normalize_features, mlp_layers, add_relu_to_end)
		self.feature_model = feature_model
		self.feature_model.fc = torch.nn.Identity()

class ImageModel(torch.nn.Module):
	def __init__(self, image_model, keypoint_model):
		super(ImageModel, self).__init__()
		self.model_feature = image_model
		self.keypoint_model = keypoint_model

	def forward(self, image, **kwargs):
		features = self.model_feature(image)
		keypoints = self.keypoint_model(features, **kwargs)
		return {**{'features': features}, **keypoints}


class RigidRotModel(MLP):
	def __init__(self, INPUT_SHAPE, parameters, dropout_p, layer_norm, rot_type, num_rot=1):
		self.num_rot = num_rot
		if rot_type == 'svd':
			OUTPUT_NUM = 9*num_rot
		else:
			raise NotImplementedError
		self.rot_type = rot_type
		super(RigidRotModel, self).__init__(INPUT_SHAPE, OUTPUT_NUM, parameters, dropout_p, layer_norm, False)

	def forward(self, x):
		if self.rot_type == 'svd':
			out = self.layers(x)
			if self.num_rot > 1:
				out = out.reshape(-1, 9)
			rot = symmetric_orthogonalization(out)
			if self.num_rot > 1:
				rot = rot.reshape(len(x), self.num_rot, 3, 3)
			return rot
		else:
			raise NotImplementedError

class MLPNorm(MLP):
	def __init__(self, INPUT_SHAPE, NUM_OUTPUT, parameters, dropout_p, layer_norm, normalization, split_output=False):
		super(MLPNorm, self).__init__(INPUT_SHAPE, NUM_OUTPUT, parameters, dropout_p, layer_norm, False)
		self.normalization = normalization
		self.split_output = split_output
		self.NUM_OUTPUT_HALF = NUM_OUTPUT // 2

	def forward(self, x):
		mlp_out = self.layers(x)
		if self.split_output: mlp_out = mlp_out.reshape(-1, 2, self.NUM_OUTPUT_HALF)
		mlp_norm = self.normalization(mlp_out)
		if self.split_output: mlp_norm = mlp_norm.reshape(2*len(mlp_norm), self.NUM_OUTPUT_HALF)
		return mlp_norm
class MLPOffsets(MLP):
	def __init__(self, INPUT_SHAPE, NUM_OUTPUT, parameters, dropout_p, norm_fnc):
		super(MLPOffsets, self).__init__(INPUT_SHAPE, NUM_OUTPUT, parameters, dropout_p, None, False)
		self.norm_fnc = norm_fnc
	def forward(self, x):
		return self.norm_fnc(self.layers(x))

class OffsetsMLP(MLP):
	def __init__(self, parameters, num_input_pts, skeleton_base, skeleton_goal):
		super(OffsetsMLP, self).__init__((num_input_pts, 2), skeleton_goal.offsets_vals.numel(), parameters, dropout_p=None, layer_norm=None, add_sigmoid=False)
		self.skeleton_base = skeleton_base
		self.skeleton_goal = skeleton_goal
		self.limits_min = skeleton_goal.offsets_constraints[None,:,0]
		self.limits_dif = skeleton_goal.offsets_constraints[None,:,1] - self.limits_min
		self.neg_values_mask = skeleton_goal.offsets_vals < 0

	def enforceConstraint(self, y):
		y[torch.logical_xor(y < 0, self.neg_values_mask.to(y.device))] *= -1.0
		# assert (self.neg_values_mask == (y < 0)).all()
		return y
