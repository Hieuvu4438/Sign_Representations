"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import numpy as np
import torch
from utils import euler_angles_to_matrix

class ForwardK:
	@staticmethod
	def angles2rotsBackward(angles, skeleton, R_nodes):
		"""
		angles: B x A
		R_nodes: list, the leaf nodes are full : B x N x 3 x 3, other will be filled
		"""
		angular_tensor = skeleton.getAnglesTensor(angles)

		def appendR(node_idx, R):
			if R_nodes[node_idx] is not None:
				if isinstance(R_nodes[node_idx], list):
					R_nodes[node_idx] += [R]
				else:
					R_nodes[node_idx] = [R_nodes[node_idx], R]
			else:
				R_nodes[node_idx] = R

		def recursiveBFS(nodes):
			if len(nodes) == 0:
				return
			new_nodes = []
			for node in nodes:
				for child in node.children:
					if len(child.children) == 0: # child is leaf
						if child.dependent_ori:
							if len(child.ori) != 0:
								appendR(node.idx, torch.matmul(R_nodes[child.idx], euler_angles_to_matrix(angular_tensor[:,child.idx,child.mask_ori], child.ori).transpose(-1,-2)))
							else:
								appendR(node.idx, R_nodes[child.idx])

				new_nodes += node.children

			recursiveBFS(new_nodes)
			for node in nodes:
				for child in node.children:
					if len(child.children) != 0:
						if child.dependent_ori:
							if len(child.ori) != 0:
								# print('after fill', node.idx)
								appendR(node.idx, torch.matmul(R_nodes[child.idx], euler_angles_to_matrix(angular_tensor[:,child.idx,child.mask_ori], child.ori).transpose(-1,-2)))
							else:
								# print('after fill no ori', node.idx)
								appendR(node.idx, R_nodes[child.idx])
						elif len(child.ori) != 0:
							# print('after fill not dependent', node.idx)
							appendR(node.idx, euler_angles_to_matrix(angular_tensor[:,child.idx,child.mask_ori], child.ori).transpose(-1,-2))

		recursiveBFS(skeleton.roots)

	@staticmethod
	def forward(angles, pose, offsets_, nodes, R_list=None):
		"""
		offsets should be either B x N x 3 x 1 or N x 3 x 1
		pose is B x N x 3
		angles is B x N x 3
		"""
		assert angles.ndim == 3
		if R_list is None:
			R_list = np.array([None for i in range(pose.shape[1])])
			# todo: convert rotations in advance, fix multiplication over children (not in indices)
			for node in nodes:
				if np.sum(node.mask_ori) == 0:
					R_list[node.idx] = torch.tensor(np.identity(3), dtype=angles.dtype, device=angles.device).view(-1,3,3).repeat(angles.shape[0],1,1)
				else:
					R_list[node.idx] = euler_angles_to_matrix(angles[:, node.idx, node.mask_ori], node.ori)
		offsets = offsets_[None,:] if offsets_.ndim == 3 else offsets_
		assert offsets.ndim == 4 # offsets are B x N x 3 x 1
		while len(nodes) > 0:
			new_nodes = []
			for node in nodes:
				children_idxs = []
				for child in node.children:
					if child.dependent_ori:
						if len(child.ori) != 0:
							R_list[child.idx] = torch.matmul(R_list[node.idx], euler_angles_to_matrix(angles[:,child.idx,child.mask_ori], child.ori))
						else:
							R_list[child.idx] = R_list[node.idx]
					elif len(child.ori) != 0:
						R_list[child.idx] = euler_angles_to_matrix(angles[:,child.idx,child.mask_ori], child.ori)
					children_idxs.append(child.idx)
				if children_idxs != []:
					pose[:, children_idxs] = torch.matmul(torch.stack(list(R_list[children_idxs]),1), offsets[:,children_idxs])[...,0] + pose[:, None, node.idx]
				new_nodes += node.children
			nodes = new_nodes
		return R_list

	@staticmethod
	def forwardPoseSimple(skeleton, angles, offsets, origin=None):
		"""
		angles: B x A
		offsets: O or B x O
		origin: None or B x K x 3
		"""
		assert angles.ndim == 2
		pose = torch.zeros((len(angles), skeleton.num_nodes, 3), dtype=angles.dtype, device=angles.device)
		if origin is not None: pose[..., skeleton.getRootIdxs(),:] = origin
		ForwardK.forward(skeleton.getAnglesTensor(angles), pose, skeleton.getOffsetsFull(offsets), skeleton.roots)
		return pose

	@staticmethod
	def forwardAll(skeleton, angles, offsets_, origin=None, scale=None, angles_R=None):
		pose = torch.zeros((len(angles), skeleton.num_nodes, 3), dtype=angles.dtype, device=angles.device)
		if origin is not None: pose[:, skeleton.getRootIdxs()] = origin
		offsets = offsets_ if scale is None else offsets_ * scale
		ForwardK.forward(skeleton.getAnglesTensor(angles), pose, skeleton.getOffsetsFull(offsets), skeleton.roots)
		return skeleton.subSamplePose(pose)

	@staticmethod
	def getRot(skeleton, angles_, local_to_global):
		nodes = skeleton.roots
		if local_to_global:
			angles = skeleton.getAnglesTensor(angles_)
		else:
			angles = angles_
			if len(angles.shape) == 3:
				assert len(angles.shape) == 3 and angles.shape[-1] == 3
				R = euler_angles_to_matrix(angles, 'ZXY')
			elif len(angles.shape) == 4:
				assert angles.shape[-1] == 3 and angles.shape[-2] == 3
				R = angles

		R_list = np.array([None for i in range(angles.shape[1])])
		for node in nodes:
			if np.sum(node.mask_ori) == 0:
				R_list[node.idx] = torch.tensor(np.identity(3), dtype=angles.dtype, device=angles.device).view(-1,3,3).repeat(angles.shape[0],1,1)
			else:
				if local_to_global:
					R_list[node.idx] = euler_angles_to_matrix(angles[:, node.idx, node.mask_ori], node.ori)
				else:
					R_list[node.idx] = R[:,node.idx]
		while len(nodes) > 0:
			new_nodes = []
			for node in nodes:
				for child in node.children:
					if child.dependent_ori:
						if len(child.ori) != 0:
							if local_to_global:
								R_list[child.idx] = R_list[node.idx] @ euler_angles_to_matrix(angles[:,child.idx,child.mask_ori], child.ori)
							else:
								R_list[child.idx] = R[:,node.idx].transpose(1,2) @ R[:,child.idx]
						else:
							R_list[child.idx] = R_list[node.idx]
					elif len(child.ori) != 0:
						if local_to_global:
							R_list[child.idx] = euler_angles_to_matrix(angles[:,child.idx,child.mask_ori], child.ori)
						else:
							R_list[child.idx] = R[:,child.idx]
				new_nodes += node.children
			nodes = new_nodes
		R = torch.stack(list(R_list),1)
		return R

	@staticmethod
	def forwardPose(skeleton, angles, scale=None, T=None, R=None, offsets=None, origin=None):
		# angles : NUM_BATCHES x NUM_ANGLES
		# offsets : 1d array, NUM_OFFSETS
		NUM_BATCHES = angles.shape[0]
		angles_idx, offsets_idx, scale_idx, origin_idx = 0, 0, 0, 0
		poses_all = []
		for skeleton_ in skeleton.skeletons:
			pose = torch.zeros((NUM_BATCHES, skeleton_.num_nodes, 3), dtype=angles.dtype, device=angles.device)
			if origin is not None:
				pose[:, skeleton_.getRootIdxs()] = origin[:, origin_idx:origin_idx+len(skeleton_.getRootIdxs())]
				origin_idx += len(skeleton_.getRootIdxs())
			if offsets is None:
				offsets_ = skeleton_.getOffsetsValues()
			else:
				offsets_ = offsets[..., offsets_idx:offsets_idx+skeleton_.getOffsetsValues().numel()]
				offsets_idx += skeleton_.getOffsetsValues().numel()
			ForwardK.forward(skeleton_.getAnglesTensor(angles[:, angles_idx:angles_idx+skeleton_.getNumberOfAngles()]), pose,
				skeleton_.getOffsetsFull(offsets_), skeleton_.roots)
			pose = skeleton_.subSamplePose(pose)
			angles_idx += skeleton_.getNumberOfAngles()
			poses_all.append(pose)
		pose = skeleton.concatePoses(torch.cat(poses_all, 1))
		if R is not None:
			# assert R.shape == (NUM_BATCHES, 3, 3)
			pose = torch.matmul(R, pose.transpose(1,2)).transpose(1,2)
		if T is not None:
			# assert T.shape == torch.Size([NUM_BATCHES, 1, 3])
			pose = pose + T
		if scale is not None:
			pose *= scale.view(-1,1,1)
		# pose is NUM_BATCHES x NUM_POINTS x 3
		return pose
