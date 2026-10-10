"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import torch
import numpy as np
import copy
from enum import Enum

class SKELETON_PART(Enum):
	BODY = 0; HAND = 1; HAND_LEFT = 2; HAND_RIGHT = 3; FULL = 4; UNKNOWN = -1; HAND_RIGHT_BACKWARDS = 5

class Joint:
	def __init__(self, idx, children, mask_ori, dependent_ori=False, Rx=[-180, 180], Ry=[-180, 180], Rz=[-180, 180], name=''):
		# ori is orientation is any combination of X Y Z of length 0 to 3
		# children is list
		self.idx = idx
		self.children = children
		self.dependent_ori = dependent_ori
		self.ori = ''
		self.mask_ori = mask_ori
		self.setOri(mask_ori)
		self.ori_constraints = [Rz, Rx, Ry] # Z X Y convension must follow
		self.name = name

	def setOri(self, mask_ori):
		assert len(mask_ori) == 3
		self.mask_ori = mask_ori
		self.ori = ''
		if mask_ori[0]: self.ori += 'Z'
		if mask_ori[1]: self.ori += 'X'
		if mask_ori[2]: self.ori += 'Y'

def getNumNodes(root):
	nodes = [root]
	num_nodes = 0
	while len(nodes) > 0:
		num_nodes += len(nodes)
		new_nodes = []
		for node in nodes: new_nodes += node.children
		nodes = new_nodes
	return num_nodes

def getOrderedNodes(nodes, num_nodes):
	nodes_ordered = [None for i in range(num_nodes)]
	parents_idxs = torch.zeros(num_nodes, dtype=torch.int)-1

	leaves = []
	while len(nodes) > 0:
		new_nodes = []
		for node in nodes:
			nodes_ordered[node.idx] = node
			if len(node.children) == 0:
				leaves.append(node.idx)
			else:
				new_nodes += node.children
				for child in node.children:
					parents_idxs[child.idx] = node.idx
		nodes = new_nodes
	return nodes_ordered, parents_idxs, torch.tensor(leaves, dtype=int)

class Skeleton:
	def __init__(self, device, dtype=torch.float32):
		self.convention = 'ZXY'
		self.device = device
		self.dtype = dtype
		self.num_virtual_nodes = 0
		self.sk_part = SKELETON_PART.UNKNOWN

	def setOffsets(self, roots, num_nodes, offsets_dict):
		self.angles_constraints, self.offsets_sign, self.offsets_idxs, self.values_idxs, self.offsets_vals = [], [], [], [], []
		self.angles_tensor_idxs, self.angles_values_range = [], []
		self.node_angle_indexing = {}
		self.angle_idx2joint = []

		self.nodes_ordered, self.parents_idxs, self.leaves = getOrderedNodes(roots, num_nodes)
		self.root_idxs = [root.idx for root in self.roots]
		self.root_angles_idxs = []
		self.node2names = {}
		self.name2nodes = {}
		self.num_root_angles = np.sum([np.sum(node.mask_ori) for node in roots])
		for node in self.nodes_ordered:
			self.node2names[node.idx] = node.name
			self.name2nodes[node.name] = node.idx
			for k_ori, is_ori in enumerate(node.mask_ori):
				if is_ori:
					# print(len(self.angles_tensor_idxs), ') idx', node.idx * 3 + k_ori, 'node', node.idx, 'constraint', node.ori_constraints[k_ori])
					if node.idx not in self.node_angle_indexing: self.node_angle_indexing[node.idx] = {}
					self.node_angle_indexing[node.idx][k_ori] = len(self.angles_tensor_idxs)
					if node.idx in self.root_idxs:
						self.root_angles_idxs.append(self.node_angle_indexing[node.idx][k_ori])
					self.angles_tensor_idxs.append(node.idx * 3 + k_ori)
					self.angle_idx2joint.append((node.idx, k_ori))
					self.angles_constraints.append(node.ori_constraints[k_ori])
		self.root_angles_mask = torch.zeros(len(self.angles_constraints), dtype=bool)
		self.root_angles_mask[self.root_angles_idxs] = True
		# print('------------------------------------------------')
		offsets_key_to_val_idx = {}
		for k, key in enumerate(offsets_dict.keys()):
			self.offsets_vals.append(offsets_dict[key][0])
			offsets_key_to_val_idx[key] = (offsets_dict[key][0], k)
			if   key[-1] == 'y': shift = 1
			elif key[-1] == 'z': shift = 2
			elif key[-1] == 'x': shift = 0
			else: assert False
			for (joint_idx, sign) in offsets_dict[key][1]:
				self.values_idxs.append(k)
				self.offsets_idxs.append(joint_idx * 3 + shift)
				self.offsets_sign.append(sign)

		self.proportions_constraints = []

		self.proportions_idxs = []
		self.proportions_sign = []
		self.values_idxs_for_proportions = []
		self.offsets_idxs_proportions = []

		self.proportions_idxs_double = []
		self.proportions_sign_double = []
		self.values_idxs_for_proportions_double = []
		self.offsets_idxs_proportions_double = []

		has_offsets_constraints_defined = hasattr(self, 'offsets_constraints')
		if hasattr(self, 'proportions'):
			proportions_keys_idxs = {}
			for i, proportion in enumerate(self.proportions):
				proportions_keys_idxs[proportion['name']] = len(offsets_key_to_val_idx) + i

			if not has_offsets_constraints_defined:
				self.offsets_constraints = []
				for v in self.offsets_vals:
					assert v > 0
					self.offsets_constraints.append([1e-2*v, 1e2*v])

			for proportion in self.proportions:
				ref = proportion['ref']
				key = proportion['name']
				if   key[-1] == 'y': shift = 1
				elif key[-1] == 'z': shift = 2
				elif key[-1] == 'x': shift = 0
				else: assert False

				self.offsets_vals.append(.5*(proportion['min']+proportion['max']))
				self.proportions_constraints.append([proportion['min'], proportion['max']])
				# print(proportion['name'], proportion['ref'], proportion['min'], proportion['max'])
				# self.offsets_vals.append(offsets_key_to_val_idx[ref][0])
				assert len(proportion['nodes']) == len(proportion['signs'])
				for node, sign in zip(proportion['nodes'], proportion['signs']):
					if ref in offsets_key_to_val_idx:
						self.proportions_idxs.append(len(self.offsets_vals)-1)
						self.values_idxs_for_proportions.append(offsets_key_to_val_idx[ref][1])
						self.offsets_idxs_proportions.append(node * 3 + shift)
						self.proportions_sign.append(sign)
					else:
						# had to be double reference
						assert ref in proportions_keys_idxs
						self.proportions_idxs_double.append(len(self.offsets_vals)-1)
						self.values_idxs_for_proportions_double.append(proportions_keys_idxs[ref])
						self.offsets_idxs_proportions_double.append(node * 3 + shift)
						self.proportions_sign_double.append(sign)

					# self.offsets_idxs.append(proportion['node'] * 3 + shift)
					# self.values_idxs.append(offsets_key_to_val_idx[ref][1])
					# self.offsets_sign.append((proportion['max'] + proportion['min'])*.5)
			self.proportions_sign = torch.tensor(self.proportions_sign, dtype=self.dtype)
			if len(self.proportions_sign_double) > 0:
				self.proportions_sign_double = torch.tensor(self.proportions_sign_double, dtype=self.dtype)
			self.offsets_constraints = torch.tensor(self.offsets_constraints+self.proportions_constraints, dtype=self.dtype)
		else:
			if not has_offsets_constraints_defined:
				self.offsets_constraints = self.computeOffsetsContraints(self.offsets_vals)

		self.angles_tensor_idxs = torch.tensor(self.angles_tensor_idxs, dtype=int)
		self.angles_constraints = torch.tensor(np.array(self.angles_constraints), dtype=self.dtype, device=self.device) * np.pi / 180
		self.offsets_vals = torch.tensor(self.offsets_vals, dtype=self.dtype, device=self.device)
		self.offsets_idxs = torch.tensor(self.offsets_idxs, dtype=int)
		self.offsets_sign = torch.tensor(self.offsets_sign, dtype=self.dtype, device=self.device)
		self.values_idxs = torch.tensor(self.values_idxs, dtype=int)
		self.offsets_init = torch.zeros((self.num_nodes, 3, 1), dtype=self.dtype, device=self.device)
		# self.num_offsets = len(offsets_dict.keys())
		self.num_offsets = len(self.offsets_vals)
		self.neg_offsets_mask = self.offsets_vals < 0

		# self.edges_pairs1, self.edges_pairs2_upper, self.edges_pairs2_lower, self.idxs_edges_order = self.computeEdges()

		self.offsets_edges = {}
		for key in offsets_dict:
			self.offsets_edges[key] = []
			for (node, _) in offsets_dict[key][1]:
				self.offsets_edges[key].append([self.parents_idxs[node], node])
			if key[-1] == 'x': axis = 0
			elif key[-1] == 'y': axis = 1
			elif key[-1] == 'z': axis = 2
			else: assert False
			self.offsets_edges[key] = (torch.tensor(self.offsets_edges[key],dtype=torch.long), axis)

	def to(self, device, deepcopy=True):
		copy_ = copy.deepcopy(self) if deepcopy else self
		copy_.device = device
		if copy_.getName() == 'body_mediapipe':
			return copy_
		if copy_.getName() == 'mediapipe_orig':
			copy_.skeleton_body = copy_.skeleton_body.to(device, deepcopy)
			copy_.skeleton_hand = copy_.skeleton_hand.to(device, deepcopy)
			return copy_
		if copy_.getName() == 'human3.6_full':
			copy_.skeleton_body = copy_.skeleton_body.to(device, deepcopy)
			return copy_
		copy_.angles_constraints = copy_.angles_constraints.to(device)
		copy_.offsets_vals = copy_.offsets_vals.to(device)
		copy_.offsets_sign = copy_.offsets_sign.to(device)
		copy_.offsets_init = copy_.offsets_init.to(device)
		copy_.offsets_constraints = copy_.offsets_constraints.to(device)
		if isinstance(copy_, HumanSkeleton):
			if hasattr(copy_, 'skeleton_hand'): copy_.skeleton_hand = copy_.skeleton_hand.to(device, deepcopy)
			if hasattr(copy_, 'skeleton_body'): copy_.skeleton_body = copy_.skeleton_body.to(device, deepcopy)
		return copy_

	def cpu(self, deepcopy=True):
		return self.to('cpu', deepcopy)
	def getEdges(self):
		raise NotImplementedError
	def computeOffsetsContraints(self, offsets_vals, min_scale=5e-2, max_scale=15):
		# assert offsets_vals.ndim == 1

		offsets_constraints = []
		for ii, (k, v) in enumerate(self.offsets_dict.items()):
			v = v[0]
			constraint = [min_scale*v, max_scale*v]
			offsets_constraints.append(constraint if v > 0 else constraint[::-1])
		return torch.tensor(offsets_constraints+self.proportions_constraints, dtype=self.dtype)
	def getOrigin(self, poses):
		raise NotImplementedError
	def getLeaves(self):
		return self.leaves
	def getAnglesTensor(self, angles_values, angles=None):
		if angles is None:
			if angles_values.ndim == 1:
				angles = torch.zeros((1, self.num_nodes, 3), dtype=self.dtype, device=angles_values.device)
			else:
				angles = torch.zeros((angles_values.shape[0], self.num_nodes, 3), dtype=self.dtype, device=angles_values.device)
		# assert angles_values.shape[1] == self.angles_tensor_idxs.numel()
		angles.view(angles.shape[0],-1)[:, self.angles_tensor_idxs] = angles_values
		return angles
	def getNumberOfAngles(self):
		return self.angles_tensor_idxs.numel()
	def getOffsetsValues(self, is_grad=False):
		return self.offsets_vals.clone().requires_grad_(is_grad)
	def getOffsetsFull(self, offsets_values, offsets_init=None):
		"""
		offsets_values: B x D or D
		"""
		# off2 = offsets_values.clone()
		assert offsets_values.shape[-1] == len(self.offsets_vals)
		if offsets_values.ndim == 1:
			offsets = self.offsets_init.clone(); offsets = offsets.to(offsets_values.device)
			if self.proportions_idxs:
				offsets.flatten()[self.offsets_idxs] = offsets_values[self.values_idxs]
				offsets.flatten()[self.offsets_idxs_proportions] = self.proportions_sign.to(offsets_values.device) * offsets_values[self.proportions_idxs] * offsets_values[self.values_idxs_for_proportions]
				if len(self.offsets_idxs_proportions_double) > 0:
					# torch.linalg.norm(off2 - offsets_values) # should be zero
					# todo: could be done more efficient, x = a * p1, y = a * p1 * p2
					off2 = offsets_values.clone()
					off2[self.proportions_idxs] *= offsets_values[self.values_idxs_for_proportions]
					offsets.flatten()[self.offsets_idxs_proportions_double] = self.proportions_sign_double.to(offsets_values.device) * off2[self.proportions_idxs_double] * off2[self.values_idxs_for_proportions_double]
			else:
				offsets.flatten()[self.offsets_idxs] = self.offsets_sign.to(offsets_values.device) * offsets_values[self.values_idxs]
		elif offsets_values.ndim == 2:
			offsets = (offsets_init.clone() if offsets_init is not None else self.offsets_init.clone()[None,:].repeat(offsets_values.shape[0],1,1,1)).to(offsets_values.device)
			if self.proportions_idxs:
				offsets.view(offsets.shape[0],-1)[:, self.offsets_idxs] = offsets_values[:,self.values_idxs]
				offsets.view(offsets.shape[0],-1)[:, self.offsets_idxs_proportions] = self.proportions_sign.view(1,-1).to(offsets_values.device) * offsets_values[:,self.proportions_idxs] * offsets_values[:,self.values_idxs_for_proportions]
				if len(self.offsets_idxs_proportions_double) > 0:
					off2 = offsets_values.clone()
					off2[:,self.proportions_idxs] *= offsets_values[:,self.values_idxs_for_proportions]
					offsets.view(offsets.shape[0],-1)[:, self.offsets_idxs_proportions_double] = self.proportions_sign_double.view(1,-1).to(offsets_values.device) * off2[:, self.proportions_idxs_double] * off2[:, self.values_idxs_for_proportions_double]
			else:
				offsets.view(offsets.shape[0],-1)[:, self.offsets_idxs] = self.offsets_sign.view(1,-1).to(offsets_values.device) * offsets_values[:,self.values_idxs]
		else: assert False
		return offsets
	def getOffsets(self, is_grad=False):
		return self.getOffsetsFull(self.getOffsetsValues(is_grad))
	def getRootIdxs(self):
		return self.root_idxs
	def subSamplePose(self, pose):
		return pose
	def extendProbs(self, probs):
		return probs
	def extendPose(self, poses):
		return poses
	def concatePoses(self, poses):
		return poses
	def centralizePose(self, poses):
		raise NotImplementedError
	def getJointWeight(self):
		raise NotImplementedError
	def getZeroPose(self, batch_size=1):
		return torch.zeros((batch_size, self.num_nodes, 3), dtype=self.dtype, device=self.device)
	def getRootAnglesIdxs(self):
		return self.root_angles_idxs
	def normalizeMLP(self, poses):
		raise NotImplementedError
	def getConnections(self):
		raise NotImplementedError
	def getConnectionsVirtual(self):
		return self.getConnections()

	def computeEdges(self):
		"""
		out:
		edges_pairs1 : num_offsets - 1
		edges_pairs2_upper, edges_pairs2_lower : (num_offsets-1) * (num_offsets-2) / 2
		"""
		edges_pairs1 = torch.tensor([(i, 0) for i in range(1, self.num_offsets)], dtype=int)
		edges_pairs2_upper = []
		edges_pairs2_lower = []
		idxs_order = []
		cnt1 = cnt2 = 0
		for i in range(1, self.num_offsets):
			for j in range(1, self.num_offsets):
				if j > i:
					edges_pairs2_upper.append([i, j])
					idxs_order.append(cnt1)
					cnt1 += 1
				elif j < i:
					edges_pairs2_lower.append([i, j])
					idxs_order.append(cnt2 + int((self.num_offsets -1) * (self.num_offsets - 2) / 2))
					cnt2 += 1

		idxs_order = []
		edges_pairs2_upper = torch.tensor(edges_pairs2_upper, dtype=int)
		total = torch.cat((edges_pairs2_upper, edges_pairs2_upper[:,[1,0]]))
		for i in range(1, self.num_offsets):
			for j in range(1, self.num_offsets):
				if i != j:
					idxs_order.append( torch.where((total == torch.tensor([i,j])).sum(-1) == 2)[0].item())
		idxs_order = torch.tensor(idxs_order, dtype=int)
		assert idxs_order.numel() == idxs_order.unique().numel()
		return edges_pairs1, edges_pairs2_upper, torch.tensor(edges_pairs2_lower, dtype=int), idxs_order

	def multiplyEdges(self, params, scale=1.0):
		"""
		params:
		for every except first multiply every except first and itself + symmetry
		of size B x (num_offsets - 1) * (num_offsets - 2) / 2 + (num_offsets - 1)
		"""

		offsets = torch.ones(len(params), self.num_offsets, device=params.device) * scale
		offsets[:, 1:] *= params[:, :self.num_offsets-1]
		offsets_upper = offsets[:, self.edges_pairs2_upper[:,1]] * params[:, self.num_offsets-1:]
		offsets_lower = offsets[:, self.edges_pairs2_upper[:,0]] / params[:, self.num_offsets-1:]
		# for k, (x,y) in enumerate(self.edges_pairs2_upper):
		# 	if self.offsets_vals[x] / self.offsets_vals[y] < 0:
		# 		assert (params[:,self.num_offsets-1+ k] < 0).all()
		# 	else:
		# 		assert (params[:,self.num_offsets-1+ k] > 0).all()
		# 	assert (offsets_upper[:,k] * offsets[:,x] > 0).all()
		# 	assert (offsets_lower[:,k] * offsets[:,y] > 0).all()

		offsets_symm = torch.cat((offsets_upper, offsets_lower), -1)[:, self.idxs_edges_order].view(-1, self.num_offsets-1, self.num_offsets-2)
		offsets[:, 1:] = (offsets_symm.sum(-1) + offsets[:, 1:]) / (self.num_offsets - 1)

		# assert torch.logical_xor(offsets < 0, self.neg_offsets_mask[None,:]).sum() == 0
		# assert torch.logical_xor(params[:, :self.num_offsets-1] < 0, self.neg_offsets_mask[None,1:]).sum() == 0
		return offsets

class HandSkeleton(Skeleton):
	###########################################
	############## constraints ################
	###########################################
	free_joint = [-180, 180]
	full_rot = [True, True, True]
	no_rot = [False, False, False]
	# accordingly to this bottom part does not have roll angle, in total should be 26 DoF for the hand.
	# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6347551
	bottom_rot = [True, True, False]
	# right / left hand: Z - bottom-up,  Y - around, X - right-left
	finger_rot = [True, False, False]
	deg90_top = [-90, 3]
	deg135_middle = [-135, 0]; deg90_bottom = [-90, 0]
	pinky_bottom_rx = [-15,40]; finger_bottom_ry = [-15,15]
	thumb_bottom_rz = [-25,25]; thumb_bottom_ry = [-60,15]; thumb_bottom_rx = [-40,5]
	ring_bottom_rx = [-17,17]; middle_bottom_rx = [-20,20]; index_bottom_rx = [-40,25]
	###########################################

	@staticmethod
	def getOffsets_():
		return {'thumb_start_y' : [1.13, [(2, 1)]], 'thumb_middle_y' : [0.93, [(3, 1)]], 'thumb_top_y' : [1.25, [(4, 1)]],
				'finger_start_y' : [1.07, [(6, 1), (10, 1), (14, 1)]], 'finger_middle_y' : [0.98, [(7, 1), (11, 1), (15, 1)]], 'finger_top_y' : [0.88, [(8, 1), (12, 1), (16, 1)]],
				'pinky_start_y' : [.93, [(18, 1)]], 'pinky_middle_y' : [.82, [(19, 1)]], 'pinky_top_y' : [.75, [(20, 1)]],
				'thumb_x' : [.19, [(1, 1)]], 'thumb_y' : [.8, [(1, 1)]], 'thumb_z' : [-.3, [(1, 1)]],
				'index_x' : [-.1, [(5, 1)]], 'index_y' : [2.7, [(5, 1)]], 'index_z' : [-0.01, [(5, 1)]],
				'middle_x' : [-.17, [(9, 1)]], 'middle_y' : [2.8, [(9, 1)]], 'middle_z' : [.48, [(9, 1)]],
				'ring_x' : [-.09, [(13, 1)]], 'ring_y' : [2.5, [(13, 1)]], 'ring_z' : [.84, [(13, 1)]],
				'pinky_x' : [-0.07, [(17, 1)]], 'pinky_y' : [2.3, [(17, 1)]], 'pinky_z' : [1.15, [(17, 1)]]}

	@staticmethod
	def getRightHandRoot(enforce_wrist_constraints):
		finger_rot = HandSkeleton.finger_rot
		deg135_middle = HandSkeleton.deg135_middle
		deg90_top = HandSkeleton.deg90_top
		pinky_top = Joint(20, [], finger_rot, True, Rz=deg90_top, Rx=deg90_top, Ry=deg90_top, name='pinky_top r')
		ring_top = Joint(16, [], finger_rot, True, Rz=deg90_top, Rx=deg90_top, Ry=deg90_top, name='ring_top r')
		middle_top = Joint(12, [], finger_rot, True, Rz=deg90_top, Rx=deg90_top, Ry=deg90_top, name='middle_top r')
		index_top = Joint(8, [], finger_rot, True, Rz=deg90_top, Rx=deg90_top, Ry=deg90_top, name='index_top r')
		thumb_top = Joint(4, [], finger_rot, True, Rz=deg90_top, Rx=deg90_top, Ry=deg90_top, name='thumb_top r')

		pinky_middle = Joint(19, [pinky_top], finger_rot, True, Rz=deg135_middle, Rx=deg135_middle, Ry=deg135_middle, name='pinky_middle r')
		ring_middle = Joint(15, [ring_top], finger_rot, True, Rz=deg135_middle, Rx=deg135_middle, Ry=deg135_middle, name='ring_middle r')
		middle_middle = Joint(11, [middle_top], finger_rot, True, Rz=deg135_middle, Rx=deg135_middle, Ry=deg135_middle, name='middle_middle r')
		index_middle = Joint(7, [index_top], finger_rot, True, Rz=deg135_middle, Rx=deg135_middle, Ry=deg135_middle, name='index_middle r')
		thumb_middle = Joint(3, [thumb_top], HandSkeleton.full_rot, True, Rx=[-5, 5], Ry=[-7, 7], Rz=[-70,0], name='thumb_middle r')

		pinky_bottom = Joint(18, [pinky_middle], HandSkeleton.bottom_rot, True, Rx=HandSkeleton.pinky_bottom_rx, Ry=HandSkeleton.finger_bottom_ry, Rz=HandSkeleton.deg90_bottom, name='pinky_bottom r')
		ring_bottom = Joint(14, [ring_middle], HandSkeleton.bottom_rot, True, Rx=HandSkeleton.ring_bottom_rx, Ry=HandSkeleton.finger_bottom_ry, Rz=HandSkeleton.deg90_bottom, name='ring_bottom r')
		middle_bottom = Joint(10, [middle_middle], HandSkeleton.bottom_rot, True, Rx=HandSkeleton.middle_bottom_rx, Ry=HandSkeleton.finger_bottom_ry, Rz=HandSkeleton.deg90_bottom, name='middle_bottom r')
		index_bottom = Joint(6, [index_middle], HandSkeleton.bottom_rot, True, Rx=HandSkeleton.index_bottom_rx, Ry=HandSkeleton.finger_bottom_ry, Rz=HandSkeleton.deg90_bottom, name='index_bottom r')
		thumb_bottom = Joint(2, [thumb_middle], HandSkeleton.full_rot, True, Rx=HandSkeleton.thumb_bottom_rx, Ry=HandSkeleton.thumb_bottom_ry, Rz=HandSkeleton.thumb_bottom_rz, name='thumb_bottom r')

		pinky = Joint(17, [pinky_bottom], HandSkeleton.no_rot, True, name='pinky r')
		ring = Joint(13, [ring_bottom], HandSkeleton.no_rot, True, name='ring r')
		middle = Joint(9, [middle_bottom], HandSkeleton.no_rot, True, name='middle r')
		index = Joint(5, [index_bottom], HandSkeleton.no_rot, True, name='index r')
		thumb = Joint(1, [thumb_bottom], HandSkeleton.no_rot, True, name='thumb r')

		if enforce_wrist_constraints:
			return Joint(0, [pinky, ring, index, middle, thumb], HandSkeleton.full_rot, True, Rz=[-80, 80], Rx=[-45, 45], Ry=[-180, 90], name='hand wrist r')
		else:
			return Joint(0, [pinky, ring, index, middle, thumb], HandSkeleton.full_rot, Rx=HandSkeleton.free_joint, Ry=HandSkeleton.free_joint, Rz=HandSkeleton.free_joint, name='hand wrist r')

	def __init__(self, device, dtype=torch.float32, offsets_dict=None, enforce_wrist_constraints=False):
		super().__init__(device, dtype=dtype)
		# assert False, 'it is deprecated'
		self.sk_part = SKELETON_PART.HAND
		if offsets_dict is None:
			self.offsets_dict = HandSkeleton.getOffsets_()
		else:
			self.offsets_dict = copy.deepcopy(offsets_dict)

		self.root = HandSkeleton.getRightHandRoot(enforce_wrist_constraints)
		self.root_left = copy.deepcopy(self.root)
		if enforce_wrist_constraints:
			self.root_left.ori_constraints[2] = -np.flip(self.root.ori_constraints[2])

		self.num_hand_pts = 21
		nodes = [self.root_left]
		while len(nodes) > 0:
			new_nodes = []
			for node in nodes:
				node.ori_constraints[0] = [-node.ori_constraints[0][1], -node.ori_constraints[0][0]] # change Z range
				node.idx += self.num_hand_pts
				new_nodes += node.children
				node.name = node.name[:-1] + 'l'
			nodes = new_nodes
		for k in self.offsets_dict.keys():
			# TODO: do not change angle (constraint) but add scale -1 for this angle (it simplifies constraint as well). Reason: With training network, we do not need separate angles for two different hand, but it's enough to have one universal set of angles for both hands 
			s = -1 if k[-1] == 'x' else 1
			self.offsets_dict[k][1] += [(v[0]+self.num_hand_pts, s*v[1]) for v in self.offsets_dict[k][1]]

		# print(self.offsets_dict, 'hand offsets after')
		self.num_nodes = 2*self.num_hand_pts
		self.roots = [self.root, self.root_left]
		self.setOffsets(self.roots, self.num_nodes, self.offsets_dict)
		self.num_hand_angles = len(self.angles_constraints)//2

		flip_hand_angles_idxs = []
		for i in range(self.num_hand_angles):
			if not (self.angles_constraints[i] == self.angles_constraints[i+self.num_hand_angles]).all():
				# print('angle', i, 'constraint', self.angles_constraints[i], self.angles_constraints[i+self.num_hand_angles], self.angle_idx2joint[i])
				flip_hand_angles_idxs.append(i)
				joint = self.angle_idx2joint[i][0]
				if 2 in self.node_angle_indexing[joint]: # add also roll angle
					flip_hand_angles_idxs.append(self.node_angle_indexing[joint][2])
		flip_hand_angles_idxs += [0,2,3,5]
		self.flip_hand_angles_idxs = torch.tensor(flip_hand_angles_idxs, dtype=torch.long)
		self.root_angles_idxs = list(self.node_angle_indexing[self.root.idx].values())

	@staticmethod
	def getConnectionsHand_():
		return [(0, 1), (1, 2), (2, 3), (3, 4),
			 (0, 5), (5, 6), (6, 7), (7, 8),
			 (5, 9), (9, 10), (10, 11), (11, 12),
			 (9, 13), (13, 14), (14, 15), (15, 16),
			 (13, 17), (17, 18), (18, 19), (19, 20), (17, 0)]

	def getConnectionsHand(self):
		return HandSkeleton.getConnectionsHand_()

	def getConnections(self):
		hand1 = self.getConnectionsHand()
		hands = hand1 + [(v[0]+self.num_hand_pts, v[1]+self.num_hand_pts) for v in hand1]
		return hands

	def getName(self):
		return 'hands'

	def centralizePose(self, poses):
		if poses.shape[-2] == self.num_hand_pts:
			return poses - poses[...,[0],:]
		elif poses.shape[-2] == self.num_nodes:
			return torch.cat((poses[...,:self.num_hand_pts,:] - poses[...,[0],:], poses[...,self.num_hand_pts:,:] - poses[...,[self.num_hand_pts],:]),-2)
		assert False
	def extendPose(self, poses):
		return poses
	def extendProbs(self, weights):
		return weights
	def getJointWeight(self):
		weights = torch.ones(self.num_nodes, dtype=self.dtype)
		weights[[5,17,self.num_hand_pts+5, self.num_hand_pts+17]] *= 10 # weight index and pinky that determines orientation of a wrist
		return weights
	def getAngularWeights(self):
		weights = torch.ones(self.num_hand_angles, dtype=self.dtype)
		weights[self.root_angles_idxs] *= 10
		return torch.cat((weights, weights))
	def colorsLeft(self):
		return np.array([[255, 0, 0]]).repeat(self.num_hand_pts, 0)
	def colorsLeftEdges(self):
		return np.array([[100, 0, 0]]).repeat(len(self.getConnections())//2, 0)
	def colorsRight(self):
		return np.array([[0, 255, 0]]).repeat(self.num_hand_pts, 0)
	def colorsRightEdges(self):
		return np.array([[200, 50, 50]]).repeat(len(self.getConnections())//2, 0)
	def normalizeMLP(self, poses):
		poses = self.centralizePose(poses)
		return poses / (1e-10 + poses.abs().max(-1)[0].max(-1)[0][..., None, None])

def setAnglesConstraints(root, angles_constraints):
	num_nodes = getNumNodes(root)
	nodes_ordered = getOrderedNodes([root], num_nodes)[0]
	cnt = 0
	for i in range(num_nodes):
		for k, is_ori in enumerate(nodes_ordered[i].mask_ori):
			if is_ori:
				nodes_ordered[i].ori_constraints[k] = angles_constraints[cnt]
				cnt += 1

class HandSkeletonNew(Skeleton):
	def changeProportions(self, proportions, key):
		for p in proportions:
			if p['name'] == key:
				p_key = p
				break
		for p in proportions:
			if p['name'] == key: continue
			print(p['name'], p['min'], p['max'], 1/p_key['min'], 1/p_key['max'], p['min'] * (1/p_key['min']), p['max'] * (1/p_key['max']))

	@staticmethod
	def getOffsetsAndProportions(REMOVE_SCALE):
		# we define only one bone length while other would be derived by proportions.
		# it could be a bit dangerous to predict only one value, since all other will depend on it
		# however, it will be solved either by 1) removing the scale, 2) constraining the first value prediction by a-priori data.

		# take largest limb length to minimize errors
		offsets_dict = {'index_root_y': [0.1, [(5, 1)]]}
		offsets_constraints = [[1.0, 1.0]] if REMOVE_SCALE else	[[0.03, 0.15]]
		proportions = []
		proportions.append({'ref': 'index_root_y', 'name': 'thumb_root_y', 'nodes': [1], 'min': 0.35, 'max': 0.55, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'thumb_start_y', 'nodes': [2], 'min': 0.34, 'max': 0.54, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'thumb_middle_y', 'nodes': [3], 'min': 0.3, 'max': 0.4, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'thumb_top_y', 'nodes': [4], 'min': 0.21, 'max': 0.31, 'signs': [1]})

		proportions.append({'ref': 'index_root_y', 'name': 'index_start_y', 'nodes': [6], 'min': 0.40, 'max': 0.56, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'index_middle_y', 'nodes': [7], 'min': 0.22, 'max': 0.32, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'index_top_y', 'nodes': [8], 'min': 0.16, 'max': 0.26, 'signs': [1]})

		proportions.append({'ref': 'index_root_y', 'name': 'middle_root_y', 'nodes': [9], 'min': 0.86, 'max': 1.15, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'middle_start_y', 'nodes': [10], 'min': 0.47, 'max': 0.65, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'middle_middle_y', 'nodes': [11], 'min': 0.24, 'max': 0.34, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'middle_top_y', 'nodes': [12], 'min': 0.18, 'max': 0.28, 'signs': [1]})

		proportions.append({'ref': 'index_root_y', 'name': 'ring_root_y', 'nodes': [13], 'min': 0.80, 'max': 0.99, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'ring_start_y', 'nodes': [14], 'min': 0.42, 'max': 0.58, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'ring_middle_y', 'nodes': [15], 'min': 0.21, 'max': 0.31, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'ring_top_y', 'nodes': [16], 'min': 0.15, 'max': 0.25, 'signs': [1]})

		proportions.append({'ref': 'index_root_y', 'name': 'pinky_root_y', 'nodes': [17], 'min': 0.76, 'max': 0.95, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'pinky_start_y', 'nodes': [18], 'min': 0.33, 'max': 0.6, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'pinky_middle_y', 'nodes': [19], 'min': 0.16, 'max': 0.26, 'signs': [1]})
		proportions.append({'ref': 'index_root_y', 'name': 'pinky_top_y', 'nodes': [20], 'min': 0.11, 'max': 0.21, 'signs': [1]})
		return offsets_dict, proportions, offsets_constraints

	@staticmethod
	def getAngles(enforce_wrist_constraints, angles_constraints=None):
		finger_rot = HandSkeleton.finger_rot
		deg90_top = HandSkeleton.deg90_top

		deg135_middle = [-135, 4]; deg90_bottom = [-90, 10]
		pinky_top = Joint(20, [], finger_rot, True, Rz=deg90_top, Rx=deg90_top, Ry=deg90_top, name='pinky_top_r')
		ring_top = Joint(16, [], finger_rot, True, Rz=deg90_top, Rx=deg90_top, Ry=deg90_top, name='ring_top_r')
		middle_top = Joint(12, [], finger_rot, True, Rz=deg90_top, Rx=deg90_top, Ry=deg90_top, name='middle_top_r')
		index_top = Joint(8, [], finger_rot, True, Rz=deg90_top, Rx=deg90_top, Ry=deg90_top, name='index_top_r')
		thumb_top = Joint(4, [], finger_rot, True, Rz=deg90_top, Rx=deg90_top, Ry=deg90_top, name='thumb_top_r')

		pinky_middle = Joint(19, [pinky_top], finger_rot, True, Rz=deg135_middle, Rx=deg135_middle, Ry=deg135_middle, name='pinky_middle_r')
		ring_middle = Joint(15, [ring_top], finger_rot, True, Rz=deg135_middle, Rx=deg135_middle, Ry=deg135_middle, name='ring_middle_r')
		middle_middle = Joint(11, [middle_top], finger_rot, True, Rz=deg135_middle, Rx=deg135_middle, Ry=deg135_middle, name='middle_middle_r')
		index_middle = Joint(7, [index_top], finger_rot, True, Rz=deg135_middle, Rx=deg135_middle, Ry=deg135_middle, name='index_middle_r')
		thumb_middle = Joint(3, [thumb_top], HandSkeleton.full_rot, True, Rx=[-5, 20], Ry=[-30, 15], Rz=[-40,-3], name='thumb_middle_r')

		thumb_bottom_rz = [-25, 25]; thumb_bottom_rx = [5,40]; thumb_bottom_ry = [-60, -20]
		ring_bottom_rx = [-20,15]
		pinky_bottom_rx = [-25,20]
		index_bottom_rx = [-15,32]
		pinky_bottom = Joint(18, [pinky_middle], HandSkeleton.bottom_rot, True, Rx=pinky_bottom_rx, Ry=HandSkeleton.finger_bottom_ry, Rz=deg90_bottom, name='pinky_start_r')
		ring_bottom = Joint(14, [ring_middle], HandSkeleton.bottom_rot, True, Rx=ring_bottom_rx, Ry=HandSkeleton.finger_bottom_ry, Rz=deg90_bottom, name='ring_start_r')
		middle_bottom = Joint(10, [middle_middle], HandSkeleton.bottom_rot, True, Rx=HandSkeleton.middle_bottom_rx, Ry=HandSkeleton.finger_bottom_ry, Rz=deg90_bottom, name='middle_start_r')
		index_bottom = Joint(6, [index_middle], HandSkeleton.bottom_rot, True, Rx=index_bottom_rx, Ry=HandSkeleton.finger_bottom_ry, Rz=deg90_bottom, name='index_start_r')
		thumb_bottom = Joint(2, [thumb_middle], HandSkeleton.full_rot, True, Rx=thumb_bottom_rx, Ry=thumb_bottom_ry, Rz=thumb_bottom_rz, name='thumb_start_r')
		# Z - up and down, X - left / right, Y - around
		# pinky = Joint(17, [pinky_bottom], HandSkeleton.full_rot, True, name='pinky r', Rx=[15,30], Ry=[10, 40], Rz=[-20, 10])
		# ring = Joint(13, [ring_bottom], HandSkeleton.full_rot, True, name='ring r', Rx=[0,15], Ry=[-10, 30], Rz=[-20, 10])
		# middle = Joint(9, [middle_bottom], HandSkeleton.full_rot, True, name='middle r', Rx=[-15,-5], Ry=[-10, 15], Rz=[-10, 10])
		# index = Joint(5, [index_bottom], HandSkeleton.full_rot, True, name='index r', Rx=[-30,-15], Ry=[-15, 15], Rz=[-10, 10])
		# thumb = Joint(1, [thumb_bottom], HandSkeleton.full_rot, True, name='thumb r', Rx=[-70,-60], Ry=[-15, 15], Rz=[-30, -10])

		pinky = Joint(17, [pinky_bottom], HandSkeleton.full_rot, True, name='pinky_root_r', Rx=[22,48], Ry=[5, 50], Rz=[-22, 7])
		ring = Joint(13, [ring_bottom], HandSkeleton.full_rot, True, name='ring_root_r', Rx=[12,40], Ry=[-7, 34], Rz=[-10, 10])
		middle = Joint(9, [middle_bottom], HandSkeleton.full_rot, True, name='middle_root_r', Rx=[-5,20], Ry=[-19, 26], Rz=[-5, 17])
		index = Joint(5, [index_bottom], HandSkeleton.full_rot, True, name='index_root_r', Rx=[-20,-3], Ry=[-22, 22], Rz=[-5, 16])
		thumb = Joint(1, [thumb_bottom], HandSkeleton.full_rot, True, name='thumb_root_r', Rx=[-70,-20], Ry=[-40, 0], Rz=[-40, 10])

		# more restrictive:
		# pinky = Joint(17, [pinky_bottom], HandSkeleton.full_rot, True, name='pinky r', Rx=[35,36], Ry=[-5, 30], Rz=[-22, 7])
		# ring = Joint(13, [ring_bottom], HandSkeleton.full_rot, True, name='ring r', Rx=[19,21], Ry=[-2, 2], Rz=[-10, 10])
		# middle = Joint(9, [middle_bottom], HandSkeleton.full_rot, True, name='middle r', Rx=[5,6], Ry=[-3, 3], Rz=[-5, 17])
		# index = Joint(5, [index_bottom], HandSkeleton.full_rot, True, name='index r', Rx=[-10,-9], Ry=[-4, 4], Rz=[-5, 16])
		# thumb = Joint(1, [thumb_bottom], HandSkeleton.full_rot, True, name='thumb r', Rx=[-31,-29], Ry=[-40, 0], Rz=[-40, 10])

		if enforce_wrist_constraints:
			root = Joint(0, [pinky, ring, index, middle, thumb], HandSkeleton.full_rot, True, Rz=[-80, 80], Rx=[-45, 45], Ry=[-180, 90], name='hand_wrist_r')
		else:
			root = Joint(0, [pinky, ring, index, middle, thumb], HandSkeleton.full_rot, Rx=HandSkeleton.free_joint, Ry=HandSkeleton.free_joint, Rz=HandSkeleton.free_joint, name='hand_wrist_r')

		if angles_constraints is not None: setAnglesConstraints(root, angles_constraints)
		return root

	def __init__(self, device, dtype=torch.float32, offsets_dict=None, proportions=None, offsets_constraints=None, angles_constraints=None, enforce_wrist_constraints=False, REMOVE_SCALE=False):
		super().__init__(device, dtype=dtype)
		self.num_hand_pts = 21
		self.offsets_dict, self.proportions, self.offsets_constraints = HandSkeletonNew.getOffsetsAndProportions(REMOVE_SCALE)
		if offsets_dict is not None: self.offsets_dict = copy.deepcopy(offsets_dict)
		if proportions is not None: self.proportions = copy.deepcopy(proportions)
		if offsets_constraints is not None: self.offsets_constraints = copy.deepcopy(offsets_constraints)
		self.scale_constraint = copy.deepcopy(self.offsets_constraints)

		key = list(self.offsets_dict.keys())[0]
		self.offsets_dict[key][1].append((self.offsets_dict[key][1][0][0]+self.num_hand_pts, 1))
		for k in range(len(self.proportions)):
			self.proportions[k]['nodes'].append(self.num_hand_pts+self.proportions[k]['nodes'][0])
			self.proportions[k]['signs'].append(1)

		self.root = HandSkeletonNew.getAngles(enforce_wrist_constraints, angles_constraints)
		self.root_left = copy.deepcopy(self.root)

		# if enforce_wrist_constraints: # being flipped later
		# 	self.root_left.ori_constraints[2] = -np.flip(self.root.ori_constraints[2])

		nodes = [self.root_left]
		while len(nodes) > 0:
			new_nodes = []
			for node in nodes:
				node.ori_constraints[0] = -np.flip(node.ori_constraints[0]) # change Z range
				# node.ori_constraints[1] = -np.flip(node.ori_constraints[1])
				node.ori_constraints[2] = -np.flip(node.ori_constraints[2])
				node.idx += self.num_hand_pts
				new_nodes += node.children
				node.name = node.name[:-1] + 'l'
			nodes = new_nodes

		# for k in range(len(self.root_left.children)):
		# 	self.root_left.children[k].ori_constraints[1] = -np.flip(self.root_left.children[k].ori_constraints[1])

		# print(self.offsets_dict, 'hand offsets after')
		self.num_nodes = 2*self.num_hand_pts
		self.roots = [self.root, self.root_left]
		self.setOffsets(self.roots, self.num_nodes, self.offsets_dict)
		self.num_hand_angles = len(self.angles_constraints)//2

		# flip_hand_angles_idxs = []
		flip_hand_angles_idxs = set()
		for i in range(self.num_hand_angles):
			if not (self.angles_constraints[i] == self.angles_constraints[i+self.num_hand_angles]).all():
				# print('angle', i, 'constraint', self.angles_constraints[i], self.angles_constraints[i+self.num_hand_angles], self.angle_idx2joint[i])
				flip_hand_angles_idxs.add(i)
				joint = self.angle_idx2joint[i][0]
				if 2 in self.node_angle_indexing[joint]: # add also roll angle
					flip_hand_angles_idxs.add(self.node_angle_indexing[joint][2])
		flip_hand_angles_idxs.add(0)
		flip_hand_angles_idxs.add(2)
		flip_hand_angles_idxs.add(6)
		flip_hand_angles_idxs.add(27)
		flip_hand_angles_idxs = list(flip_hand_angles_idxs)
		self.flip_hand_angles_idxs = torch.tensor(flip_hand_angles_idxs, dtype=torch.long)
		# print(flip_hand_angles_idxs)

		self.root_angles_idxs = list(self.node_angle_indexing[self.root.idx].values())
		self.num_non_virtual_nodes = self.num_nodes

	def getEdges(self):
		return HandRightSkeletonNew.getEdges_()

	def getOrigin(self, poses):
		return poses[...,[0],:]
	@staticmethod
	def getConnectionsHand_():
		return [(0, 1), (1, 2), (2, 3), (3, 4),
			 (0, 5), (5, 6), (6, 7), (7, 8),
			 (5, 9), (9, 10), (10, 11), (11, 12),
			 (9, 13), (13, 14), (14, 15), (15, 16),
			 (13, 17), (17, 18), (18, 19), (19, 20), (17, 0)]
	def getConnectionsHand(self):
		return HandSkeleton.getConnectionsHand_()
	def getConnections(self):
		hand1 = self.getConnectionsHand()
		hands = hand1 + [(v[0]+self.num_hand_pts, v[1]+self.num_hand_pts) for v in hand1]
		return hands
	def getName(self):
		return 'hands_new'
	def centralizePose(self, poses):
		if poses.shape[-2] == self.num_hand_pts:
			return poses - poses[...,[0],:]
		elif poses.shape[-2] == self.num_nodes:
			return torch.cat((poses[...,:self.num_hand_pts,:] - poses[...,[0],:], poses[...,self.num_hand_pts:,:] - poses[...,[self.num_hand_pts],:]),-2)
		assert False
	def extendPose(self, poses):
		return poses
	def extendProbs(self, weights):
		return weights
	def getJointWeight(self):
		weights = torch.ones(self.num_hand_pts, dtype=self.dtype)
		# weight wrist palm and finger tips
		weights[[0,5,17,4,8,12,16,20]] *= 2
		weights = torch.cat((weights, weights))
		return weights
	def getAngularWeights(self):
		weights = torch.ones(self.num_hand_angles, dtype=self.dtype)
		weights[self.root_angles_idxs] *= 10
		return torch.cat((weights, weights))
	def colorsLeft(self):
		return np.array([[255, 0, 0]]).repeat(self.num_hand_pts, 0)
	def colorsLeftEdges(self):
		return np.array([[100, 0, 0]]).repeat(len(self.getConnections())//2, 0)
	def colorsRight(self):
		return np.array([[0, 255, 0]]).repeat(self.num_hand_pts, 0)
	def colorsRightEdges(self):
		return np.array([[200, 50, 50]]).repeat(len(self.getConnections())//2, 0)
	def normalizeMLP(self, poses):
		poses = self.centralizePose(poses)
		return poses / (1e-10 + poses.abs().max(-1)[0].max(-1)[0][..., None, None])

class HandRightSkeletonNew(Skeleton):
	def __init__(self, device='cpu', dtype=torch.float32, offsets_dict=None, proportions=None, offsets_constraints=None, angles_constraints=None, enforce_wrist_constraints=False, REMOVE_SCALE=False):
		super().__init__(device, dtype=dtype)
		self.offsets_dict, self.proportions, self.offsets_constraints = HandSkeletonNew.getOffsetsAndProportions(REMOVE_SCALE)
		if offsets_dict is not None: self.offsets_dict = copy.deepcopy(offsets_dict)
		if proportions is not None: self.proportions = copy.deepcopy(proportions)
		if offsets_constraints is not None: self.offsets_constraints = copy.deepcopy(offsets_constraints)
		self.scale_constraint = copy.deepcopy(self.offsets_constraints)

		self.root = HandSkeletonNew.getAngles(enforce_wrist_constraints, angles_constraints)
		self.num_nodes = self.num_hand_pts = 21
		self.roots = [self.root]
		self.setOffsets(self.roots, self.num_nodes, self.offsets_dict)
		self.num_hand_angles = len(self.angles_constraints)
		self.root_angles_idxs = list(self.node_angle_indexing[self.root.idx].values())
		self.sk_part = SKELETON_PART.HAND_RIGHT
		self.num_non_virtual_nodes = self.num_nodes

		self.non_overlapping_segments = torch.tensor([[2, 3, 5,  6],
													  [2, 3, 6,  7],
													  [2, 3, 7,  8],

													  # top thumb to other fingers
													  [3, 4, 5,  6],
													  [3, 4, 6,  7],
													  [3, 4, 7,  8],

													  [3, 4, 9, 10],
													  [3, 4, 10,11],
													  [3, 4, 11,12],

													  [3, 4, 13,14],
													  [3, 4, 14,15],
													  [3, 4, 15,16],

													  [3, 4, 17,18],
													  [3, 4, 18,19],
													  [3, 4, 19,20],

													  # index to middle
													  [5, 6, 9, 10],
													  [5, 6, 10,11],
													  [5, 6, 11,12],

													  [6, 7, 9, 10],
													  [6, 7, 10,11],
													  [6, 7, 11,12],

													  [7, 8, 9, 10],
													  [7, 8, 10,11],
													  [7, 8, 11,12],

													  # middle to ring
													  [9,10, 13,14],
													  [9,10, 14,15],
													  [9,10, 15,16],

													  [10,11,13,14],
													  [10,11,14,15],
													  [10,11,15,16],

													  [11,12,13,14],
													  [11,12,14,15],
													  [11,12,15,16],

													  # ring to pinky
													  [13,14,17,18],
													  [13,14,18,19],
													  [13,14,19,20],

													  [14,15,17,18],
													  [14,15,18,19],
													  [14,15,19,20],

													  [15,16,17,18],
													  [15,16,18,19],
													  [15,16,19,20]])

	@staticmethod
	def getEdges_():
		return [(0, 1), (1, 2), (2, 3), (3, 4),
			 (0, 5), (5, 6), (6, 7), (7, 8),
			 (0, 9), (9, 10), (10, 11), (11, 12),
			 (0, 13), (13, 14), (14, 15), (15, 16),
			 (0, 17), (17, 18), (18, 19), (19, 20)]
	def getEdges(self):
		return HandRightSkeletonNew.getEdges_()
	def getConnections(self):
		return HandSkeleton.getConnectionsHand_()
	def getName(self):
		return 'right_hand_new'
	def centralizePose(self, poses):
		if poses.shape[-2] == self.num_hand_pts:
			return poses - poses[...,[0],:]
		assert False
	def extendPose(self, poses):
		return poses
	def extendProbs(self, weights):
		return weights
	def getJointWeight(self):
		weights = torch.ones(self.num_nodes, dtype=self.dtype)
		weights[[0,5,17,4,8,12,16,20]] *= 2
		return weights
	def getAngularWeights(self):
		weights = torch.ones(self.num_hand_angles, dtype=self.dtype)
		weights[self.root_angles_idxs] *= 10
		return weights
	def getOrigin(self, poses):
		return poses[...,[0],:]

class HandRightSkeletonMesh(Skeleton):
	def __init__(self, device='cpu', dtype=torch.float32, enforce_wrist_constraints=False, num_v_nodes_finger=4, circle_sc=0.10, circle_splits=5, circle_sc_wrist=None):
		super().__init__(device, dtype=dtype)

		self.offsets_dict, self.proportions, self.offsets_constraints = HandSkeletonNew.getOffsetsAndProportions(False)
		self.scale_constraint = copy.deepcopy(self.offsets_constraints)

		self.root = HandSkeletonNew.getAngles(enforce_wrist_constraints, None)
		self.num_nodes = self.num_hand_pts = 21
		self.roots = [self.root]

		levels = [self.roots]
		nodes = self.roots
		while len(nodes) > 0:
			new_nodes = []
			for node in nodes:
				new_nodes += node.children
			nodes = new_nodes
			levels.append(nodes)

		if circle_splits != 5:
			assert circle_splits % 2 == 0
			scale_range_top = [0.06, 0.2, 0.5, 1.0]
			circle_sc_top = [0.75, 0.87, 0.935, 1]
		else:
			circle_sc_top = [0.5, 0.65, 0.85, 1]
			scale_range_top = [0.05, 0.2, 0.5, 1]
		degrees = np.linspace(0, 2*np.pi, circle_splits+1)[:-1] # the first and last values for angles are the same
		nodes_cnt = self.num_nodes
		scale_range = np.linspace(0, 1, num_v_nodes_finger)
		node2idxs_dict = {}
		self.new_connections = []
		fingers_rescale = {'thumb': 1.35, 'index': 1.0, 'middle': 1.0, 'ring': 1.0, 'pinky': 0.8}
		self.faces = []

		assert len(scale_range_top) == num_v_nodes_finger
		assert len(circle_sc_top) == len(scale_range_top)
		for l, level in enumerate(levels[::-1]):
			# print('level', l, '/', len(levels))
			for node in level:
				# print(node.idx, 'num children', len(node.children), 'level idx', l)
				f_node_name, f_node_part = node.name.split('_')[:2]
				for child in node.children:
					f_child_name, f_child_part = child.name.split('_')[:2]
					# print(f_child_name, f_child_part)
					if f_child_part == 'root' or (f_child_name == 'thumb' and f_child_part == 'start'): continue
					if l == 2:
						scale_range_ = scale_range_top
					else:
						scale_range_ = scale_range
					for i in range(len(scale_range_)):
						# print('y offset', scale_range_[i], l, i)
						self.proportions.append({'ref': f'{f_child_name}_{f_child_part}_y', 'name': f'{f_node_name}_{f_node_part}_{i}_y', 'nodes': np.arange(nodes_cnt, nodes_cnt+circle_splits).tolist(), 'min': scale_range_[i], 'max': scale_range_[i], 'signs': (-np.ones(circle_splits)).tolist()})
						for d in range(circle_splits):
							node2idxs_dict[f'{f_node_name}_{f_node_part}_{i}_{d}'] = nodes_cnt
							# print('to node', node.name, 'add new child', f'{f_node_name}_{f_node_part}_{i}_{d}', 'use reference', f'{f_child_name}_{f_child_part}_y', 'node id', nodes_cnt, 'child', child.idx, 'node', node.idx, 'level', l)
							joint = Joint(nodes_cnt, [], HandSkeleton.no_rot, True, name=f'{f_node_name}_{f_node_part}_{i}_{d}')
							x, y = np.sin(degrees[d]), np.cos(degrees[d])
							if l == 2:
								circle_sc_ = circle_sc * circle_sc_top[i]
								level_c = 1
							else:
								magn = ((l-3)*num_v_nodes_finger + i - (l-3))
								# if l == 4 and i == 3:
								# 	print('big magn', l, i)
								# 	level_c = 1.11**magn
								# else:
								level_c = 1.04**magn
								if i == 0: level_c *= 1.005 # make slightly bigger, otherwise faces will be broken
								circle_sc_ = circle_sc
							# print(i, l)
							if circle_splits != 5:
								if (i == 0 or i == num_v_nodes_finger-1):
									if l == 2 and i == 0:
										pass
									else:
										# print('rescale')
										circle_sc_ *= 1.092
							# if f_node_name == 'index': print(level_c, circle_sc_, 'prod', level_c*circle_sc_, 'level', l, 'vertical', i)
							scale_x = level_c*circle_sc_*abs(x) * fingers_rescale[f_node_name]
							scale_y = level_c*circle_sc_*abs(y) * fingers_rescale[f_node_name]
							scale_x *= 0.8 # to make it more narrow, fingers are not cylinders
							self.proportions.append({'ref': 'index_root_y', 'name': f'{f_node_name}_{f_node_part}_{i}_{d}_x', 'nodes': [nodes_cnt], 'min': scale_x, 'max': scale_x, 'signs': [np.sign(x)]})
							self.proportions.append({'ref': 'index_root_y', 'name': f'{f_node_name}_{f_node_part}_{i}_{d}_z', 'nodes': [nodes_cnt], 'min': scale_y, 'max': scale_y, 'signs': [np.sign(y)]})

							if i != len(scale_range_)-1:
								self.new_connections.append([nodes_cnt, nodes_cnt+circle_splits])
								if d != circle_splits-1:
									# (nodes_cnt+1, nodes_cnt+circle_splits) are on diagonal
									self.faces.append([nodes_cnt, nodes_cnt+1, nodes_cnt+circle_splits])
									self.faces.append([nodes_cnt+1, nodes_cnt+circle_splits+1, nodes_cnt+circle_splits])
								else:
									self.faces.append([nodes_cnt, nodes_cnt-circle_splits+1, nodes_cnt+circle_splits])
									self.faces.append([nodes_cnt-circle_splits+1, nodes_cnt+1, nodes_cnt+circle_splits])

							if d != circle_splits-1:
								self.new_connections.append([nodes_cnt, nodes_cnt+1])
								if i == 0 and f_node_part == 'middle':
									self.faces.append([child.idx, nodes_cnt, nodes_cnt+1])
							else:
								self.new_connections.append([nodes_cnt, nodes_cnt-circle_splits+1])
								if i == 0 and f_node_part == 'middle':
									self.faces.append([child.idx, nodes_cnt, nodes_cnt-circle_splits+1])

							if f_node_part != 'middle' and i == 0:
								offs = circle_splits + 4*circle_splits*num_v_nodes_finger
								# print('offs', offs)
								if d != circle_splits-1:
									self.faces.append([nodes_cnt, nodes_cnt-offs, nodes_cnt-offs+1])
									self.faces.append([nodes_cnt+1, nodes_cnt,    nodes_cnt-offs+1])
								else:
									self.faces.append([nodes_cnt, nodes_cnt-offs,            nodes_cnt-offs-circle_splits+1])
									self.faces.append([nodes_cnt-circle_splits+1, nodes_cnt, nodes_cnt-offs-circle_splits+1])

							nodes_cnt += 1
							child.children.append(joint)

		######### add ellipse around the root node: ###########
		ellipse_splits = 6
		ellipse_width = 6
		ellipse_height = 4
		ellipse_degrees = np.linspace(0, 2*np.pi, ellipse_splits+1)[:-1] # the first and last values for angles are the same
		root_ellipse_idx = nodes_cnt
		if circle_sc_wrist is None: circle_sc_wrist = circle_sc
		for d in range(ellipse_splits):
			joint = Joint(nodes_cnt, [], HandSkeleton.no_rot, True, name=f'root_ellipse_{d}')
			x, y = ellipse_height * np.sin(ellipse_degrees[d]), ellipse_width * np.cos(ellipse_degrees[d])
			self.proportions.append({'ref': f'index_start_y', 'name': f'root_ellipse_{d}_x', 'nodes': [nodes_cnt], 'min': circle_sc_wrist*abs(x), 'max': circle_sc_wrist*abs(x), 'signs': [np.sign(x)]})
			self.proportions.append({'ref': f'index_start_y', 'name': f'root_ellipse_{d}_z', 'nodes': [nodes_cnt], 'min': circle_sc_wrist*abs(y), 'max': circle_sc_wrist*abs(y), 'signs': [np.sign(y)]})
			self.roots[0].children.append(joint)
			if d != ellipse_splits-1:
				self.new_connections.append([nodes_cnt, nodes_cnt+1])
				self.faces.append([0, nodes_cnt, nodes_cnt+1])
			else:
				self.new_connections.append([nodes_cnt, nodes_cnt-ellipse_splits+1])
				self.faces.append([0, nodes_cnt, nodes_cnt-ellipse_splits+1])
			nodes_cnt += 1
		########################################################

		# assert circle_splits == 5
		parts = ['thumb', 'index', 'middle', 'ring', 'pinky']
		if circle_splits == 5:
			for k, part in enumerate(parts[:-1]):
				part_next = parts[k+1]
				if part == 'thumb':
					n_idx_left = node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_2']
					n_idx_down = node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_1']
					n_idx_up = node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_3']

					c_idx_right = node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_0']
					c_idx_down = node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_1']
					c_idx_up = node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_4']
					# upper faces:
					self.faces.append([root_ellipse_idx+4, n_idx_up, c_idx_up])
					self.faces.append([root_ellipse_idx+4, n_idx_up+1, n_idx_up])
					# bottom faces:
					self.faces.append([root_ellipse_idx+2, c_idx_down, c_idx_down+1])
					self.faces.append([c_idx_down, root_ellipse_idx+2, n_idx_down])

				else:
					# next
					n_idx_left = node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_3']
					n_idx_down = node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_2']
					n_idx_up = node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_4']
					# center
					c_idx_right = node2idxs_dict[part+f'_root_{num_v_nodes_finger-1}_0']
					c_idx_down = node2idxs_dict[part+f'_root_{num_v_nodes_finger-1}_1']
					c_idx_up = node2idxs_dict[part+f'_root_{num_v_nodes_finger-1}_{circle_splits-1}']

					# upper faces:
					if part == 'ring':
						pass
					elif part == 'middle':
						self.faces.append([c_idx_up, root_ellipse_idx+4, root_ellipse_idx+5])
						self.faces.append([c_idx_up, root_ellipse_idx+5, n_idx_up])
					else:
						self.faces.append([c_idx_up, root_ellipse_idx+4, n_idx_up])
					if part_next == 'pinky':
						self.faces.append([node2idxs_dict[f'ring_root_{num_v_nodes_finger-1}_{circle_splits-1}'], root_ellipse_idx+5, n_idx_up])
						# self.faces.append([root_ellipse_idx+4, root_ellipse_idx+5, n_idx_up])
						# bottom:
						self.faces.append([root_ellipse_idx+1, n_idx_down-1, root_ellipse_idx+2])

					# bottom faces:
					self.faces.append([c_idx_down, root_ellipse_idx+2, n_idx_down])
					self.faces.append([n_idx_down, root_ellipse_idx+2, n_idx_down-1])

				self.new_connections.append([c_idx_right, n_idx_left])
				self.new_connections.append([c_idx_down, n_idx_down])
				self.new_connections.append([c_idx_up, n_idx_up])

				self.faces.append([c_idx_right, n_idx_left, n_idx_up])
				self.faces.append([n_idx_left, c_idx_right, n_idx_down])

				self.faces.append([c_idx_right, c_idx_down, n_idx_down])
				self.faces.append([c_idx_up, c_idx_right, n_idx_up])

			self.bottom_visual_connections = []

			self.bottom_visual_connections.append([node2idxs_dict[f'ring_root_{num_v_nodes_finger-1}_{circle_splits-1}'], root_ellipse_idx+ellipse_splits-1])
			self.bottom_visual_connections.append([node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_{circle_splits-1}'], root_ellipse_idx])
			self.bottom_visual_connections.append([node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_1'], root_ellipse_idx+1])
			# self.new_connections.append([node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_0'], root_ellipse_idx])
			# self.new_connections.append([node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_{circle_splits-1}'], root_ellipse_idx+ellipse_splits-1])
			# #### pinky
			# down part
			self.faces.append([node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_1'], root_ellipse_idx, root_ellipse_idx+1])
			# upper part
			self.faces.append([root_ellipse_idx, node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_{circle_splits-1}'], root_ellipse_idx+ellipse_splits-1])
			# middle upper part
			self.faces.append([node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_{circle_splits-1}'], root_ellipse_idx, node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_0']])
			# middle down part
			self.faces.append([node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_1'], root_ellipse_idx, node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_0']])

			######
			# bottom & up connections
			self.bottom_visual_connections.append([node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_2'], root_ellipse_idx+ellipse_splits//2-1])
			self.bottom_visual_connections.append([node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_0'], root_ellipse_idx+ellipse_splits//2+1])
			self.bottom_visual_connections.append([node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{circle_splits-2}'], root_ellipse_idx+ellipse_splits//2])
			# self.new_connections.append([node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{circle_splits-2}'], 1])
			# self.new_connections.append([1, root_ellipse_idx+ellipse_splits//2])

			# #### thumb - palm
			# self.faces.append([1, root_ellipse_idx+ellipse_splits//2-1, root_ellipse_idx+ellipse_splits//2])
			# self.faces.append([1, root_ellipse_idx+ellipse_splits//2+1, root_ellipse_idx+ellipse_splits//2])
			# self.faces.append([1, node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_2'], node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{circle_splits-2}']])
			# self.faces.append([1, node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{circle_splits-1}'], node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{circle_splits-2}']])
			# up
			# self.faces.append([1, node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{circle_splits-1}'], root_ellipse_idx+ellipse_splits//2+1])
			# down
			# self.faces.append([1, node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_2'], root_ellipse_idx+ellipse_splits//2-1])


			# top
			self.faces.append([node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{circle_splits-2}'], root_ellipse_idx+ellipse_splits//2, node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{circle_splits-1}']])
			self.faces.append([root_ellipse_idx+ellipse_splits//2, root_ellipse_idx+ellipse_splits//2+1, node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{circle_splits-1}']])
			# bottom
			self.faces.append([ root_ellipse_idx+ellipse_splits//2-1, node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_2'], node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{circle_splits-2}']])
			self.faces.append([root_ellipse_idx+ellipse_splits//2, root_ellipse_idx+ellipse_splits//2-1, node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{circle_splits-2}']])

			self.new_connections += self.bottom_visual_connections
			self.bottom_connections = [(0,5), (0,9), (0,13)] + self.bottom_visual_connections
		else:
			i_h = circle_splits // 2 # half
			i_q = circle_splits // 4 # quarter
			i_3q = i_q + i_h # 3 quarter
			for k, part in enumerate(parts[:-1]):
				part_next = parts[k+1]
				ext_s = 'start' if part == 'thumb' else 'root'
				if part != 'pinky':
					i_op = i_h + i_q
					for cnt, ii in enumerate(range(i_h+i_q, 2*i_h+i_q)):
						i_ = ii % circle_splits
						self.faces.append([node2idxs_dict[part+f'_{ext_s}_{num_v_nodes_finger-1}_{i_}'], node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_op}'], node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_op-1}']])
						self.faces.append([node2idxs_dict[part+f'_{ext_s}_{num_v_nodes_finger-1}_{i_}'], node2idxs_dict[part+f'_{ext_s}_{num_v_nodes_finger-1}_{(i_+1)%circle_splits}'], node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_op-1}']])
						i_op -= 1

					if part == 'thumb':
						for cnt, ii in enumerate(range(i_q, i_3q+1)):
							if ii == i_3q:
								self.faces.append([node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_{ii}'], root_ellipse_idx+4, root_ellipse_idx+3])
							elif ii == i_h:
								self.faces.append([node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_{ii}'], root_ellipse_idx+2, root_ellipse_idx+3])
							else:
								if ii < i_h:
									self.faces.append([node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_{ii}'], node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_{ii+1}'], root_ellipse_idx+2])
								else:
									self.faces.append([node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_{ii}'], node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_{ii+1}'], root_ellipse_idx+3])

						# thumb-palm
						self.faces.append([node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_{i_q}'], root_ellipse_idx+1, root_ellipse_idx+2])
						# thumb
						self.faces.append([node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_{i_h}'], node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_{i_h+1}'], root_ellipse_idx+3])
						# thumb-index upper
						self.faces.append([node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_{i_h+i_q}'], node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_h+i_q}'], root_ellipse_idx+4])

						# self.faces.append([node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_h+i_q}'], root_ellipse_idx+3, root_ellipse_idx+4])
						# self.faces.append([node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_{i_h+i_q}'], node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_h+i_q}'], root_ellipse_idx+3])

						# self.faces.append([node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_{i_h+i_q}'], node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_h+i_q-3}'], root_ellipse_idx+4])
						# self.faces.append([node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_h+i_q-3}'], node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_h+i_q-2}'], root_ellipse_idx+4])
						# self.faces.append([node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_h+i_q-2}'], node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_h+i_q-1}'], root_ellipse_idx+4])
						# self.faces.append([node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_h+i_q-1}'], node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_h+i_q}'], root_ellipse_idx+4])
						# thumb-index bottom
						self.faces.append([node2idxs_dict[part+f'_start_{num_v_nodes_finger-1}_{i_q}'], node2idxs_dict[part_next+f'_root_{num_v_nodes_finger-1}_{i_q}'], root_ellipse_idx+1])

				self.new_connections.append([node2idxs_dict[f'{part}_{ext_s}_{num_v_nodes_finger-1}_{i_3q}'], node2idxs_dict[f'{part_next}_root_{num_v_nodes_finger-1}_{i_3q}']])
				self.new_connections.append([node2idxs_dict[f'{part}_{ext_s}_{num_v_nodes_finger-1}_{i_q}'], node2idxs_dict[f'{part_next}_root_{num_v_nodes_finger-1}_{i_q}']])


			# index-middle upper
			self.faces.append([node2idxs_dict[f'index_root_{num_v_nodes_finger-1}_{i_h+i_q}'], node2idxs_dict[f'middle_root_{num_v_nodes_finger-1}_{i_h+i_q}'], root_ellipse_idx+5])
			# index upper
			self.faces.append([node2idxs_dict[f'index_root_{num_v_nodes_finger-1}_{i_h+i_q}'], root_ellipse_idx+5, root_ellipse_idx+4])

			# index-middle bottom
			self.faces.append([node2idxs_dict[f'index_root_{num_v_nodes_finger-1}_{i_q}'], node2idxs_dict[f'middle_root_{num_v_nodes_finger-1}_{i_q}'], root_ellipse_idx+1])
			# middle-ring upper
			self.faces.append([node2idxs_dict[f'middle_root_{num_v_nodes_finger-1}_{i_h+i_q}'], node2idxs_dict[f'ring_root_{num_v_nodes_finger-1}_{i_h+i_q}'], root_ellipse_idx+5])
			# ring-pinky upper
			self.faces.append([node2idxs_dict[f'ring_root_{num_v_nodes_finger-1}_{i_h+i_q}'], node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_{i_h+i_q}'], root_ellipse_idx+5])

			# middle-ring bottom
			self.faces.append([node2idxs_dict[f'middle_root_{num_v_nodes_finger-1}_{i_q}'], node2idxs_dict[f'ring_root_{num_v_nodes_finger-1}_{i_q}'], root_ellipse_idx+1])
			# ring-pinky bottom
			self.faces.append([node2idxs_dict[f'ring_root_{num_v_nodes_finger-1}_{i_q}'], node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_{i_q}'], root_ellipse_idx+1])

			part = 'pinky'
			for cnt, ii in enumerate(range(i_h+i_q, 2*i_h+i_q)):
				i_ = ii % circle_splits
				if i_ == i_h + i_q:
					self.faces.append([node2idxs_dict[part+f'_root_{num_v_nodes_finger-1}_{i_}'], root_ellipse_idx+5, root_ellipse_idx])
				elif i_ == 0:
					self.faces.append([node2idxs_dict[part+f'_root_{num_v_nodes_finger-1}_{i_}'], root_ellipse_idx, root_ellipse_idx+1])
				if cnt < i_q:
					self.faces.append([node2idxs_dict[part+f'_root_{num_v_nodes_finger-1}_{i_}'], node2idxs_dict[part+f'_root_{num_v_nodes_finger-1}_{(i_+1)%circle_splits}'], root_ellipse_idx])
				else:
					self.faces.append([node2idxs_dict[part+f'_root_{num_v_nodes_finger-1}_{i_}'], root_ellipse_idx+1, node2idxs_dict[part+f'_root_{num_v_nodes_finger-1}_{(i_+1)%circle_splits}']])
			self.new_connections.append([root_ellipse_idx+5, node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_{i_3q}']])
			self.new_connections.append([root_ellipse_idx, node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_0']])
			self.new_connections.append([root_ellipse_idx+1, node2idxs_dict[f'pinky_root_{num_v_nodes_finger-1}_{i_q}']])

			self.new_connections.append([root_ellipse_idx+4, node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{i_3q}']])
			self.new_connections.append([root_ellipse_idx+3, node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{i_h}']])
			self.new_connections.append([root_ellipse_idx+2, node2idxs_dict[f'thumb_start_{num_v_nodes_finger-1}_{i_q}']])


		self.num_nodes = self.num_hand_pts = nodes_cnt
		# print('mesh num nodes', self.num_nodes)
		self.setOffsets(self.roots, self.num_nodes, self.offsets_dict)
		self.num_hand_angles = len(self.angles_constraints)
		self.root_angles_idxs = list(self.node_angle_indexing[self.root.idx].values())
		self.sk_part = SKELETON_PART.HAND_RIGHT
		self.faces = np.array(self.faces)

	def getConnections(self):
		return self.new_connections
		return HandSkeleton.getConnectionsHand_()

	def copyOffsetsFromSkeleton(self, skeleton_offsets, mesh_offsets=None, clone_skeleton_offset=True):
		assert skeleton_offsets.ndim <= 2
		if mesh_offsets is None:
			mesh_offsets = self.getOffsetsValues().clone().to(skeleton_offsets.device)
			if skeleton_offsets.ndim == 2:
				mesh_offsets = mesh_offsets[None].repeat(len(skeleton_offsets),1)
		mesh_offsets[..., :skeleton_offsets.shape[-1]] = skeleton_offsets.clone() if clone_skeleton_offset else skeleton_offsets
		return mesh_offsets


class HumanSkeleton(Skeleton):
	def __init__(self, device, name:str, is_joint:bool=False, skeleton_body=None, skeleton_hand=None, enforce_wrist_constraints=False):
		super().__init__(device)
		self.name = name
		self.sk_part = SKELETON_PART.FULL
		if name != 'human3.6_full':
			if name == 'mediapipe':
				self.skeleton_hand = HandSkeleton(device, enforce_wrist_constraints=enforce_wrist_constraints) if skeleton_hand is None else skeleton_hand
			else:
				self.skeleton_hand = HandSkeletonNew(device, enforce_wrist_constraints=enforce_wrist_constraints) if skeleton_hand is None else skeleton_hand
		if skeleton_body is None:
			if name == 'mediapipe_spine':
				from media_pipe.skeleton_mediapipe import BodyMediaPipeSkeletonForSignLanguageSpine
				self.skeleton_body = BodyMediaPipeSkeletonForSignLanguageSpine(device)
			elif name == 'mediapipe_orig':
				from media_pipe.skeleton_mediapipe import BodySkeletonMediapipe
				self.skeleton_body = BodySkeletonMediapipe(device)
				self.num_nodes = self.skeleton_body.num_nodes + self.skeleton_hand.num_nodes
				self.num_body_pts = self.skeleton_body.num_nodes
				return
			else: assert False, name
		else: self.skeleton_body = skeleton_body
		self.num_virtual_nodes = self.skeleton_body.num_virtual_nodes + (self.skeleton_hand.num_virtual_nodes if self.skeleton_hand is not None else 0)
		self.roots = copy.deepcopy(self.skeleton_body.roots)
		hand_roots = copy.deepcopy(self.skeleton_hand.roots)
		self.enforce_wrist_constraints = enforce_wrist_constraints
		nodes = hand_roots
		while len(nodes) > 0:
			new_nodes = []
			for node in nodes:
				node.idx += self.skeleton_body.num_nodes
				new_nodes += node.children
			nodes = new_nodes
		nodes = self.roots
		while len(nodes) > 0:
			new_nodes = []
			for node in nodes:
				if name == 'openpose':
					if node.idx == 4: node.children.append(hand_roots[0])
					if node.idx == 7: node.children.append(hand_roots[1])
				elif name == 'mediapipe' or name == 'mediapipe_virtual' or name == 'mediapipe_new' or name == 'mediapipe_spine':
					if node.idx == 2: node.children.append(hand_roots[0])
					if node.idx == 5: node.children.append(hand_roots[1])
				elif name == 'panoptic':
					if node.idx == 11: node.children.append(hand_roots[0])
					if node.idx == 5: node.children.append(hand_roots[1])
				else: assert False, name
				new_nodes += node.children
			nodes = new_nodes

		offsets_dict_hand = self.skeleton_hand.offsets_dict.copy()
		for k in offsets_dict_hand.keys():
			offsets_dict_hand[k] = [offsets_dict_hand[k][0], [(v[0]+self.skeleton_body.num_nodes, v[1]) for v in offsets_dict_hand[k][1]]]

		if hasattr(self.skeleton_body, 'proportions'):
			self.proportions = copy.deepcopy(self.skeleton_body.proportions)
		if hasattr(self.skeleton_hand, 'proportions'):
			hand_proportions = copy.deepcopy(self.skeleton_hand.proportions)
			self.proportions = [] if not hasattr(self, 'proportions') else self.proportions
			for hp in hand_proportions:
				for k in range(len(hp['nodes'])):
					hp['nodes'][k] += self.skeleton_body.num_nodes
				self.proportions.append(hp)
		self.has_proportions = hasattr(self, 'proportions')
		self.num_body_pts, self.num_hand_pts = self.skeleton_body.num_nodes, self.skeleton_hand.num_hand_pts
		self.num_body_angles, self.num_hand_angles = self.skeleton_body.getNumberOfAngles(), self.skeleton_hand.getNumberOfAngles()
		self.offsets_dict = {**self.skeleton_body.offsets_dict.copy(), **offsets_dict_hand}
		if hasattr(self.skeleton_body, 'scale_constraint') and hasattr(self.skeleton_hand, 'scale_constraint'):
			self.offsets_constraints = self.skeleton_body.scale_constraint + self.skeleton_hand.scale_constraint
		self.num_nodes = self.skeleton_body.num_nodes + self.skeleton_hand.num_nodes
		self.setOffsets(self.roots, self.num_nodes, self.offsets_dict)
		self.skeletons = [self] if is_joint else [self.skeleton_body, self.skeleton_hand]
		self.is_joint = is_joint

		if self.getName() == 'mediapipe' or self.getName() == 'mediapipe_virtual' or self.getName() == 'mediapipe_new' or self.getName() == 'mediapipe_spine':
			self.idx_right_wrist, self.idx_left_wrist = 2, 5
		else: assert False
		self.num_body_pts_no_virtual = self.skeleton_body.num_nodes-self.skeleton_body.num_virtual_nodes

	# angles_tensor_idxs must be ordered!!!!!!!!!!
	def getBodyAngles(self, angles):
		if angles.shape[-1] == len(self.angles_tensor_idxs):
			return angles[..., :len(self.skeleton_body.angles_tensor_idxs)]
		else: assert False
	def getHandAngles(self, angles, flip_left=False):
		if angles.shape[-1] == len(self.angles_tensor_idxs):
			angles_hands = angles[..., len(self.skeleton_body.angles_tensor_idxs):]
			if flip_left:
				angles2 = angles_hands.reshape(-1, 2, self.skeleton_hand.num_hand_angles).clone()
				angles2[:,1,self.skeleton_hand.flip_hand_angles_idxs] *= -1.0
				angles_hands = angles2.reshape(len(angles), -1)
			return angles_hands
		else: assert False
	def getRightHandAngles(self, angles):
		if angles.shape[-1] == len(self.angles_tensor_idxs):
			return angles[..., len(self.skeleton_body.angles_tensor_idxs):len(self.skeleton_body.angles_tensor_idxs)+len(self.skeleton_hand.angles_tensor_idxs)//2]
		else: assert False
	def getLeftHandAngles(self, angles, flip=False):
		if angles.shape[-1] == len(self.angles_tensor_idxs):
			if flip:
				angles_ = angles[..., len(self.skeleton_body.angles_tensor_idxs)+len(self.skeleton_hand.angles_tensor_idxs)//2:].clone()
				angles_[...,self.skeleton_hand.flip_hand_angles_idxs] *= -1.0
				return angles_
			else:
				return angles[..., len(self.skeleton_body.angles_tensor_idxs)+len(self.skeleton_hand.angles_tensor_idxs)//2:]
		else: assert False

	def getBodyOrder(self):
		return [12,14,16,11,13,15,24,23,0,1,2,3,4,5,6,7,8,9,10]

	def getConnections_(self, with_virtual):
		connections_hand = self.skeleton_hand.getConnections()
		connections_body = self.skeleton_body.getConnectionsVirtual() if with_virtual else self.skeleton_body.getConnections()
		num_pts_body = self.skeleton_body.num_nodes if with_virtual else self.skeleton_body.num_nodes - self.skeleton_body.num_virtual_nodes
		return connections_body + [(num_pts_body+x, num_pts_body+y) for (x,y) in connections_hand]

	def getConnections(self):
		return self.getConnections_(False)

	def getConnectionsVirtual(self):
		return self.getConnections_(True)

	def getFaceConnections(self):
		face_connections = []
		if self.name == 'openpose':
			for i in range(16): face_connections.append((i, i+1))
			for i in range(17, 21, 1): face_connections.append((i, i+1))
			for i in range(22, 26, 1): face_connections.append((i, i+1))
			for i in range(27, 30, 1): face_connections.append((i, i+1))
			for i in range(31, 35, 1): face_connections.append((i, i+1))
			for i in range(36, 41, 1): face_connections.append((i, i+1))
			face_connections.append((41,36))
			for i in range(42, 47, 1): face_connections.append((i, i+1))
			face_connections.append((47,42))
			for i in range(48, 59, 1): face_connections.append((i, i+1))
			face_connections.append((59,48))
			for i in range(60, 67, 1): face_connections.append((i, i+1))
			face_connections.append((67,60))
			face_connections += [(48,60), (54,64)]
			face_connections += [(37,68), (68,38), (43, 69), (69, 44)]
		return face_connections
	def getConnectionsWithFace(self):
		connections = self.getConnections()
		face_connections = self.getFaceConnections()
		num_pose_pts = self.skeleton_body.num_nodes + self.skeleton_hand.num_nodes
		return connections + [(num_pose_pts+x, num_pose_pts+y) for (x,y) in face_connections]
	def getName(self):
		return self.name
	def centralizePose(self, poses):
		return self.concatePoses(torch.cat((self.skeleton_body.centralizePose(self.getBody(poses)), self.getHands(poses)), -2))

	def transformPose(self, poses):
		if poses.shape[-2] == self.num_nodes:
			poses_new = self.centralizePose(poses)
		elif poses.shape[-2] + self.num_virtual_nodes == self.num_nodes:
			poses_new = self.centralizePose(self.extendPose(poses))
		else: assert False
		return poses_new / torch.std(poses_new, axis=[-2,-1])[...,None,None]
	def extendPose(self, poses):
		return torch.cat((self.skeleton_body.extendPose(poses[...,:self.num_body_pts-self.skeleton_body.num_virtual_nodes,:]), poses[...,self.num_body_pts-self.skeleton_body.num_virtual_nodes:,:]), -2)
	def subSamplePose(self, pose):
		return torch.cat((self.skeleton_body.subSamplePose(pose[...,:self.num_body_pts,:]), self.skeleton_hand.subSamplePose(pose[...,self.num_body_pts:,:])),-2)
	def colors(self):
		return np.concatenate((self.skeleton_body.colors(), self.skeleton_hand.colorsRight(), self.skeleton_hand.colorsLeft()))
	def colorsEdges(self):
		return np.concatenate((self.skeleton_body.colorsEdges(), self.skeleton_hand.colorsRightEdges(), self.skeleton_hand.colorsLeftEdges()))
	def concatePoses(self, poses):
		poses[...,-2*self.num_hand_pts:-self.num_hand_pts,:] -= poses[...,[-2*self.num_hand_pts],:] - poses[...,[self.idx_right_wrist],:]
		poses[...,-  self.num_hand_pts:                  ,:] -= poses[...,[-  self.num_hand_pts],:] - poses[...,[self.idx_left_wrist ],:]
		return poses

	def getBody(self, poses):
		if poses.shape[-2] == self.num_nodes:
			return poses[..., :self.num_body_pts, :]
		elif poses.shape[-2] == self.num_nodes - self.num_virtual_nodes:
			return poses[...,:self.num_body_pts-self.num_virtual_nodes, :]
		assert False
	def getHands(self, poses, all_right=False):
		if all_right:
			return torch.cat((self.getRightHands(poses), self.getLeftHands(poses, flip=True)), 0)
		else:
			return poses[..., -2*self.num_hand_pts:, :]

	def getRightHands(self, poses):
		return poses[...,-2*self.num_hand_pts:-self.num_hand_pts,:]

	def getLeftHands(self, poses, flip=False):
		left_hands = poses[:,-self.num_hand_pts:]
		if flip:
			left_hands = left_hands.clone()
			left_hands[...,0] *= -1
		return left_hands

	def mergeOffsets(self, offsets_body, offsets_hand):
		if self.has_proportions:
			return torch.cat((offsets_body[...,[0]], offsets_hand[...,[0]], offsets_body[...,1:], offsets_hand[...,1:]),-1)
		else:
			raise NotImplementedError

	def getBodyOffsets(self, offsets):
		if self.has_proportions:
			return torch.cat((offsets[...,[0]], offsets[...,2:2+self.skeleton_body.offsets_constraints.shape[0]-1]),-1)
		return offsets[...,:self.skeleton_body.offsets_constraints.shape[0]]

	def getHandOffsets(self, offsets):
		if self.has_proportions:
			return torch.cat((offsets[...,[1]], offsets[...,2+self.skeleton_body.offsets_constraints.shape[0]-1:]),-1)
		return offsets[...,self.skeleton_body.offsets_constraints.shape[0]:]

	def getWeightsFor2D(self):
		# seems like uniform weights works the best for HRNet detector
		weights = torch.ones(self.skeleton_body.num_nodes + self.skeleton_hand.num_nodes)
		weights[self.skeleton_body.num_nodes:] *= 1.0
		return weights
	def extendProbs(self, probs):
		return torch.cat((self.skeleton_body.extendProbs(self.getBody(probs)), self.skeleton_hand.extendProbs(self.getHands(probs))), -2)
	def normalizeMLP(self, poses):
		return self.skeleton_body.normalizeMLP(torch.cat((self.skeleton_body.extendPose(self.getBody(poses)), self.getHands(poses)),-2))
