"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import copy
import torch
import numpy as np
from skeleton_human36 import Human36Skeleton
from skeleton import Joint, Skeleton, SKELETON_PART, setAnglesConstraints

class BodyMediaPipeSkeletonForSignLanguageNew(Skeleton):
	def __init__(self, device, dtype=torch.float32, offsets_dict=None, proportions=None, offsets_constraints=None, angles_constraints=None, REMOVE_SCALE=False, init=True):
		super().__init__(device, dtype)
		self.sk_part = SKELETON_PART.BODY
		self.proportions = []
		if not init: return
		from utils import PrintColors
		print(PrintColors.WARNING+'Warning! BodyMediaPipeSkeletonForSignLanguageNew is deprecated!'+PrintColors.ENDC)

		self.offsets_dict = {'neck_hip_y' : [0.6, [(19, 1)]]}
		self.proportions.append({'ref': 'neck_hip_y', 'name': 'shoulder_x', 'nodes': [3,0], 'min': 3.002E-01, 'max': 3.949E-01, 'signs': [1,-1]})
		self.proportions.append({'ref': 'neck_hip_y', 'name': 'wrist_y', 'nodes': [2,5], 'min': 4.104E-01, 'max': 5.704E-01, 'signs': [1,1]})
		self.proportions.append({'ref': 'neck_hip_y', 'name': 'nose_y', 'nodes': [8], 'min': 3.067E-01, 'max': 4.950E-01, 'signs': [-1]})
		self.proportions.append({'ref': 'neck_hip_y', 'name': 'hip_x', 'nodes': [6,7], 'min': 3.941E-01, 'max': 4.949E-01, 'signs': [-.5,.5]})
		self.proportions.append({'ref': 'neck_hip_y', 'name': 'elbow_y', 'nodes': [1,4], 'min': 4.913E-01, 'max': 6.6532E-01, 'signs': [1,1]})

		self.proportions.append({'ref': 'neck_hip_y', 'name': 'ear_x', 'nodes': [16,15], 'min': 1.003E-01, 'max': 2.663E-01, 'signs': [-1,1]})
		self.proportions.append({'ref': 'neck_hip_y', 'name': 'ear_y', 'nodes': [16,15], 'min': 0.100E-03, 'max': 8.286E-02, 'signs': [1,1]})
		self.proportions.append({'ref': 'neck_hip_y', 'name': 'ear_z', 'nodes': [16,15], 'min': 2.001E-03, 'max': 2.363E-01, 'signs': [1,1]})

		self.proportions.append({'ref': 'neck_hip_y', 'name': 'mouth_x', 'nodes': [18,17], 'min': 0.495E-04, 'max': 5.343E-02, 'signs': [-1,1]})
		self.proportions.append({'ref': 'neck_hip_y', 'name': 'mouth_y', 'nodes': [18,17], 'min': 4.079E-02, 'max': 8.741E-02, 'signs': [1,1]})
		self.proportions.append({'ref': 'neck_hip_y', 'name': 'mouth_z', 'nodes': [18,17], 'min': 0.305E-03, 'max': 6.967E-02, 'signs': [1,1]})

		self.proportions.append({'ref': 'neck_hip_y', 'name': 'eye_inner_x', 'nodes': [14,11,9,12], 'min': 1.e-05, 'max': 3.4e-02, 'signs': [-1,1,-1,1]})
		self.proportions.append({'ref': 'neck_hip_y', 'name': 'eye_x', 'nodes': [13,10], 'min': 5.193E-02 , 'max': 8.561E-02, 'signs': [-1,1]})
		self.proportions.append({'ref': 'neck_hip_y', 'name': 'eye_y', 'nodes': [13,10], 'min': 3.363E-02, 'max': 8.156E-02, 'signs': [-1,-1]})
		self.proportions.append({'ref': 'neck_hip_y', 'name': 'eye_z', 'nodes': [13,10], 'min': 1.114E-03, 'max': 8.651E-02, 'signs': [1,1]})

		self.offsets_constraints = [[1.0, 1.0]] if REMOVE_SCALE else [[0.3, 0.8]]

		if offsets_dict is not None: self.offsets_dict = copy.deepcopy(offsets_dict)
		if proportions is not None: self.proportions = copy.deepcopy(proportions)
		if offsets_constraints is not None: self.offsets_constraints = copy.deepcopy(offsets_constraints)

		self.scale_constraint = copy.deepcopy(self.offsets_constraints)

		shoulder_shoulder_z = [-50, 45]
		elbow_joint_y = [-90, 45]
		shoulder_y = [-40, 60]

		free_joint = [-180, 180]
		elbow_joint_x = [-180, 80]
		elbow_joint_z = [-30, 160]
		deg90_joint = [-90, 90]
		elbow_joint = [-160, 0]
		hip_hip_z = [-25, 25]
		hip_x = [-60, 60]
		hip_y = [-45, 45]
		hip_z = [-15, 15]
		full_rot = [True, True, True]
		no_rot = [False, False, False]
		ankle_elbow_rot = [False, True, False]
		wrist_right = Joint(2, [], ankle_elbow_rot, True, Rx=-np.flip(Human36Skeleton.elbow_joint), name='wrist r')
		wrist_left  = Joint(5, [], ankle_elbow_rot, True, Rx=-np.flip(Human36Skeleton.elbow_joint), name='wrist l')

		eye_right_inner_left  = Joint(12, [], no_rot, True)
		eye_right_inner_right = Joint(14,[], no_rot, True)
		eye_right = Joint(13, [eye_right_inner_left, eye_right_inner_right], no_rot, True)
		ear_right = Joint(16, [], no_rot, True)
		eye_left_inner_left  = Joint(9, [], no_rot, True)
		eye_left_inner_right = Joint(11,[], no_rot, True)
		eye_left = Joint(10, [eye_left_inner_left, eye_left_inner_right], no_rot, True)
		ear_left = Joint(15, [], no_rot, True)
		mouth_right = Joint(18, [], no_rot, True)
		mouth_left = Joint(17, [], no_rot, True)
		hip_left = Joint(7, [], no_rot, True) # Z -left-right X - bottom-up, Y is around
		hip_right = Joint(6, [], no_rot, True) # Z -left-right X - bottom-up, Y is around
		nose = Joint(8, [mouth_left, mouth_right, ear_left, ear_right, eye_left, eye_right], full_rot, True, Rx=np.array(deg90_joint)+np.array([20,20]), Ry=deg90_joint, Rz=np.array(deg90_joint))

		elbow_right = Joint(1, [wrist_right], full_rot, True, Rx=elbow_joint_x, Ry=elbow_joint_y, Rz=elbow_joint_z) # Z-left-right, X - bottom-up, Y-around
		elbow_left  = Joint(4, [wrist_left ], full_rot, True, Rx=elbow_joint_x, Ry=-np.flip(elbow_joint_y), Rz=-np.flip(elbow_joint_z))
		shoulder_left  = Joint(3, [elbow_left], [True, False, True], True, Rz=-np.flip(shoulder_shoulder_z), Ry=-np.flip(shoulder_y))
		shoulder_right = Joint(0, [elbow_right], [True, False, True], True, Rz=shoulder_shoulder_z, Ry=shoulder_y)

		# elbow_right = Joint(1, [wrist_right], full_rot, True, Rx=-np.flip(shoulder_joint_x), Ry=-np.flip(shoulder_joint_y), Rz=-np.flip(shoulder_joint_z)) # Z-left-right, X - bottom-up, Y-around
		# elbow_left  = Joint(4, [wrist_left ], full_rot, True, Rx=-np.flip(shoulder_joint_x), Ry=-np.flip(shoulder_joint_y), Rz=-np.flip(shoulder_joint_z))
		# shoulder_left  = Joint(3, [elbow_left], [True, False, True], True, Rz=shoulder_shoulder_z, Ry=shoulder_y)
		# shoulder_right = Joint(0, [elbow_right], [True, False, True], True, Rz=shoulder_shoulder_z, Ry=np.array(shoulder_y)+np.pi)

		hip_center = Joint(19, [hip_left, hip_right], full_rot, True, Rx=hip_x, Ry=hip_y, Rz=hip_z)
		# self.root = Joint(20, [shoulder_right, shoulder_left, nose, hip_center], full_rot, False, Rx=[-25, 25], Ry=[-135, 135], Rz=[-25, 25]) # full rot for training mlp
		self.root = Joint(20, [shoulder_right, shoulder_left, nose, hip_center], full_rot, False, Rx=[-25, 25], Ry=[-135, 135], Rz=[-25, 25]) # full rot for training mlp

		if angles_constraints is not None: setAnglesConstraints(self.root, angles_constraints)

		self.roots = [self.root]
		self.num_nodes = 21
		self.setOffsets(self.roots, self.num_nodes, self.offsets_dict)
		self.skeletons = [self]
		self.subsample_points = np.ones((self.num_nodes), dtype=bool)
		self.subsample_points[[19,20]] = False
		self.num_virtual_nodes = 2
		self.root_angles_idxs = list(self.node_angle_indexing[self.root.idx].values())
		self.num_non_virtual_nodes = self.num_nodes - self.num_virtual_nodes

	def subSamplePose(self, pose):
		# return pose[...,self.subsample_points,:]
		if pose.shape[-2] == self.num_nodes - self.num_virtual_nodes:
			return pose
		elif pose.shape[-2] == self.num_nodes:
			return pose[...,self.subsample_points,:]
		else:
			raise NotImplementedError

	def getEdges(self):
		return [(0,1),(0,20),(1,2),(20,8),(6,7),(19,20),(8,15,0),(8,15,1),(8,15,2),(8,18,0),(8,18,1),(8,18,2),(13,14),(8,10,0),(8,10,1),(8,10,2)]

	def getConnections(self):
		return [(0,1),(1,2), # arm right
			(0,3), # shoulder-shoulder
			(3,4),(4,5), # arm left
			(0,6),(6,7),(7,3), # shoulders-hips, hip-hip
			(8,9),(9,10),(10,11),(11,15),(8,12),(12,13),(13,14),(14,16),(17,18)] # head
	def getName(self):
		return 'body_mediapipe_sign'
	def getBodyOrder(self):
		return [12,14,16,11,13,15,24,23,0,1,2,3,4,5,6,7,8,9,10]
	def extendPose(self, poses):
		if poses.shape[-2] == self.num_non_virtual_nodes:
			return torch.cat((poses, .5*(poses[...,[6],:]+poses[...,[7],:]), .5*(poses[...,[0],:]+poses[...,[3],:])), -2)
		else:
			return poses
	def extendProbs(self, weights):
		# weights should be ... x NUM_JOINTS x 1
		assert weights.shape[-1] == 1
		return torch.cat((weights, weights[...,[6],:]*weights[...,[7],:], weights[...,[0],:]*weights[...,[3],:]), -2)
	def centralizePose(self, poses):
		# center origin in the middle of hips, and standard deviation for each pose is 1
		# return poses - poses[...,[19],:]
		# try origin point
		return poses - poses[..., [self.root.idx], :]

	def transformPose(self, poses):
		# poses is ... x NUM_POINTS x 2
		if poses.shape[-2] == self.num_nodes:
			poses_new = self.centralizePose(poses)
		elif poses.shape[-2] == self.num_non_virtual_nodes:
			poses_new = self.centralizePose(self.extendPose(poses))
		else: assert False
		return poses_new / torch.std(poses_new, axis=[-2,-1])[...,None,None]

	def getJointWeight(self):
		weights = torch.ones(self.num_nodes, dtype=self.dtype)
		weights[[0,3]] *= 5 # weight shoulder
		weights[[2,5]] *= 5 # weight wrists
		weights[19] *= 3 # weight center of a hip
		head_pts = [i for i in range(8, 19, 1)]
		weights[head_pts] /= len(head_pts) # decrease weight of head points
		return weights

	def getAngularWeights(self):
		weights = torch.ones(len(self.angles_constraints), dtype=self.dtype)
		weights[self.root_angles_idxs] *= 10
		return weights
	def getBody(self, pose):
		return pose

	def colors(self):
		colors = np.array([[0,0,255]]).repeat(self.num_nodes,0)
		colors[[3,4,5,7,9,10,11,15,17]] = np.array([255,0,255])
		return colors

	def colorsEdges(self):
		colors = np.array([[100,100,0]]).repeat(len(self.getConnections()),0)
		colors[[3,4,7,9,10,11,12]] = np.array([100,0,100])
		return colors

	def getWeightsFor2D(self):
		return torch.ones(self.num_nodes)

	def normalizeMLP(self, poses_in):
		if poses_in.shape[-2] == self.num_non_virtual_nodes:
			poses = self.centralizePose(self.extendPose(poses_in))
		else:
			poses = self.centralizePose(poses_in)
		return poses / (1e-10 + poses.abs()[...,19,1][...,None,None])


class BodyMediaPipeSkeletonForSignLanguageSpine(BodyMediaPipeSkeletonForSignLanguageNew):
	def __init__(self, device='cpu', dtype=torch.float32, offsets_dict=None, proportions=None, offsets_constraints=None, angles_constraints=None, REMOVE_SCALE=False):
		super(BodyMediaPipeSkeletonForSignLanguageSpine, self).__init__(device, dtype, None, None, None, None, False, init=False)
		self.offsets_dict = {'elbow_y' : [0.31, [(1, 1), (4, 1)]]} # elbow-shoulder should be more fixed / constant and visible during the sequence than neck - hips
		self.proportions.append({'ref': 'elbow_y', 'name': 'shoulder_x', 'nodes': [3,0], 'min':3.6923e-01, 'max': 4.9955e-01, 'signs': [1,-1]})
		self.proportions.append({'ref': 'elbow_y', 'name': 'wrist_y', 'nodes': [2,5], 'min': 4.8334e-01, 'max': 6.5394e-01, 'signs': [1,1]})
		self.proportions.append({'ref': 'elbow_y', 'name': 'nose_y', 'nodes': [8], 'min': 4.6283e-01, 'max': 6.2617e-01, 'signs': [-1]})
		self.proportions.append({'ref': 'elbow_y', 'name': 'hip_x', 'nodes': [6,7], 'min': 3.1431e-01, 'max': 4.2525e-01, 'signs': [-1,1]})
		self.proportions.append({'ref': 'elbow_y', 'name': 'spine_y', 'nodes': [19,21], 'min': 6.0538e-01, 'max': 8.1904e-01, 'signs': [1,1]})

		self.proportions.append({'ref': 'elbow_y', 'name': 'ear_x', 'nodes': [16,15], 'min': 1.5104e-01, 'max': 2.0434e-01, 'signs': [-1,1]})
		self.proportions.append({'ref': 'elbow_y', 'name': 'ear_y', 'nodes': [16,15], 'min': 1.3126e-03, 'max': 1.7758e-03, 'signs': [1,1]})
		self.proportions.append({'ref': 'elbow_y', 'name': 'ear_z', 'nodes': [16,15], 'min': 1.8669e-01, 'max': 2.5258e-01, 'signs': [1,1]})

		self.proportions.append({'ref': 'elbow_y', 'name': 'mouth_x', 'nodes': [18,17], 'min': 4.9957e-02, 'max': 6.7589e-02, 'signs': [-1,1]})
		self.proportions.append({'ref': 'elbow_y', 'name': 'mouth_y', 'nodes': [18,17], 'min': 6.1739e-02, 'max': 8.3529e-02, 'signs': [1,1]})
		self.proportions.append({'ref': 'elbow_y', 'name': 'mouth_z', 'nodes': [18,17], 'min': 2.3333e-04, 'max': 3.1567e-04, 'signs': [1,1]})

		self.proportions.append({'ref': 'elbow_y', 'name': 'eye_inner_x', 'nodes': [14,11,9,12], 'min': 3.1348e-02, 'max': 4.2412e-02, 'signs': [-1,1,-1,1]})
		self.proportions.append({'ref': 'elbow_y', 'name': 'eye_x', 'nodes': [13,10], 'min': 8.0045e-02, 'max': 1.0830e-01, 'signs': [-1,1]})
		self.proportions.append({'ref': 'elbow_y', 'name': 'eye_y', 'nodes': [13,10], 'min': 7.4891e-02, 'max': 1.0132e-01, 'signs': [-1,-1]})
		self.proportions.append({'ref': 'elbow_y', 'name': 'eye_z', 'nodes': [13,10], 'min': 7.2010e-02, 'max': 9.7426e-02, 'signs': [1,1]})

		self.offsets_constraints = [[1.0, 1.0]] if REMOVE_SCALE else [[0.10, 0.5]]

		if offsets_dict is not None: self.offsets_dict = copy.deepcopy(offsets_dict)
		if proportions is not None: self.proportions = copy.deepcopy(proportions)
		if offsets_constraints is not None: self.offsets_constraints = copy.deepcopy(offsets_constraints)

		self.scale_constraint = copy.deepcopy(self.offsets_constraints)

		shoulder_shoulder_z = [-50, 45]
		elbow_joint_y = [-90, 45]
		shoulder_y = [-40, 60]

		chest_rot = [-40, 25]

		elbow_joint_x = [-180, 80]
		elbow_joint_z = [-30, 160]
		deg90_joint = [-90, 90]
		hip_x = [-60, 60]
		hip_y = [-45, 45]
		hip_z = [-20, 20]
		free_joint = [-180, 180]
		full_rot = [True, True, True]
		no_rot = [False, False, False]
		ankle_elbow_rot = [False, True, False]
		wrist_right = Joint(2, [], ankle_elbow_rot, True, Rx=-np.flip(Human36Skeleton.elbow_joint), name='wrist r')
		wrist_left  = Joint(5, [], ankle_elbow_rot, True, Rx=-np.flip(Human36Skeleton.elbow_joint), name='wrist l')

		eye_right_inner_left  = Joint(12, [], no_rot, True)
		eye_right_inner_right = Joint(14,[], no_rot, True)
		eye_right = Joint(13, [eye_right_inner_left, eye_right_inner_right], no_rot, True)
		ear_right = Joint(16, [], no_rot, True)
		eye_left_inner_left  = Joint(9, [], no_rot, True)
		eye_left_inner_right = Joint(11,[], no_rot, True)
		eye_left = Joint(10, [eye_left_inner_left, eye_left_inner_right], no_rot, True)
		ear_left = Joint(15, [], no_rot, True)
		mouth_right = Joint(18, [], no_rot, True)
		mouth_left = Joint(17, [], no_rot, True)
		hip_left = Joint(7, [], no_rot, True) # Z -left-right X - bottom-up, Y is around
		hip_right = Joint(6, [], no_rot, True) # Z -left-right X - bottom-up, Y is around
		nose = Joint(8, [mouth_left, mouth_right, ear_left, ear_right, eye_left, eye_right], full_rot, True, Rx=np.array(deg90_joint)+np.array([20,20]), Ry=deg90_joint, Rz=np.array(deg90_joint))

		elbow_right = Joint(1, [wrist_right], full_rot, True, Rx=elbow_joint_x, Ry=elbow_joint_y, Rz=elbow_joint_z) # Z-left-right, X - bottom-up, Y-around
		elbow_left  = Joint(4, [wrist_left ], full_rot, True, Rx=elbow_joint_x, Ry=-np.flip(elbow_joint_y), Rz=-np.flip(elbow_joint_z))
		shoulder_left  = Joint(3, [elbow_left], [True, False, True], True, Rz=-np.flip(shoulder_shoulder_z), Ry=-np.flip(shoulder_y))
		shoulder_right = Joint(0, [elbow_right], [True, False, True], True, Rz=shoulder_shoulder_z, Ry=shoulder_y)

		hip_center = Joint(19, [hip_left, hip_right], ankle_elbow_rot, True, Rx=chest_rot, Ry=chest_rot, Rz=chest_rot)
		chest = Joint(21, [hip_center], full_rot, True, Rx=hip_x, Ry=hip_y, Rz=hip_z)

		self.root = Joint(20, [shoulder_right, shoulder_left, nose, chest], full_rot, False, Rx=free_joint, Ry=free_joint, Rz=free_joint) # full rot for training mlp

		if angles_constraints is not None: setAnglesConstraints(self.root, angles_constraints)
		self.roots = [self.root]
		self.num_nodes = 22
		self.setOffsets(self.roots, self.num_nodes, self.offsets_dict)
		self.skeletons = [self]
		self.subsample_points = np.ones((self.num_nodes), dtype=bool)
		self.subsample_points[[19,20,21]] = False
		self.num_virtual_nodes = 3
		self.root_angles_idxs = list(self.node_angle_indexing[self.root.idx].values())
		self.num_non_virtual_nodes = self.num_nodes - self.num_virtual_nodes
		self.left_hand_index = 5
		self.right_hand_index = 5

	def getEdges(self):
		return [(0,1),(0,20),(1,2),(20,8),(6,7),(20,21),(8,15,0),(8,15,1),(8,15,2),(8,18,0),(8,18,1),(8,18,2),(13,14),(8,10,0),(8,10,1),(8,10,2)]

	def getName(self):
		return 'body_mediapipe_spine'

	def getConnectionsVirtual(self):
		connections = self.getConnections()
		connections.remove((0,3))
		connections.remove((6,7))
		connections.remove((0,6))
		connections.remove((7,3))
		connections.append((19,21))
		connections.append((20,21))
		connections.append((6,19))
		connections.append((19,7))
		connections.append((0,20))
		connections.append((3,20))
		return connections

	def getOrigin(self, poses):
		if poses.shape[-2] == self.num_non_virtual_nodes:
			return (.5*(poses[...,[0],:]+poses[...,[3],:])) # in the middle of shoulder
		elif poses.shape[-2] == self.num_nodes:
			return poses[..., [self.root.idx], :]
		else: assert False

	def extendPose(self, poses):
		if poses.shape[-2] == self.num_non_virtual_nodes:
			poses_ = torch.cat((poses, .5*(poses[...,[6],:]+poses[...,[7],:]), .5*(poses[...,[0],:]+poses[...,[3],:])), -2)
			return torch.cat((poses_, .5*(poses_[...,[19],:]+poses_[...,[20],:])), -2)
		return poses

	def extendProbs(self, weights):
		# weights should be ... x NUM_JOINTS x 1
		assert weights.shape[-1] == 1
		weights_ = torch.cat((weights, weights[...,[6],:]*weights[...,[7],:], weights[...,[0],:]*weights[...,[3],:]), -2)
		return torch.cat((weights_, weights_[...,[19],:]*weights_[...,[20],:]), -2)

	def getJointWeight(self):
		weights = torch.ones(self.num_nodes, dtype=self.dtype)
		weights[[0,3]] *= 5 # weight shoulder
		weights[[2,5]] *= 5 # weight wrists
		weights[19] *= 3 # weight center of a hip
		head_pts = [i for i in range(8, 19, 1)]
		weights[head_pts] /= len(head_pts) # decrease weight of head points
		return weights

	def centralizePose(self, poses):
		return poses - self.getOrigin(poses)


class BodySkeletonMediapipe(Skeleton):
	def __init__(self, device='cpu', dtype=torch.float32):
		super().__init__(device, dtype)
		self.num_nodes = 33
		self.num_virtual_nodes = 0
		self.num_body_pts = self.num_nodes

	def getConnections(self):
		return [(0,1),(1,2),(2,3),(3,7), (0,4),(4,5),(5,6),(6,8),(9,10),
				(12,14),(14,16),(16,22),(16,20),(16,18),(18,20),
				(11,13),(13,15),(15,21),(15,19),(15,17),(17,19),
				(11,12),(12,24),(11,23),(23,24),(23,25),(24,26),
				(26,28),(25,27),(28,32),(30,32),(28,30),
				(27,29),(27,31),(29,31)]
	def getName(self):
		return 'body_mediapipe'
	def centralizePose(self, poses):
		# try origin point
		return poses - (poses[..., [23], :] + poses[..., [24], :]) *.5
	def getBody(self, pose):
		return pose
