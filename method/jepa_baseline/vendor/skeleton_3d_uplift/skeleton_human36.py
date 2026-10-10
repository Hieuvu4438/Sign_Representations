"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import torch
import numpy as np
from skeleton import Skeleton, Joint

class Human36Skeleton(Skeleton):
	free_joint = [-180, 180]
	spine_around = [-80, 80]
	spine_left_right = [-60, 60]
	spine_bottom_up = [-155, 45]
	elbow_joint = [0, 160]
	ankle_joint = [-160, 0]
	elbow_bottom_up = [-80, 180]
	elbow_left_right = [-180, 45]
	elbow_around = [-45, 90]
	ankle_joint_bottom_up = [-70, 110]
	ankle_joint_left_right = [-80, 45]
	ankle_joint_around = [-90, 60]
	shoulder_bottom_up = [-60, 50]
	shoulder_left_right = [-40, 50]
	headtop_around = [-90, 90]
	headtop_left_right = [-80, 80]
	headtop_bottom_up = [-90, 75]
	def __init__(self, device, dtype=torch.float32, offsets_dict=None):
		super().__init__(device, dtype)
		self.JOINTS = {
			"HipCenter": 0,
			"RHip": 1,
			"RKnee": 2,
			"RFoot": 3,
			"LHip": 4,
			"LKnee": 5,
			"LFoot": 6,
			"Spine": 7,
			"Neck": 8,
			"HeadCenter": 9,
			"HeadTop": 10,
			"LShoulder": 11,
			"LElbow": 12,
			"LWrist": 13,
			"RShoulder": 14,
			"RElbow": 15,
			"RWrist": 16,
		}
		self.BONES = [
			["HipCenter", "RHip"],
			["RHip", "RKnee"],
			["RKnee", "RFoot"],
			["HipCenter", "LHip"],
			["LHip", "LKnee"],
			["LKnee", "LFoot"],
			["HipCenter", "Spine"],
			["Spine", "Neck"],
			["Neck", "HeadCenter"],
			["HeadCenter", "HeadTop"],
			["Neck", "LShoulder"],
			["LShoulder", "LElbow"],
			["LElbow", "LWrist"],
			["Neck", "RShoulder"],
			["RShoulder", "RElbow"],
			["RElbow", "RWrist"],
		]
		self.EDGES = [[self.JOINTS[s], self.JOINTS[e]] for s, e in self.BONES]
		if offsets_dict is None:
			self.offsets_dict = {'Shoulder_x' : [0.1510, [(self.JOINTS['RShoulder'], -1), (self.JOINTS['LShoulder'], 1)]],
							 'Elbow_z' : [0.2789, [(self.JOINTS['RElbow'], -1), (self.JOINTS['LElbow'], -1)]],
							 'Wrist_z' : [0.2517, [(self.JOINTS['RWrist'], -1), (self.JOINTS['LWrist'], -1)]],
							 'Spine_z': [0.2335, [(self.JOINTS['Spine'], 1)]],
							 'Hip_x': [0.1329, [(self.JOINTS['RHip'], -1), (self.JOINTS['LHip'], 1)]],
							 'Knee_z': [0.4429, [(self.JOINTS['RKnee'], -1), (self.JOINTS['LKnee'], -1)]],
							 'Foot_z': [0.4542, [(self.JOINTS['RFoot'], -1), (self.JOINTS['LFoot'], -1)]],
							 'Neck_z': [0.2571, [(self.JOINTS['Neck'], 1)]],
							 'HeadCenter_z': [0.1211, [(self.JOINTS['HeadCenter'], 1)]],
							 'HeadTop_y': [0.1150, [(self.JOINTS['HeadTop'], 1)]]
							 }
		else: self.offsets_dict = offsets_dict

		full_rot = [True, True, True]
		no_rot = [False, False, False]
		ankle_elbow_rot = [False, True, False] # onle one rotation for wrist
		shoulder_rot = [True, False, True]
		# shoulder_joint_x = shoulder_joint_y = shoulder_joint_z = deg90_joint = elbow_joint = shoulder_shoulder_z = hip_hip_z = hip_z = hip_y = hip_x = free_joint

		wrist_r = Joint(16, [], ankle_elbow_rot, True, Rx=-np.flip(Human36Skeleton.elbow_joint), name='wrist r')
		wrist_l = Joint(13, [], ankle_elbow_rot, True, Rx=-np.flip(Human36Skeleton.elbow_joint), name='wrist l')

		elbow_r = Joint(15, [wrist_r], full_rot, True, Rx=-np.flip(Human36Skeleton.elbow_bottom_up), Ry=-np.flip(Human36Skeleton.elbow_left_right), Rz=Human36Skeleton.elbow_around, name='elbow r')
		elbow_l = Joint(12, [wrist_l], full_rot, True, Rx=-np.flip(Human36Skeleton.elbow_bottom_up), Ry=Human36Skeleton.elbow_left_right, Rz=-np.flip(Human36Skeleton.elbow_around), name='elbow l')

		shoulder_r = Joint(14, [elbow_r], shoulder_rot, True, Ry=-np.flip(Human36Skeleton.shoulder_bottom_up), Rz=Human36Skeleton.shoulder_left_right, name='shoulder r')
		shoulder_l = Joint(11, [elbow_l], shoulder_rot, True, Ry=Human36Skeleton.shoulder_bottom_up, Rz=-np.flip(Human36Skeleton.shoulder_left_right), name='shoulder l')

		foot_r = Joint(3, [], ankle_elbow_rot, True, Rx=-np.flip(Human36Skeleton.ankle_joint), name='foot r')
		foot_l = Joint(6, [], ankle_elbow_rot, True, Rx=-np.flip(Human36Skeleton.ankle_joint), name='foot l')

		knee_r = Joint(2, [foot_r], full_rot, True, Rx=-np.flip(Human36Skeleton.ankle_joint_bottom_up), Ry=-np.flip(Human36Skeleton.ankle_joint_left_right), Rz=Human36Skeleton.ankle_joint_around, name='knee r')
		knee_l = Joint(5, [foot_l], full_rot, True, Rx=-np.flip(Human36Skeleton.ankle_joint_bottom_up), Ry=Human36Skeleton.ankle_joint_left_right, Rz=-np.flip(Human36Skeleton.ankle_joint_around), name='knee l')

		hip_r = Joint(1, [knee_r], no_rot, True, name='hip r')
		hip_l = Joint(4, [knee_l], no_rot, True, name='hip l')

		head_top = Joint(10, [], [False, True, False], True, Rx=[0, 15], name='head top')
		head_center = Joint(9, [head_top], full_rot, True, Rz=Human36Skeleton.headtop_around,
					Rx=-np.flip(Human36Skeleton.headtop_bottom_up), Ry=Human36Skeleton.headtop_left_right, name='head center')
		# neck = Joint(8, [shoulder_r, shoulder_l, head_center], full_rot, True, Rz=[-10, 10], Rx=[-40,0], Ry=[-15,15])# probably redundant joint angles
		neck = Joint(8, [shoulder_r, shoulder_l, head_center], [False, True, False], True, Rx=[0, 40], name='neck')
		spine = Joint(7, [neck], full_rot, True, Rz=Human36Skeleton.spine_around, Rx=-np.flip(Human36Skeleton.spine_bottom_up), Ry=Human36Skeleton.spine_left_right, name='spine')

		self.root = Joint(0, [hip_r, hip_l, spine], full_rot, Rx=Human36Skeleton.free_joint, Ry=Human36Skeleton.free_joint, Rz=Human36Skeleton.free_joint, name='hip center')

		self.roots = [self.root]
		self.num_nodes = len(self.JOINTS)
		self.setOffsets(self.roots, self.num_nodes, self.offsets_dict)
		self.skeletons = [self]
		self.num_virtual_nodes = 0
		self.skeleton_body = self
	def getConnections(self):
		return self.EDGES
	def getName(self):
		return 'human3.6'
	def centralizePose(self, poses):
		assert poses.shape[-2] == self.num_nodes
		return poses - poses[..., [self.root.idx], :]
	def transformPose(self, poses):
		raise NotImplementedError
	def getJointWeight(self):
		weights = torch.ones(self.num_nodes, dtype=self.dtype)
		weights[[self.name2nodes['hip l'],self.name2nodes['hip r']]] *= 5 # weight hip
		weights[[self.name2nodes['wrist l'],self.name2nodes['wrist r']]] *= 4 # weight wrists
		return weights
	def getAngularWeights(self):
		weights = torch.ones(len(self.angles_constraints), dtype=self.dtype)
		weights[:3] *= 10
		return weights