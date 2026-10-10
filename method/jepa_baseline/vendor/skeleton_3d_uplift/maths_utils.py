"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import numpy as np
import torch

from utils import euler_angles_to_matrix

def symmetric_orthogonalization(x):
	"""Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

	x: should have size [batch_size, 9]

	Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
	"""
	m = x.view(-1, 3, 3)
	u, s, v = torch.svd(m)
	vt = torch.transpose(v, 1, 2)
	uvt = torch.matmul(u, vt)
	det = torch.linalg.det(uvt)
	det = det.view(-1, 1, 1)
	vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
	r = torch.matmul(u, vt)
	return r

def project3DPoints(Ps_, points_3D_):
	NUM_VIEWS = len(Ps_)
	points_3D = points_3D_ if points_3D_.shape[0] == 3 else points_3D_.T
	Ps = np.concatenate((Ps_)).reshape(NUM_VIEWS, 3, 4)
	images_points = Ps[:,:3,:3] @ points_3D + Ps[:,:,-1].reshape(NUM_VIEWS, 3, 1)
	images_points = images_points[:, :2] / images_points[:, 2].reshape(NUM_VIEWS, 1, -1)
	return images_points.transpose(0,2,1)

def projectBatchP(P, poses):
	# P is B x 3 x 4
	# pose is B x PTS x 3
	# out is B x PTS x 2
	poses_2d = torch.matmul(P[..., :3, :3], poses.transpose(-1,-2)) + P[..., :,  [-1]]
	return (poses_2d[...,:2,:] / poses_2d[...,[2],:]).transpose(-1,-2)

def projectBatchK(K, poses):
	# K is ... x 3 x 3
	# pose is ... x PTS x 3
	# out is ... x PTS x 2
	poses_2d = torch.matmul(K[..., :3, :3], poses.transpose(-1,-2))
	return (poses_2d[...,:2,:] / poses_2d[...,[2],:]).transpose(-1,-2)

def projectP(P, poses):
	# P is CAM x 3 x 4
	# pose is N x PTS x 3
	# out is N x CAM x PTS x 2
	if poses.ndim == 3:
		poses_2d = torch.matmul(P[None, :, :3, :3], poses.transpose(-1,-2)[:, None]) + P[None, :, :,  [-1]]
	# elif poses.ndim == 4:
	# 	poses_2d = torch.matmul(P[None, None, :, :3, :3], poses.transpose(-1,-2)[..., None, :, :]) + P[None, None, :, :,  [-1]]
	else: assert False
	return (poses_2d[...,:2,:] / (poses_2d[...,[2],:] + 1e-10)).transpose(-1,-2)

def project(pose, K, rigid_angles, T):
	# K    : 3 x 3
	# rigid_angles : C x 3
	# T : C x 3 x 1
	# pose : B x N x 3
	# return : B x C x N x 2
	# K [R t] * X = K [RX + t]
	poses_2d_est = K @ (euler_angles_to_matrix(rigid_angles, "ZXY") @ pose.transpose(1,2)[:, None] + T)
	return (poses_2d_est[:,:,:2] / poses_2d_est[:,:,[2]]).transpose(2,3)
