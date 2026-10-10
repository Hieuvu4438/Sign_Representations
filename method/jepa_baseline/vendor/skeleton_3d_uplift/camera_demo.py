"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import torchvision
import numpy as np
import torch
import cv2
import mediapipe
import time

from matplotlib import pyplot as plt
from net_utils import getImageTransform
from uplift_release import getHandImg, extendHandBBox, getImgHandModelOut
from uplift_release import processMediapipeFrameOut, clampHandWristAngles, get3DPosesForward
from drawer import getDimBox, plotPoses, plotMesh, plotOverImage, concatenateImages
from forward_kinematics import ForwardK
from maths_utils import projectBatchK
from process_mediapipe import extractHolisticPoseMP
from media_pipe.skeleton_mediapipe import BodySkeletonMediapipe
from skeleton import HumanSkeleton, HandRightSkeletonMesh
from utils import matrix_to_euler_angles
from mlp_config import getModelFromCKPT


@torch.no_grad()
def main(plot=True, plot_mesh=False, plot_2d=True):
	mlp_body = getModelFromCKPT('models/mlp_body.ckpt', eval=True, get_cfg=False)
	mlp_hand = getModelFromCKPT('models/mlp_hand.ckpt', eval=True, get_cfg=False)
	image_model = getModelFromCKPT('models/img_hand.ckpt', eval=True, get_cfg=False, to_cuda=True)

	transform = getImageTransform()
	to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

	camera = cv2.VideoCapture(0)
	assert camera.isOpened()

	img_cnt = 0
	mp_holistic = mediapipe.solutions.holistic.Holistic(model_complexity=2, refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
	mediapipe_body = BodySkeletonMediapipe()
	skeleton = HumanSkeleton('cpu', 'mediapipe_spine')
	IMG_DIAG = (camera.get(cv2.CAP_PROP_FRAME_WIDTH)**2 + camera.get(cv2.CAP_PROP_FRAME_HEIGHT)**2)**0.5
	resolution = torch.tensor([camera.get(cv2.CAP_PROP_FRAME_WIDTH), camera.get(cv2.CAP_PROP_FRAME_HEIGHT)])
	PX_THR_SAME_HANDS = 1.5 * IMG_DIAG / 1414 # 1.5 for 1414 px image diagonal (ie, 1000 x 1000 img)
	px_dif_min = px_dif_max = 18
	def_shift = 0.25
	bb = (256,256)
	azim=-92; elev=-84
	focal_d = 2.75e3
	pp_d = 500
	K_default = torch.tensor([[focal_d, 0.00000000e+00, pp_d],
							  [0.00000000e+00, focal_d, pp_d-150],
							  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
	skeleton_mesh = HandRightSkeletonMesh(circle_splits=12, circle_sc=0.08)
	import matplotlib; matplotlib.use('Agg') # important, otherwise program is getting killed!!!
	MAX_FRAMES_TO_ESTIMATE = 50
	dim_box = None
	poses_collect = []
	while True:
		st_time = time.time()
		ret, frame = camera.read()
		if not ret:
			print("failed to grab frame")
			break

		pose_2d_mp, body_3d_mp, success = extractHolisticPoseMP(mp_holistic, frame)
		if not success[0]:
			print('body estimation failed')
			continue
		pose_2d, success = processMediapipeFrameOut(pose_2d_mp, success, skeleton, mediapipe_body, PX_THR_SAME_HANDS)
		print(img_cnt, 'hand success, right', success[1], 'left', success[2])
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # to rgb
		body3d_out = mlp_body(skeleton.skeleton_body.centralizePose(pose_2d[:19][None]))

		if not success[1] and image_model is not None:
			img_r = getHandImg(frame, extendHandBBox(pose_2d_mp[[16,18,20,22],:2], resolution, px_dif_min, px_dif_max), resolution, def_shift, bb)
			hand_3d_r_angles, hand_3d_r_offsets, _ = getImgHandModelOut(image_model, [transform(to_tensor(img_r))])
		else:
			hand_3d_r_out = mlp_hand((pose_2d[19:40]-pose_2d[[19]])[None])
			hand_3d_r_angles, hand_3d_r_offsets = hand_3d_r_out['angles'], hand_3d_r_out['offsets']
			if 'cam_rot' in hand_3d_r_out:
				hand_3d_r_angles[..., :3] = matrix_to_euler_angles(hand_3d_r_out['cam_rot'], skeleton.convention)

		if not success[2] and image_model is not None:
			img_l = getHandImg(frame, extendHandBBox(pose_2d_mp[[15,17,19,21],:2], resolution, px_dif_min, px_dif_max), resolution, def_shift, bb)
			img_l = cv2.flip(img_l, 1)
			hand_3d_l_angles, hand_3d_l_offsets, _ = getImgHandModelOut(image_model, [transform(to_tensor(img_l))])
		else:
			left_hand_2d = (pose_2d[40:]-pose_2d[[40]])[None].clone()
			left_hand_2d[...,0] *= -1
			hand_3d_l_out = mlp_hand(left_hand_2d)
			hand_3d_l_angles, hand_3d_l_offsets = hand_3d_l_out['angles'], hand_3d_l_out['offsets']
			if 'cam_rot' in hand_3d_l_out:
				hand_3d_l_angles[..., :3] = matrix_to_euler_angles(hand_3d_l_out['cam_rot'], skeleton.convention)
		hand_offsets = (hand_3d_r_offsets + hand_3d_l_offsets) * 0.5
		angles_global_right_clamped, angles_global_left_clamped = clampHandWristAngles(skeleton, hand_3d_r_angles, hand_3d_l_angles, body3d_out['angles'], get_global=True)

		if plot:
			if plot_2d:
				img_plot = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
				plotOverImage(img_plot, [pose_2d], skeleton.getConnections(), None, None, skeleton.subSamplePose(torch.tensor(skeleton.colors())).numpy(), skeleton.colorsEdges(), 5, 3, None, True)
			else:
				img_plot = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			if plot_mesh:
				hand_rl_mesh = ForwardK.forwardPoseSimple(skeleton_mesh, torch.cat((angles_global_right_clamped, angles_global_left_clamped)), skeleton_mesh.copyOffsetsFromSkeleton(hand_offsets)[0])
				hand_rl_mesh[0] += body3d_out['pose_3d'][0,[2]]
				hand_rl_mesh[1,:,0] *= -1
				hand_rl_mesh[1] += body3d_out['pose_3d'][0,[5]]

				if img_cnt == MAX_FRAMES_TO_ESTIMATE:
					dim_box = getDimBox(torch.cat(poses_collect))*1.15
					poses_collect = None
				elif img_cnt < MAX_FRAMES_TO_ESTIMATE:
					poses_collect.append(torch.cat((hand_rl_mesh.reshape(1,-1,3), body3d_out['pose_3d']),-2))
					dim_box = getDimBox(poses_collect[-1])*1.15

				fig = plt.figure(figsize=(11.0, 10.0))
				ax = plotMesh(hand_rl_mesh, skeleton_mesh.faces, ax=fig.add_subplot(projection='3d'), azim=azim, elev=elev, is_blank=True)
				plotPoses(body3d_out['pose_3d'], skeleton.skeleton_body.getConnections(), ax=ax, azim=azim, elev=elev, is_blank=True, show_axes=False, colors_l=skeleton.skeleton_body.colorsEdges(), colors_p=skeleton.skeleton_body.subSamplePose(torch.tensor(skeleton.skeleton_body.colors())).numpy(), dim_box=dim_box)
				fig.canvas.draw()
				img_3d = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))

				final_img = concatenateImages(img_plot, img_3d)
			else:
				img_plot_3d = np.ones((1000, 1000, 3), dtype=np.uint8)*255
				poses_3d_clamped = get3DPosesForward(skeleton, mlp_hand.skeleton, mlp_body.skeleton, body3d_out['angles'], angles_global_right_clamped, angles_global_left_clamped, body3d_out['offsets'], hand_offsets)
				poses_3d_clamped_proj = projectBatchK(K_default, poses_3d_clamped+torch.tensor([[0,0,3]]))
				plotOverImage(img_plot_3d, poses_3d_clamped_proj, skeleton.getConnections(), None, None, skeleton.subSamplePose(torch.tensor(skeleton.colors())).numpy(), skeleton.colorsEdges(), 5, 3, None, True)

				final_img = concatenateImages(img_plot, img_plot_3d)

			cv2.imshow("demo", final_img)

		img_cnt += 1

		k = cv2.waitKey(1)
		if k == 27:
			print('ESC detected! Break')
			break

		print('time (%.3f s)' % (time.time()-st_time))
	camera.release()
	cv2.destroyAllWindows()

import sys, traceback, pdb
if __name__ == '__main__':
	try:
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.enabled = True
		seed = 0; torch.manual_seed(seed)
		main()
	except:
		extype, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
