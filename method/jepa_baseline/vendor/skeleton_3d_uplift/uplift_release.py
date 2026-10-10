"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import argparse, torchvision, numpy as np, torch, cv2, mediapipe
import os

from matplotlib import pyplot as plt
from drawer import plotPoses2dRGB, plotPoses
from forward_kinematics import ForwardK
from mlp_config import getModelFromCKPT
from process_mediapipe import extractHolisticPoseMP
from net_utils import getImageTransform
from media_pipe.skeleton_mediapipe import BodySkeletonMediapipe
from skeleton import HumanSkeleton
from utils import matrix_to_euler_angles, euler_angles_to_matrix
from video import VideoReader, getImageWidthVideo

def extendHandBBox(pts, img_shape, px_dif_min=60, px_dif_max=60):
	"""
	img_shape: [width, height]
	"""
	min_x, min_y = pts.min(0)[0]
	max_x, max_y = pts.max(0)[0]
	bbox = torch.tensor([[max(0,min_x-px_dif_min), max(0, min_y-px_dif_min)], [min(img_shape[0], max_x+px_dif_max), min(img_shape[1], max_y+px_dif_max)]])
	return bbox

def filterNdimTensor(x, mask, w):
	"""
	x : B x N
	"""
	x_new = x.clone()
	for k in range(w//2, len(x)-(w//2+1), 1):
		if mask is None:
			x_new[k] = x[k-w//2:k+w//2+1].median(0).values
		elif mask[k-w//2:k+w//2+1].sum() == w:
			x_new[k] = x[k-w//2:k+w//2+1][mask[k-w//2:k+w//2+1]].median(0).values
	return x_new

def getRoi2DImg(points, img_wh):
	"""
	points: B x N x 2
	output: B x [x1, y1, x2, y2]
	"""
	return torch.cat((torch.clamp_min(points.min(-2)[0], 0.0), torch.clamp_max(points.max(-2)[0], img_wh)),-1)

def getBBoxShifting(img, roi, bb, rot=None):
	"""
	img : h x w x 3 numpy
	roi : (x_min, y_min, x_max, y_max) -- region of interest
	bb : (width, height) dimension of output bounding box
	rot : angle (in radians) rotation of the cropped part
	output:
	cropped image, affine transform
	"""
	scale = min(bb[0] / (roi[2] - roi[0]), bb[1] / (roi[3] - roi[1])) # one scale = same aspect ratio. scale must be `min`, otherwise roi will be out of the bb
	bb_diff = (torch.tensor(bb) - (roi[2:] - roi[:2]) * scale) * 0.5
	if rot is not None:
		bb = np.array(bb)
		A1 = np.identity(3)
		A2 = A1.copy(); A3 = A1.copy(); A4 = A1.copy()
		A1[[0,1],2] = -roi[:2] + bb_diff/scale
		A1[:2] *= scale.item()
		# to rotate bb, first shift by half, rotate, and shift back
		A2[[0,1],2] = -bb * 0.5
		A3[0,0] = A3[1,1] = np.cos(rot)
		A3[0,1] = -np.sin(rot); A3[1,0] = -A3[0,1].copy()
		A4[[0,1],2] = bb * 0.5
		aff =(A4 @ A3 @ A2 @ A1)[:2]
	else:
		aff = np.array([[scale, 0, -scale*roi[0]+bb_diff[0]], [0, scale, -scale*roi[1]+bb_diff[1]]])
	return cv2.warpAffine(img, aff, bb), aff, rot

def getMinMaxShifts(roi, img_shape):
	"""
	img_shape: width x height
	roi : x_min, y_min, x_max, y_max
	"""
	shifts = roi.clone()
	shifts[...,[2,3]] = torch.max(torch.tensor(0.0), img_shape - roi[...,[2,3]]) # avoid roi to be bigger than image dim
	return torch.cat((-torch.max(torch.tensor(0.0), shifts[...,[0,1]]), torch.zeros(2))), torch.cat((torch.zeros(2), shifts[...,[2,3]]))

def getMinMaxShiftsImgAugm(roi, img_shape, min_shift_scale, max_shift_scale):
	"""
	img_shape: width x height
	roi : x_min, y_min, x_max, y_max
	"""
	min_shifts_, max_shifts_ = getMinMaxShifts(roi, img_shape)
	roi_len = roi[2:] - roi[:2]
	min_shifts, max_shifts = min_shifts_.clone(), max_shifts_.clone()
	min_shifts[:2] = torch.clamp_min(min_shifts_[:2], -max_shift_scale*roi_len) # -max = min
	max_shifts[:2] = torch.clamp_min(min_shifts_[:2], -min_shift_scale*roi_len) # -min = max
	min_shifts[2:] = torch.clamp_max(max_shifts_[2:], min_shift_scale*roi_len)
	max_shifts[2:] = torch.clamp_max(max_shifts_[2:], max_shift_scale*roi_len)
	return min_shifts, max_shifts

def getHandImg(img, pts, resolution, def_shift, bb):
	roi = getRoi2DImg(pts, resolution)
	mins, maxs = getMinMaxShiftsImgAugm(roi, resolution, def_shift, def_shift)
	return getBBoxShifting(img,  roi + (mins + maxs) * .5, bb, rot=None)[0]

@torch.no_grad()
def getImgHandModelOut(image_model, imgs):
	if len(imgs) == 0: return []
	out = image_model(torch.stack(imgs).to('cuda' if torch.cuda.is_available() else 'cpu'))
	angles = out['angles']
	if 'cam_rot' in out:
		angles[..., :3] = matrix_to_euler_angles(out['cam_rot'], image_model.keypoint_model.skeleton.convention)
	return angles.detach().cpu(), out['offsets'].detach().cpu(), out['pose_3d'].detach().cpu()


def visualizeResults(file, azim=-92, elev=-84):
	assert os.path.exists(file), f'file {file} does not exist'
	data = torch.load(file)
	video = VideoReader(data['video_name'], convert_rgb=True)
	skeleton = HumanSkeleton('cpu', 'mediapipe_spine')
	for i, key in enumerate(data['frames_list']):
		print('show frame', i)
		img = video.getFrame(key)
		figs, axs = plt.subplots(1, 2, figsize=(18, 12), subplot_kw=dict(projection='3d'))
		axs[0] = figs.add_subplot(1, 2, 1)
		plotPoses2dRGB([torch.cat((data['body2d'][i], data['right2d'][i], data['left2d'][i]))], skeleton.getConnections(), img, ax=axs[0], colors_edges=skeleton.colorsEdges(), colors_pts=skeleton.subSamplePose(torch.tensor(skeleton.colors())).numpy())
		plotPoses([data['poses_3d'][i]], skeleton.getConnections(), ax=axs[1], azim=azim, elev=elev, is_blank=True, show_axes=True, colors_l=skeleton.colorsEdges(), colors_p=skeleton.subSamplePose(torch.tensor(skeleton.colors())).numpy())
		plt.show()

def processMediapipeFrameOut(pose_2d_mp, success, skeleton, mediapipe_body, PX_THR_SAME_HANDS):
	pose_2d_mp = pose_2d_mp[...,:2]
	body_2d = pose_2d_mp[:mediapipe_body.num_nodes][skeleton.skeleton_body.getBodyOrder()]
	hand_2d = pose_2d_mp[mediapipe_body.num_nodes:]
	pose_2d = torch.cat((body_2d, hand_2d))

	if success[1] and success[2]:
		are_hands_same = torch.linalg.norm(pose_2d[19:40] - pose_2d[40:], dim=-1).mean() < PX_THR_SAME_HANDS
		if are_hands_same:
			wrist_dists_r = torch.linalg.norm(body_2d[2] - hand_2d[0], dim=-1)
			wrist_dists_l = torch.linalg.norm(body_2d[5] - hand_2d[21], dim=-1)
			# if hands are the same, decide to which hand it belongs by comparing distance to wrist
			if wrist_dists_r < wrist_dists_l:
				success[2] = False # close to right wrist, then left hand is wrong
			else:
				success[1] = False
	return pose_2d, success


def clampHandWristAngles(skeleton, right_angles, left_angles, body_angles, get_global=False):
	left_angles_flipped = left_angles.clone(); left_angles_flipped[...,skeleton.skeleton_hand.flip_hand_angles_idxs] *= -1.0
	R_wrist_left = euler_angles_to_matrix(left_angles_flipped[...,:3], skeleton.convention)

	R_wrist_right = euler_angles_to_matrix(right_angles[...,:3], skeleton.convention)
	R_body = ForwardK.getRot(skeleton.skeleton_body, body_angles, True)
	R_wrist_rel_right = R_body[:,2].transpose(-1,-2) @ R_wrist_right
	R_wrist_rel_left = R_body[:,5].transpose(-1,-2) @ R_wrist_left
	angles_wrist_rel_right = matrix_to_euler_angles(R_wrist_rel_right, skeleton.convention)
	angles_wrist_rel_left = matrix_to_euler_angles(R_wrist_rel_left, skeleton.convention)
	wrist_body_constraint_z = np.array([-80, 80])
	wrist_body_constraint_y = np.array([-180, 180]) # leave for now like this
	wrist_body_constraint_x = np.array([-50, 50])
	wrist_constraint_right = torch.FloatTensor(np.stack((wrist_body_constraint_z, wrist_body_constraint_x, wrist_body_constraint_y))) * np.pi/180.
	wrist_constraint_left = wrist_constraint_right.clone()
	wrist_constraint_left[2] = -wrist_constraint_left[2][[1,0]]
	angles_wrist_rel_right_clamped = angles_wrist_rel_right.clone()
	angles_wrist_rel_left_clamped = angles_wrist_rel_left.clone()
	for i in range(3):
		angles_wrist_rel_right_clamped[...,i] = torch.clamp(angles_wrist_rel_right_clamped[...,i], wrist_constraint_right[i,0], wrist_constraint_right[i,1])
		angles_wrist_rel_left_clamped[...,i] = torch.clamp(angles_wrist_rel_left_clamped[...,i], wrist_constraint_left[i,0], wrist_constraint_left[i,1])

	if get_global:
		angles_global_right_clamped = right_angles.clone()
		angles_global_left_clamped = left_angles_flipped.clone()

		angles_global_left_clamped[...,:3] = matrix_to_euler_angles(R_body[:,5] @ euler_angles_to_matrix(angles_wrist_rel_left_clamped[...,:3], skeleton.convention), skeleton.convention)
		angles_global_right_clamped[...,:3] = matrix_to_euler_angles(R_body[:,2] @ euler_angles_to_matrix(angles_wrist_rel_right_clamped[...,:3], skeleton.convention), skeleton.convention)
		angles_global_left_clamped[...,skeleton.skeleton_hand.flip_hand_angles_idxs] *= -1.0
		return angles_global_right_clamped, angles_global_left_clamped
	else:
		angles_rel_right_clamped = right_angles.clone()
		angles_rel_left_clamped = left_angles_flipped.clone()
		angles_rel_right_clamped[...,:3] = angles_wrist_rel_right_clamped
		angles_rel_left_clamped[...,:3] = angles_wrist_rel_left_clamped
		return angles_rel_right_clamped, angles_rel_left_clamped

def get3DPosesForward(skeleton, skeleton_hand, skeleton_body, angles_body, angles_right, angles_left, body_offsets, offsets_hand):
	body3d = skeleton.skeleton_body.subSamplePose(ForwardK.forwardPoseSimple(skeleton_body, angles_body, body_offsets))
	right3d = ForwardK.forwardPoseSimple(skeleton_hand, angles_right, offsets_hand)
	left3d  = ForwardK.forwardPoseSimple(skeleton_hand, angles_left, offsets_hand)
	left3d[...,0] *= -1
	return skeleton.concatePoses(torch.cat((body3d, right3d, left3d), -2))

def fill_hand_one_gap(successes, angles):
	fails = torch.where(~successes)[0]
	if len(fails) == 0: return
	fails = fails[torch.logical_and(fails > 0, fails < len(successes)-1)]
	if len(fails) == 0: return
	neighbors_success = torch.logical_and(successes[fails-1], successes[fails+1])
	if neighbors_success.any():
		print('# filled gaps', len(neighbors_success.nonzero()))
		failed_idxs_neighbor_success = fails[neighbors_success]
		# average only wrist rotation
		angles[failed_idxs_neighbor_success,:3] = (angles[failed_idxs_neighbor_success-1,:3] + angles[failed_idxs_neighbor_success+1,:3]) * .5


def demo2Duplift(skeleton, save_file, video_name, mlp_body, mlp_hand, image_model, bb):
	mp_holistic = mediapipe.solutions.holistic.Holistic(model_complexity=2, refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
	MAX_BATCH_IMG = 64
	px_dif_min = px_dif_max = 18
	def_shift = 0.25

	video = VideoReader(video_name)
	resolution = torch.tensor(getImageWidthVideo(video_name))
	frame_idx = -1
	mediapipe_body = BodySkeletonMediapipe()
	transform = getImageTransform()
	to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

	results = {'body2d': [], 'right2d': [], 'left2d': [], 'success_right': [], 'success_left': []}
	frames_saved = 0
	frames_list = []

	imgs_r, imgs_l = [], []
	IMG_DIAG = (resolution[0]**2 + resolution[1]**2)**0.5
	PX_THR_SAME_HANDS = 1.5 * IMG_DIAG / 1414 # 1.5 for 1414 px image diagonal (ie, 1000 x 1000 img)
	num_frames = len(video)
	img_model_3d_r, img_model_3d_l = [], []
	while True:
		frame_idx += 1
		if video is None:
			frame = cv2.imread(video_name(frame_idx))
		else:
			frame = video.read()
		if frame is None: break
		print(frame_idx, '/', num_frames)

		pose_2d_mp, body_3d_mp, success = extractHolisticPoseMP(mp_holistic, frame)
		if not success[0]:
			print('body estimation failed')
			continue
		pose_2d, success = processMediapipeFrameOut(pose_2d_mp, success, skeleton, mediapipe_body, PX_THR_SAME_HANDS)

		results['success_right'].append(success[1])
		results['success_left'].append(success[2])
		results['right2d'].append(pose_2d[19:40])
		results['left2d'].append(pose_2d[40:])
		results['body2d'].append(pose_2d[:19])

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # to rgb

		if not success[1] and image_model is not None:
			imgs_r.append(getHandImg(frame, extendHandBBox(pose_2d_mp[[16,18,20,22],:2], resolution, px_dif_min, px_dif_max), resolution, def_shift, bb))
			print('right hand failed')
			# plt.imshow(frame); plt.figure(); plt.imshow(imgs_r[-1]); plt.show()
			imgs_r[-1] = transform(to_tensor(imgs_r[-1]))
			if len(imgs_r) > MAX_BATCH_IMG:
				img_model_3d_r.append(getImgHandModelOut(image_model, imgs_r)[0]); imgs_r = []

		if not success[2] and image_model is not None:
			imgs_l.append(getHandImg(frame, extendHandBBox(pose_2d_mp[[15,17,19,21],:2], resolution, px_dif_min, px_dif_max), resolution, def_shift, bb))
			imgs_l[-1] = cv2.flip(imgs_l[-1], 1)
			# plt.imshow(frame); plt.figure(); plt.imshow(imgs_l[-1]); plt.show()
			imgs_l[-1] = transform(to_tensor(imgs_l[-1]))
			print('left hand failed')
			if len(imgs_l) > MAX_BATCH_IMG:
				img_model_3d_l.append(getImgHandModelOut(image_model, imgs_l)[0]); imgs_l = []

		frames_list.append(frame_idx)
		frames_saved += 1

	for key in results.keys(): results[key] = torch.tensor(results[key], dtype=bool) if 'success' in key else torch.stack(results[key])

	body2d_c = skeleton.skeleton_body.centralizePose(results['body2d'])
	right2d_c = results['right2d'] - results['right2d'][:,[0]]
	left2d_c = results['left2d'] - results['left2d'][:,[0]]
	left2d_c[...,0] *= -1
	results['frames_list'] = np.array(frames_list)
	results['video_name'] = video_name
	with torch.no_grad():
		body3d_out = mlp_body(body2d_c)
		body_offsets = body3d_out['offsets'].median(dim=0).values
		right_angles = torch.zeros(len(results['right2d']), skeleton.skeleton_hand.num_hand_angles)
		left_angles = right_angles.clone()

		offsets_hand = []
		if results['success_right'].any():
			right3d_out = mlp_hand(right2d_c[results['success_right']])
			if 'cam_rot' in right3d_out:
				right3d_out['angles'][...,:3] = matrix_to_euler_angles(right3d_out['cam_rot'], skeleton.convention)
			right_angles[results['success_right']] = right3d_out['angles']
			offsets_hand.append(right3d_out['offsets'])

		if results['success_left'].any():
			left3d_out = mlp_hand(left2d_c[results['success_left']])
			if 'cam_rot' in left3d_out:
				left3d_out['angles'][...,:3] = matrix_to_euler_angles(left3d_out['cam_rot'], skeleton.convention)
			left_angles[results['success_left']] = left3d_out['angles']
			offsets_hand.append(left3d_out['offsets'])

		###
		if len(imgs_r) > 0:
			img_model_3d_r.append(getImgHandModelOut(image_model, imgs_r)[0])
		if len(imgs_l) > 0:
			img_model_3d_l.append(getImgHandModelOut(image_model, imgs_l)[0])
		if len(img_model_3d_r) > 0:
			img_model_3d_r = torch.cat(img_model_3d_r)
			right_angles[~results['success_right']] = img_model_3d_r
		if len(img_model_3d_l) > 0:
			img_model_3d_l = torch.cat(img_model_3d_l)
			left_angles[~results['success_left']] = img_model_3d_l
		###

		fill_hand_one_gap(results['success_right'], right_angles)
		fill_hand_one_gap(results['success_left'], left_angles)
		####################

		if len(offsets_hand) > 0:
			offsets_hand = torch.cat(offsets_hand).median(dim=0).values
		else:
			raise NotImplementedError

		angles_rel_right_clamped, angles_rel_left_clamped = clampHandWristAngles(skeleton, right_angles, left_angles, body3d_out['angles'])
		win_sz = 5
		body3d_angles_f = filterNdimTensor(body3d_out['angles'], None, win_sz)
		right_angles_f = filterNdimTensor(right_angles, None, win_sz)
		left_angles_f = filterNdimTensor(left_angles, None, win_sz)

	skeleton_joint = HumanSkeleton('cpu', 'mediapipe_spine', is_joint=True)
	results['poses_3d_clamped'] = skeleton_joint.subSamplePose(ForwardK.forwardPoseSimple(skeleton_joint, torch.cat((body3d_out['angles'], angles_rel_right_clamped, angles_rel_left_clamped), -1), skeleton_joint.mergeOffsets(body_offsets, offsets_hand)))

	results['poses_3d_filtered'] = get3DPosesForward(skeleton, mlp_hand.skeleton, mlp_body.skeleton, body3d_angles_f, right_angles_f, left_angles_f, body3d_out['offsets'], offsets_hand)
	results['poses_3d'] = get3DPosesForward(skeleton, mlp_hand.skeleton, mlp_body.skeleton, body3d_out['angles'], right_angles, left_angles, body3d_out['offsets'], offsets_hand)

	results['body_angles'] = body3d_out['angles']
	results['right_angles'] = right_angles
	results['left_angles'] = left_angles
	results['body_offsets'] = body_offsets
	results['hand_offsets'] = offsets_hand
	torch.save(results, save_file)

def main(video, save_file):
	mlp_body = getModelFromCKPT('models/mlp_body.ckpt', eval=True, get_cfg=False)
	mlp_hand = getModelFromCKPT('models/mlp_hand.ckpt', eval=True, get_cfg=False)
	img_hand = getModelFromCKPT('models/img_hand.ckpt', eval=True, get_cfg=False, to_cuda=True)
	demo2Duplift(HumanSkeleton('cpu', 'mediapipe_spine'), save_file, video, mlp_body, mlp_hand, img_hand, (256, 256))

import sys, traceback, pdb
if __name__ == '__main__':
	try:
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.enabled = True
		seed = 0; torch.manual_seed(seed)
		parser = argparse.ArgumentParser()
		parser.add_argument('--video_name', type=str, required=True)
		parser.add_argument('--save_file', type=str, required=True)
		parser.add_argument('--show', required=False, default=False, action='store_true')
		params, _ = parser.parse_known_args()
		if params.show:
			visualizeResults(params.save_file)
		else:
			main(params.video_name, params.save_file)
	except:
		extype, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
