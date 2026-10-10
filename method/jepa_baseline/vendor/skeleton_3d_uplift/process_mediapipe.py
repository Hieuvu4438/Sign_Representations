"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import cv2
import torch


def processImg(model, frame):
	frame.flags.writeable = False
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	results = model.process(frame)
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	frame.flags.writeable = True
	return results

def extractHolisticPoseMP(model, frame, THR_WRIST=None, body_order=None):
	img_size = torch.FloatTensor(frame.shape[:2][::-1])
	body_results = processImg(model, frame)
	WRIST_RIGHT_MP, WRIST_LEFT_MP = 16, 15

	if body_results.pose_landmarks is None:
		return None, None, [False, False, False]

	if body_results.pose_world_landmarks is None:
		body_3d = torch.zeros((19, 3))
	else:
		body_3d = torch.tensor([[p.x, p.y, p.z] for p in body_results.pose_world_landmarks.landmark])

	right_hand_detected = left_hand_detected = True
	body_img = torch.tensor([[p.x, p.y, p.visibility] for p in body_results.pose_landmarks.landmark])

	def hand_weight_fnc(dist):
		return max(0.2, THR_WRIST*.5 / dist)

	if body_results.right_hand_landmarks is None:
		right_img = torch.zeros((21,2))
		right_hand_detected = False
		right_hand_weights = torch.zeros(21,1)
	else:
		right_img = torch.tensor([[p.x, p.y] for p in body_results.right_hand_landmarks.landmark])
		right_hand_weights = torch.ones(21,1)
		if THR_WRIST is not None:
			dist = torch.linalg.norm(img_size * right_img[0,:2] - img_size * body_img[WRIST_RIGHT_MP,:2])
			if dist > THR_WRIST:
				# decrease hand weight, if wrist are too different, considering body detection seems more accurate, weight between [0.2, 0.5]
				right_hand_weights *= hand_weight_fnc(dist)

	if body_results.left_hand_landmarks is None:
		left_img = torch.zeros((21,2))
		left_hand_detected = False
		left_hand_weights = torch.zeros(21,1)
	else:
		left_img = torch.tensor([[p.x, p.y] for p in body_results.left_hand_landmarks.landmark])
		left_hand_weights = torch.ones(21,1)
		if THR_WRIST is not None:
			dist = torch.linalg.norm(img_size * left_img[0,:2] - img_size * body_img[WRIST_LEFT_MP,:2])
			if dist > THR_WRIST:
				left_hand_weights *= hand_weight_fnc(dist)

	if body_order is not None:
		body_img = body_img[body_order]
		body_3d = body_3d[body_order]

	pose_2d = (img_size * torch.cat((body_img[...,:2], right_img, left_img)).type(torch.float32))
	return torch.cat((pose_2d, torch.cat((body_img[[...,[2]]], right_hand_weights, left_hand_weights))), -1), body_3d, [True, right_hand_detected, left_hand_detected]
