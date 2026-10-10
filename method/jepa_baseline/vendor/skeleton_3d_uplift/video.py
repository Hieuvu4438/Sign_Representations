"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import os
import cv2


def getImageWidthVideo(video_name):
	video = cv2.VideoCapture(video_name)
	assert video.isOpened()
	success, frame = video.read()
	assert success
	return frame.shape[:2][::-1]

def getNumberOfFramesInVideo(video_name):
	video = cv2.VideoCapture(video_name)
	assert video.isOpened()
	return int(video.get(cv2.CAP_PROP_FRAME_COUNT))

class VideoReader:
	def __init__(self, videoname, open_runtime=False, convert_rgb=False):
		self.open_runtime = open_runtime
		self.concert_rgb = convert_rgb
		self.videoname = videoname
		assert os.path.exists(videoname)
		if not open_runtime:
			self.video = cv2.VideoCapture(videoname)
			assert self.video.isOpened(), videoname
	def __len__(self):
		return getNumberOfFramesInVideo(self.videoname)
	def getImageWH(self):
		return self.getFrame(0).shape[:2][::-1]

	def getFPS(self):
		return self.video.get(cv2.CAP_PROP_FPS)

	def close(self):
		if not self.open_runtime:
			self.video.release()

	def read(self):
		res, frame = self.video.read()
		if not res: return None
		if self.concert_rgb:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		return frame

	def getFrame(self, frame_idx):
		if self.open_runtime:
			video = cv2.VideoCapture(self.videoname)
			video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
			res, frame = video.read()
			video.release()
		else:
			self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
			res, frame = self.video.read()
		if self.concert_rgb:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		assert res
		return frame
