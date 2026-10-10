"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle

def getDimBox(points, num_dim=3):
	"""
	points: ... N x DIM
	output:
	[[min_d1, max_d1],
	...
	[min_dN, max_dN]]
	"""
	is_torch = torch.is_tensor(points[0])
	num_dim = points[0].shape[-1]
	if isinstance(points, list):
		assert is_torch or isinstance(points[0], np.ndarray)
		if is_torch:
			return np.array([[np.median([pts[...,k].min().detach().cpu() for pts in points]), np.median([pts[...,k].max().detach().cpu() for pts in points])] for k in range(num_dim)])
		else:
			return np.array([[np.median([pts[...,k].min() for pts in points]), np.median([pts[...,k].max() for pts in points])] for k in range(num_dim)])
	else:
		if is_torch: points = points.cpu().detach()
		if isinstance(points, np.ndarray):
			return np.array([[np.median(points[...,k].min(-1)[0]), np.median(points[...,k].max(-1)[0])] for k in range(num_dim)])
		else:
			return np.array([[points[...,k].min(-1)[0].median(), points[...,k].max(-1)[0].median()] for k in range(num_dim)])
	# return np.array([[np.nanmin([np.nanmin(pts[pts[:,k] != np.inf, k]) for pts in points]), np.nanmax([np.nanmax(pts[pts[:,k] != np.inf, k]) for pts in points])] for k in range(num_dim)])

def plotTwoPointsSets(img, pts1, pts2, sz=5, l_sz=1, color1=(0,255,0), color2=(0,0,255), color_edge=(255,0,0)):
	assert pts1.shape == pts2.shape
	assert pts1.shape[1] == 2
	for p1, p2 in zip(pts1, pts2):
		cv2.circle(img, (int(p1[0]), int(p1[1])), sz, color1, -1)
		cv2.circle(img, (int(p2[0]), int(p2[1])), sz, color2, -1)
		cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color_edge, l_sz)


def plotImagesBox(images_list, poses_2d=None, connections=None, box_shape=None):
	"""
	images_list : B x H x W
	poses_2d: NUM_COMPARISON x BATCH x P x 2
	"""
	num_batch = len(images_list)
	sqr = int(np.ceil(np.sqrt(num_batch)))
	if box_shape is not None:
		w, h = box_shape
	else:
		w, h = sqr, sqr

	figs, axs = plt.subplots(h, w, figsize=(12, 12))
	figs.tight_layout(); figs.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

	for idx in range(num_batch):
		ax = axs if h == 1 and w == 1 else axs[idx // h, idx % w]
		if poses_2d is None:
			ax.imshow(images_list[idx])
		else:
			plotPoses2dRGB([pose_2d[idx] for pose_2d in poses_2d], connections, images_list[idx], ax=ax)

def plotPosesBox(poses_list, connections, legend=[], azim = -96, elev = -56, box_shape=None, title=None, title_per_ax=None):
	"""
	poses_list : NUM_COMPARISON x BATCH x P x Dim
	"""
	is_3d = poses_list[0][0].shape[-1] == 3
	poses_comparison = len(poses_list)
	num_batch = len(poses_list[0])
	sqr = int(np.ceil(np.sqrt(num_batch)))
	if title_per_ax is not None: assert len(title_per_ax) == num_batch
	if box_shape is not None:
		w, h = box_shape
	else:
		w, h = sqr, sqr
	if is_3d:
		figs, axs = plt.subplots(h, w, figsize=(12, 12), subplot_kw=dict(projection='3d'))
	else:
		figs, axs = plt.subplots(h, w, figsize=(12, 12))
	figs.tight_layout(); figs.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

	for idx in range(num_batch):
		ax = axs if h == 1 and w == 1 else axs[idx // h, idx % w]
		if is_3d:
			plotPoses([poses_list[k][idx] for k in range(poses_comparison)], connections, azim=azim, elev=elev, legend=legend, ax=ax, title=title_per_ax[idx] if title_per_ax is not None else None)
		else:
			plotPoses2d([poses_list[k][idx] for k in range(poses_comparison)], connections, legend=legend, ax=ax, title=title_per_ax[idx] if title_per_ax is not None else None)
	if title is not None: plt.suptitle(title)

def plotPoses(poses, connections, azim=0, elev=0, save_fname='', legend=[], dim_box=None, title='', ax=None, is_blank=False, s=7, lw=3, colors_l=None, colors_p=None, dpi=300, show_axes=True, legend_fontsize=10):
	if colors_l is None: colors_l = ['red', 'green', 'blue', 'black', 'magenta', 'brown']
	if colors_p is None: colors_p = ['yellow', 'purple', 'navy', 'gray', 'lime', 'skyblue']
	if ax is None:
		fig = plt.figure(figsize=(13.0, 20.0))
		ax = fig.add_subplot(111, projection='3d')

	if dim_box is None: dim_box = getDimBox(poses)
	axes_line = [None for i in range(len(poses))]

	if isinstance(colors_l, np.ndarray) and (colors_l > 1).any():
		colors_l = colors_l.copy() / 255
	if isinstance(colors_p, np.ndarray) and (colors_p > 1).any():
		colors_p = [colors_p.copy() / 255]
	for p, pose_ in enumerate(poses):
		pose = pose_.cpu().detach().numpy() if torch.is_tensor(pose_) else pose_
		ax.scatter(pose[:,0], pose[:,1], pose[:,2], color=colors_p[p], marker='o', s=s)
		# ax.scatter(pose[:,0], pose[:,1], pose[:,2], c=all_colors, marker='.', s=200)
		connections_ = connections[p] if isinstance(connections[0][0], tuple) else connections
		for k, (x,y) in enumerate(connections_):
			color = colors_l[p] if isinstance(colors_l, list) else colors_l[k]
			axes_line[p] = ax.plot([pose[x,0],pose[y,0]], [pose[x,1],pose[y,1]], zs=[pose[x,2],pose[y,2]], linewidth=lw, color=color)[0]
	ax.set_xlim(dim_box[0]); ax.set_ylim(dim_box[1]); ax.set_zlim(dim_box[2])
	ax.set_box_aspect((dim_box[0,1]-dim_box[0,0], dim_box[1,1]-dim_box[1,0], dim_box[2,1]-dim_box[2,0]))
	if title != '':
		if ax is not None: ax.set_title(title, fontsize=30)
		else: plt.title(title, fontsize=30)
	ax.view_init(azim=azim, elev=elev)
	ax.legend(axes_line, legend, prop={'size': legend_fontsize})
	if is_blank:
		ax.xaxis.pane.fill = False
		ax.yaxis.pane.fill = False
		ax.zaxis.pane.fill = False
		ax.xaxis.pane.set_edgecolor('w')
		ax.yaxis.pane.set_edgecolor('w')
		ax.zaxis.pane.set_edgecolor('w')
		if show_axes:
			ax.axes.xaxis.set_ticks([])
			ax.axes.yaxis.set_ticks([])
			ax.axes.zaxis.set_ticks([])
		else:
			ax.axis('off')
		ax.grid(False)
	else:
		ax.set_xlabel('x', fontsize=25); ax.set_ylabel('y', fontsize=25); ax.set_zlabel('z', fontsize=25)
	if save_fname is not None and save_fname != '': plt.savefig(save_fname, bbox_inches='tight', pad_inches=0, dpi=dpi, transparent=True)
	return ax

def plotOverImage(img, poses, connections, mask, colors, colors_pts, colors_edges, circle_sz, line_sz, ax, save_fname):
	axes_line = [None for i in range(len(poses))]
	for p, pose_ in enumerate(poses):
		pose = pose_.detach().cpu() if torch.is_tensor(pose_) else pose_
		if circle_sz > 0:
			for k in range(pose_.shape[0]):
				if colors_pts is not None:
					color = colors_pts[k]
					if isinstance(color, np.ndarray): color = color.tolist()
				else:
					color = colors[p]
				# print((int(pose[k,0]), int(pose[k,1])), img.flags['C_CONTIGUOUS'], img.flags['F_CONTIGUOUS'])
				cv2.circle(img, (int(pose[k,0]), int(pose[k,1])), circle_sz, color, -1) # img must be contiguous!
		if connections:
			connections_ = connections[p] if not isinstance(connections[0][0], int) else connections
		else: connections_ = []
		for k, (x,y) in enumerate(connections_):
			if colors_edges is not None:
				color = colors_edges[k]
				if isinstance(color, np.ndarray): color = color.tolist()
			else:
				color = colors[p]
			if (mask is None) or (mask[p][x] and mask[p][y]):
				if ax is not None:
					axes_line[p] = ax.plot([pose[x,0],pose[y,0]], [pose[x,1],pose[y,1]], linewidth=line_sz, color=np.array(color)/np.linalg.norm(color), zorder=1)[0]
				if save_fname:
					cv2.line(img, (int(pose[x,0]), int(pose[x,1])), (int(pose[y,0]), int(pose[y,1])), color, line_sz)
	return axes_line

def plotPoses2dRGB(poses, connections, img_, save_fname='', circle_sz=4, line_sz=1, show=False, legend=None, copy=True, ax=None, colors=None, colors_pts=None, colors_edges=None, mask=None, title=None):
	if torch.is_tensor(img_): img_ = img_.numpy()
	img = img_.copy() if copy else img_
	if colors is None: colors = [[0,0,255], [0,255,0], [255,0,0], [255, 255, 0], [100, 100, 100]] # blue, green, red (keep attention on order of channels)
	if ax is None:
		fig = plt.figure(figsize=(13.0, 20.0))
		ax = fig.add_subplot(111)
	axes_line = plotOverImage(img, poses, connections, mask, colors, colors_pts, colors_edges, circle_sz, line_sz, ax, save_fname)
	if save_fname:
		# cv2.imwrite(save_fname, img)
		cv2.imwrite(save_fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
		# plt.savefig()
	if legend is not None: ax.legend(axes_line, legend)
	if title is not None: ax.set_title(title)
	ax.axis("off")
	ax.grid(False)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.imshow(img)
	if show: plt.show()
	return img

def processPoints(poses):
	"""
	B x N x D
	"""
	if isinstance(poses, list):
		poses = torch.stack(poses) if torch.is_tensor(poses[0]) else torch.FloatTensor(np.stack(poses))
	elif not torch.is_tensor(poses):
		poses = torch.FloatTensor(poses)
	return poses

def dimBox(points):
	"""
	points ... x N x D
	return ... x 2 x D, at 0 is min, at 1 is max
	"""
	return torch.stack([torch.min(points, dim=-2).values, torch.max(points, dim=-2).values], dim=-2)

def plotMesh(mesh, faces, azim=0, elev=0, ax=None, dim_box=None, show=False, title=None, colors=None, is_blank=False, save_fname=None, dpi=250, median_dim_box=False, antialiased=False, shade=True, alpha=0.5):
	"""
	mesh  : N x 3 or B x N x 3
	faces : E x 3 or B x E x 3
	"""
	batch_meshes = isinstance(mesh, list) or mesh.ndim == 3
	batch_faces = isinstance(faces, list) or faces.ndim == 3
	if not batch_meshes: mesh = mesh[None]
	if colors is None:
		colors = ['red', 'green', 'blue', 'black', 'magenta', 'brown']
	# else: assert len(colors) == len(mesh)
	if batch_faces: assert len(faces) == len(mesh)
	if ax is None:
		fig = plt.figure(figsize=(13.0, 20.0))
		ax = fig.add_subplot(projection='3d')
	for i, (m, clr) in enumerate(zip(mesh, colors)):
		faces_ = faces[i] if batch_faces else faces
		if faces_ is None: continue
		if torch.is_tensor(m): m = m.detach().cpu().numpy()
		if torch.is_tensor(faces_): faces_ = faces_.detach().cpu().numpy()
		ax.plot_trisurf(m[:,0], m[:,1], m[:,2], triangles=faces_, color=clr, antialiased=antialiased, shade=shade, alpha=alpha, edgecolor='none', linewidth=0)

	ax.view_init(azim=azim, elev=elev)
	if dim_box is None:
		if median_dim_box:
			dim_box = getDimBox(mesh)
		else:
			pts = processPoints(mesh)
			dim_box = dimBox(pts.reshape(1, -1, pts.shape[-1]))[0].detach().cpu().numpy().T

	ax.set_xlim(dim_box[0]); ax.set_ylim(dim_box[1]); ax.set_zlim(dim_box[2])
	ax.set_box_aspect((dim_box[0,1]-dim_box[0,0], dim_box[1,1]-dim_box[1,0], dim_box[2,1]-dim_box[2,0]))
	if title is not None: ax.set_title(title)
	if is_blank:
		ax.axis("off")
		ax.grid(False)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_zticks([])
	if save_fname is not None:
		plt.savefig(save_fname, bbox_inches='tight', pad_inches=0, dpi=dpi, transparent=True)
	if show: plt.show()
	return ax

def plotPoses2d(poses, connections, save_fname='', legend=[], dim_box=None, title='', ax=None, colors_l=None, colors_p=None, lw=3, s=20, is_blank=False):
	if colors_l is None: colors_l = ['red', 'green', 'blue', 'black', 'magenta', 'brown']
	if colors_p is None: colors_p = ['yellow', 'purple', 'navy', 'gray', 'lime', 'skyblue']
	if len(colors_l) < len(poses):
		colors_l = np.random.RandomState(0).rand(len(poses),3)
		colors_p = np.random.RandomState(0).rand(len(poses),3)
	if ax is None:
		fig = plt.figure(figsize=(13.0, 20.0))
		ax = fig.add_subplot(111)
	if dim_box is None: dim_box = getDimBox(poses,num_dim=2)
	axes_line = [None for i in range(len(poses))]
	for p, pose_ in enumerate(poses):
		pose = pose_.cpu().detach().numpy() if torch.is_tensor(pose_) else pose_
		connections_ = connections[p] if isinstance(connections[0][0], tuple) else connections
		for k, (x,y) in enumerate(connections_):
			axes_line[p] = ax.plot([pose[x,0],pose[y,0]], [pose[x,1],pose[y,1]], linewidth=lw, color=colors_l[p], zorder=1)[0]
		ax.scatter(pose[:,0], pose[:,1], color=colors_p[p], marker='o', s=s, zorder=2)
	if is_blank:
		ax.axis('off')
		ax.grid(False)
	else:
		ax.set_xlabel('x', fontsize=35); ax.set_ylabel('y', fontsize=35);
		ax.tick_params(axis='x', labelsize=18)
		ax.tick_params(axis='y', labelsize=18)
		ax.legend(axes_line, legend)
	ax.set_xlim(dim_box[0]); ax.set_ylim(dim_box[1])
	ax.set_aspect('equal', 'box')
	ax.invert_yaxis()
	if title != '': plt.title(title, fontsize=30)
	if save_fname != '' and save_fname[-6:] == 'pickle': pickle.dump(fig, open(save_fname, 'wb'))
	elif save_fname != '': plt.savefig(save_fname, bbox_inches='tight',pad_inches=0, dpi=300)
	return ax

import matplotlib.gridspec as gridspec
def plotImagePointsSideBySide(images, poses_2d, poses_3d, connections_img, connections_3d, azim=-93, elev=-50, circle_size=3, legend_3d=[], save_fname=''):
	"""
	images : N x height x width
	poses_2d / poses_3d : N x POSES x pts x dim
	"""
	num_rows = len(images)
	fig = plt.figure(figsize=(12, 6*num_rows)) # width, height

	fig.tight_layout(); fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
	if connections_img is None: connections_img = connections_3d
	if connections_3d is None: connections_3d = connections_img
	for i in range(num_rows):
		# ax_1 = plt.subplot(gs1[i*2])
		ax_1 = fig.add_subplot(num_rows, 2, 1+i*2)
		if poses_2d is None:
			plt.imshow(images[i])
		else:
			plt.imshow(plotPoses2dRGB(list(poses_2d[i]), connections_img, images[i], circle_sz=circle_size, show=False, ax=ax_1))
		# ax_2 = plt.subplot(gs1[i*2+1], projection='3d')
		ax_2 = fig.add_subplot(num_rows, 2, 1+i*2+1, projection='3d')
		plotPoses(list(poses_3d[i]), connections_3d, azim=azim, elev=elev, ax=ax_2, legend=legend_3d)
	plt.subplots_adjust(wspace=0, hspace=0)
	if save_fname != '': plt.savefig(save_fname, bbox_inches='tight',pad_inches=0, dpi=300)

def plotPose(pts, connections):
	return plotPoses([pts], connections)

def plotTwoPoses(pts1, pts2, connections, i, loss, total_time, azim, elev, save_fname):
	return plotPoses([pts1, pts2], connections, i, loss, total_time, azim, elev, save_fname)

def plotLosses(losses_list, legend, ylabel, colors=None, title='Losses', yscale='log', lw=1, save_fname='', iters=None, xlabel=None, log_base=None, fontsize=20, yticks=None, ylim=None):
	fig = plt.figure(); ax = fig.add_subplot(111); ax.grid('both')
	if colors is None: colors = ['red', 'green', 'blue', 'black', 'magenta', 'brown']
	for k, losses in enumerate(losses_list):
		if iters is not None: ax.plot(iters, losses, '.-', linewidth=lw, color=colors[k], markersize=9, markerfacecolor=colors[(k+1)%len(colors)])
		else: ax.plot(losses, '.-', linewidth=lw, color=colors[k], markersize=9, markerfacecolor=colors[(k+1)%len(colors)])
	ax.set_xlabel('iterations' if xlabel is None else xlabel, fontsize=fontsize); ax.set_ylabel(ylabel, fontsize=fontsize);
	# ax.set_yscale(yscale)
	if log_base is None: ax.set_yscale(yscale)
	else: ax.set_yscale(yscale, basey=log_base)
	if legend is not None: ax.legend(legend, fontsize=fontsize);
	ax.set_title(title, fontsize=int(fontsize*1.2))
	if yticks is not None:
		ax.set_yticks(yticks)
	if ylim is not None: ax.set_ylim(ylim)
	if save_fname != '': plt.savefig(save_fname, bbox_inches='tight',pad_inches=0, dpi=300)
	return ax

def showProgress(losses_list, times_list, legends):
	plt.figure()
	for i in range(len(losses_list)):  plt.plot(losses_list[i], '--.', linewidth=1)
	plt.xlabel('iterations')
	plt.ylabel('loss')
	plt.title('Loss per iteration')
	plt.legend(legends)

	plt.figure()
	for i in range(len(losses_list)):  plt.plot(times_list[i], '--.', linewidth=1)
	plt.xlabel('iterations')
	plt.ylabel('time, sec')
	plt.title('Cumulative time')
	plt.legend(legends)

def concatenateImagesResize(i1, i2, hor=True):
	if i1.shape[0]*i1.shape[1] > i2.shape[0]*i2.shape[1]:
		if hor:
			i1 = cv2.resize(i1, (int(float(i1.shape[1])*i2.shape[0]/i1.shape[0]), i2.shape[0]), interpolation=cv2.INTER_AREA)
		else:
			raise NotImplementedError

	elif i2.shape[0]*i2.shape[1] > i1.shape[0]*i1.shape[1]:
		if hor:
			i2 = cv2.resize(i2, (int(float(i2.shape[1])*i1.shape[0]/i2.shape[0]), i1.shape[0]), interpolation=cv2.INTER_AREA)
		else:
			i2 = cv2.resize(i2, (i1.shape[1], int(float(i2.shape[0])*i1.shape[1]/i2.shape[1])), interpolation=cv2.INTER_AREA)

	if hor: return cv2.hconcat((i1,i2))
	else: return cv2.vconcat((i1,i2))

def concatenateImages(img1, img2, offset=0):
	def concatenateRow (i1, i2):
		if offset == 0:
			return cv2.hconcat((i1, i2));
		else:
			offset_img = np.zeros((i1.shape[0], offset, 3), dtype=img1.dtype)
			concatenated_img = cv2.hconcat((i1, offset_img))
			return cv2.hconcat((concatenated_img, i2))

	if img1.shape[0] == img2.shape[0]:
		return concatenateRow(img1, img2)
	else:
		if (img1.shape[0] < img2.shape[0]):
			img1_ = img1.copy();
			# add zeros matrix which missing to 1 image
			back = np.zeros((img2.shape[0] - img1.shape[0], img1.shape[1],3), dtype=img1.dtype);
			img1_ = cv2.vconcat((img1_, back))
			return concatenateRow(img1_, img2);
		else:
			img2_ = img2.copy();
			back = np.zeros((img1.shape[0] - img2.shape[0], img2.shape[1],3), dtype=img2.dtype);
			img2_ = cv2.vconcat((img2_, back))
			return concatenateRow(img1, img2_)


def plotImgs(axs, imgs_plots):
	cnt = 0
	for i in range(axs.shape[0]):
		for j in range(axs.shape[1]):
			axs[i, j].imshow(imgs_plots[cnt])
			cnt += 1
			if cnt >= len(imgs_plots): return

def tensorImg2Np(img):
	if img.ndim == 3:
		shape = (1,2,0)
	elif img.ndim == 4:
		shape = (0,2,3,1)
	else: raise NotImplementedError
	return np.array(img.cpu().numpy().transpose(shape)*255, dtype=np.uint8)
