"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import torch

def sineNorm(x):
	# converts real range to [0, 1]
	return (torch.sin(x) + 1.0) * .5

class NormFnc:
	def __init__(self, limits_min, limits_max, norm_fnc):
		super(NormFnc).__init__()
		self.norm_fnc = norm_fnc
		self.limits_dif = limits_max - limits_min
		self.limits_min = limits_min
		self.limits_max = limits_max
		self.once = True

	def getConstraints(self):
		return torch.stack((self.limits_min, self.limits_dif+self.limits_min)).T

	def __call__(self, x, *args, **kwargs):
		if self.once and self.limits_min.device != x.device:
			self.limits_min = self.limits_min.to(x.device)
			self.limits_dif = self.limits_dif.to(x.device)
			self.once = False
		return self.norm_fnc(x) * self.limits_dif + self.limits_min

class SineNormFnc(NormFnc):
	def __init__(self, min_values=None, max_values=None, min_max_values=None):
		if min_values is not None and max_values is not None:
			super(SineNormFnc, self).__init__(min_values, max_values, sineNorm)
		else:
			assert min_max_values is not None
			super(SineNormFnc, self).__init__(min_max_values[:,0], min_max_values[:,1], sineNorm)

class AngularSineNormFnc(SineNormFnc):
	def __init__(self, angles_min=None, angles_max=None, angles_min_max=None):
		super(AngularSineNormFnc, self).__init__(angles_min, angles_max, angles_min_max)

class SigmoidNormFnc(NormFnc):
	def __init__(self, limit_min, limit_max):
		super(SigmoidNormFnc, self).__init__(limit_min, limit_max, torch.nn.Sigmoid)

class Normalization2D:
	def __init__(self, skeleton):
		self.skeleton = skeleton

class Normalization2DL2(Normalization2D):
	def __init__(self, skeleton):
		super(Normalization2DL2, self).__init__(skeleton)

	def __call__(self, x):
		x = self.skeleton.centralizePose(x) if self.skeleton is not None else x
		return x / (1e-10 + torch.linalg.norm(x.flatten(start_dim=-2, end_dim=-1), axis=-1)[...,None,None])

class Normalization2DL2Med(Normalization2D):
	def __init__(self, skeleton):
		super(Normalization2DL2Med, self).__init__(skeleton)

	def __call__(self, x):
		x = self.skeleton.centralizePose(x) if self.skeleton is not None else x
		return x / (1e-10 + torch.linalg.norm(x, axis=-1).median(-1).values[...,None,None])

class Normalization2DStd(Normalization2D):
	def __init__(self, skeleton):
		""" it is the same as Normalization2DL2 up to scale"""
		super(Normalization2DStd, self).__init__(skeleton)

	def __call__(self, x):
		x = self.skeleton.centralizePose(x) if self.skeleton is not None else x
		return x / (1e-10 + x.std(-2).mean(-1)[...,None,None])

class Normalization2DCentroid(Normalization2D):
	def __init__(self, skeleton):
		""" it is the same as Normalization2DL2 up to scale"""
		super(Normalization2DCentroid, self).__init__(skeleton)

	def __call__(self, x):
		x = self.skeleton.centralizePose(x) if self.skeleton is not None else x
		return x / (1e-10 + x.abs().mean(-2).mean(-1)[...,None,None])

class Normalization2DMiddle(Normalization2D):
	def __init__(self, skeleton):
		""" it is the same as Normalization2DL2 up to scale"""
		super(Normalization2DMiddle, self).__init__(skeleton)

	def __call__(self, x):
		x = self.skeleton.centralizePose(x) if self.skeleton is not None else x
		return x / (1e-10 + x.abs().median(-2).values.mean(-1)[...,None,None])

class Normalization2DL1Med(Normalization2D):
	def __init__(self, skeleton):
		super(Normalization2DL1Med, self).__init__(skeleton)

	def __call__(self, x):
		x = self.skeleton.centralizePose(x) if self.skeleton is not None else x
		return x / (1e-10 + torch.abs(x).sum(-1).median(-1).values[...,None,None])


class Normalization2DL2MedHands2(Normalization2D):
	def __init__(self, skeleton):
		super(Normalization2DL2MedHands2, self).__init__(skeleton)

	def __call__(self, x):
		"""
		:param x: (... 2, N, 2)
		"""
		x_c = self.skeleton.centralizePose(x)
		scale = .5 * (torch.linalg.norm(x_c[...,[0],:,:].flatten(start_dim=-2, end_dim=-1), axis=-1) + torch.linalg.norm(x_c[...,[1],:,:].flatten(start_dim=-2, end_dim=-1), axis=-1))[...,None,None]
		o1 = x / (1e-10 + scale)
		o1 -= self.skeleton.getOrigin(o1[...,[0],:,:])
		return o1


class SineZeroOneNormalization(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return sineNorm(x)