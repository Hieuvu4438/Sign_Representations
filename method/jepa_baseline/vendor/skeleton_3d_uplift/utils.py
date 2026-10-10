"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

import torch

class PrintColors:
	# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

def getMinMax(points):
	"""
	points: [B, N, dim]
	output: [B, 2, dim]
	"""
	min_over_dim = points.min(-2)[0] # B x dim
	max_over_dim = points.max(-2)[0] # B x dim
	return torch.stack((min_over_dim, max_over_dim)).transpose(0,1)

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
	# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#euler_angles_to_matrix
	"""
	Return the rotation matrices for one of the rotations about an axis
	of which Euler angles describe, for each value of the angle given.

	Args:
		axis: Axis label "X" or "Y or "Z".
		angle: any shape tensor of Euler angles in radians

	Returns:
		Rotation matrices as tensor of shape (..., 3, 3).
	"""
	cos = torch.cos(angle)
	sin = torch.sin(angle)
	one = torch.ones_like(angle)
	zero = torch.zeros_like(angle)

	if   axis == "X": R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
	elif axis == "Y": R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
	elif axis == "Z": R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
	else: raise ValueError("letter must be either X, Y or Z.")
	return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
	# assert len(convention) == euler_angles.numel()
	# euler_angles must be tensor of size: [..., NUM_ANGLES] or scalar
	if euler_angles.numel() == 1 or (len(euler_angles.shape) == 1 and len(euler_angles) > 3):
		NUM_ANGLES = 1
	else: NUM_ANGLES = euler_angles.shape[-1]
	assert NUM_ANGLES >= 1 and NUM_ANGLES <= 3 and len(convention) == NUM_ANGLES
	if NUM_ANGLES == 1: 
		if euler_angles.ndim > 1: euler_angles = euler_angles[...,0] # flatten last dimension
		return _axis_angle_rotation(convention[0], euler_angles)
	matrices = [
		_axis_angle_rotation(c, e)
		for c, e in zip(convention, torch.unbind(euler_angles, -1))
	]
	if len(matrices) == 2: return torch.matmul(matrices[0], matrices[1])
	else: return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _angle_from_tan(axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool) -> torch.Tensor:
	"""
	Extract the first or third Euler angle from the two members of
	the matrix which are positive constant times its sine and cosine.

	Args:
		axis: Axis label "X" or "Y or "Z" for the angle we are finding.
		other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
			convention.
		data: Rotation matrices as tensor of shape (..., 3, 3).
		horizontal: Whether we are looking for the angle for the third axis,
			which means the relevant entries are in the same row of the
			rotation matrix. If not, they are in the same column.
		tait_bryan: Whether the first and third axes in the convention differ.

	Returns:
		Euler Angles in radians for each matrix in data as a tensor
		of shape (...).
	"""

	i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
	if horizontal: i2, i1 = i1, i2
	even = (axis + other_axis) in ["XY", "YZ", "ZX"]
	if horizontal == even: return torch.atan2(data[..., i1], data[..., i2])
	if tait_bryan: return torch.atan2(-data[..., i2], data[..., i1])
	return torch.atan2(data[..., i2], -data[..., i1])

def _index_from_letter(letter: str) -> int:
	if letter == "X": return 0
	if letter == "Y": return 1
	if letter == "Z": return 2
	raise ValueError("letter must be either X, Y or Z.")

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:

	"""
	Convert rotations given as rotation matrices to Euler angles in radians.

	Args:
		matrix: Rotation matrices as tensor of shape (..., 3, 3).
		convention: Convention string of three uppercase letters.

	Returns:
		Euler angles in radians as tensor of shape (..., 3).
	"""
	if len(convention) != 3:
		raise ValueError("Convention must have 3 letters.")
	if convention[1] in (convention[0], convention[2]):
		# raise ValueError(f"Invalid convention {convention}.")
		raise ValueError("Invalid convention "+convention)
	for letter in convention:
		if letter not in ("X", "Y", "Z"):
			raise ValueError("Invalid letter "+letter+" in convention string.")
	if matrix.size(-1) != 3 or matrix.size(-2) != 3:
		raise ValueError("Invalid rotation matrix shape "+str(matrix.shape))
	i0 = _index_from_letter(convention[0])
	i2 = _index_from_letter(convention[2])
	tait_bryan = i0 != i2
	if tait_bryan:
		# central_angle = torch.asin(matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0))
		central_angle = torch.asin((matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)).clamp(-1.0, 1.0))
	else:
		# central_angle = torch.acos(matrix[..., i0, i0])
		central_angle = torch.acos(matrix[..., i0, i0].clamp(-1.0, 1.0))
	o = (
		_angle_from_tan(convention[0], convention[1], matrix[..., i2], False, tait_bryan),
		central_angle,
		_angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True, tait_bryan),
	)
	return torch.stack(o, -1)
