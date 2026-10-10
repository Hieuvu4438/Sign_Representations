"""
Maksym Ivashechkin
University of Surrey
19 February 2024
CC-BY-NC-SA-4.0
"""

from enum import Enum

class LOSS_TYPE(Enum):
	threeD = 1; reproj = 2; angular = 3; offsets = 4; reproj_multi = 5; contrastive = 6; kl_div = 7; twoD_monocular = 9; angular_limits = 10; temporal_laplace = 11; noise = 12; denoise = 13; intersections = 14

class LOSS_FNC(Enum):
	huber = 1; l1 = 2;
