"""Video-coherent transforms; unspecified paper parameters must be supplied explicitly."""

from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import RandomResizedCrop, InterpolationMode
from torchvision.transforms import functional as F
from .upstream_planckian.planckianTransforms import PlanckianJitter
from .upstream_signrep.augmentation.video.base_video_aug import Transformation


@dataclass(frozen=True)
class AugmentationConfig:
    # Engineering preset, NOT claimed to be the authors' unpublished probabilities.
    crop_scale: tuple = (.8, 1.)
    crop_ratio: tuple = (.75, 4 / 3)
    planckian_probability: float = .8
    planckian_mode: str = "blackbody"
    blur_probability: float = .5
    blur_sigma: tuple = (.1, 2.)
    grayscale_probability: float = .2

    def __post_init__(self):
        for probability in [self.planckian_probability, self.blur_probability, self.grayscale_probability]:
            if not 0 <= probability <= 1:
                raise ValueError("Invalid augmentation probability")
        if not 0 < self.crop_scale[0] <= self.crop_scale[1] <= 1 or not 0 < self.crop_ratio[0] <= self.crop_ratio[1]:
            raise ValueError("Invalid crop range")
        if not 0 < self.blur_sigma[0] <= self.blur_sigma[1]:
            raise ValueError("Invalid blur sigma")


class VideoTransform:
    def __init__(self, training, config=None):
        self.training = training
        self.config = config or AugmentationConfig()
        self.author_inference = Transformation() if not training else None
        self.mean = torch.tensor([.485, .456, .406]).view(3, 1, 1, 1)
        self.std = torch.tensor([.229, .224, .225]).view(3, 1, 1, 1)

    def __call__(self, frames):
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError("Expected RGB uint8 T×H×W×3")
        if not self.training:
            return self.author_inference.aug_video(frames).permute(1, 0, 2, 3)
        tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
        if self.training:
            tensor = F.resize(tensor, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
            if torch.rand(()) < self.config.planckian_probability:
                jitter = PlanckianJitter(self.config.planckian_mode)
                jitter.idx = torch.randint(len(jitter.pl), (1,)).item()
                tensor = torch.stack([F.pil_to_tensor(jitter(F.to_pil_image(frame))) for frame in tensor])
            top, left, height, width = RandomResizedCrop.get_params(tensor[0], self.config.crop_scale, self.config.crop_ratio)
            tensor = F.resized_crop(tensor, top, left, height, width, [224, 224], antialias=True)
            if torch.rand(()) < self.config.blur_probability:
                sigma = torch.empty(()).uniform_(*self.config.blur_sigma).item()
                tensor = F.gaussian_blur(tensor, [23, 23], [sigma, sigma])
            if torch.rand(()) < self.config.grayscale_probability:
                tensor = F.rgb_to_grayscale(tensor, num_output_channels=3)
        tensor = tensor.float().div(255).permute(1, 0, 2, 3)
        return (tensor - self.mean) / self.std
