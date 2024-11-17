from kornia.augmentation import Normalize
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Sequence, Union, Tuple, List

import torch
import torchvision

VGG_WEIGHTS = {
    "vgg11": torchvision.models.VGG11_Weights.IMAGENET1K_V1,
    "vgg16": torchvision.models.VGG16_Weights.IMAGENET1K_V1,
    "vgg19": torchvision.models.VGG19_Weights.IMAGENET1K_V1,
}

VGG_SLICES_PRE_ACTIVATION = {
    "vgg11": [(0, 1), (1, 4), (4, 9), (9, 14), (14, 19)],
    "vgg16": [(0, 3), (3, 8), (8, 15), (15, 22), (22, 29)],
    "vgg19": [(0, 3), (3, 8), (8, 17), (17, 26), (26, 35)],
}

VGG_SLICES_POST_ACTIVATION = {
    "vgg11": [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20)],
    "vgg16": [(0, 4), (4, 9), (9, 16), (16, 23), (23, 30)],
    "vgg19": [(0, 4), (4, 9), (9, 18), (18, 27), (27, 36)],
}

class VGGPerceptualBackbone(nn.Module):
    def __init__(
        self,
        name: str = "vgg16",
        steps: int = 5,
        pre_activation: bool = False,
    ):
        super().__init__()

        name = name.lower()
        assert 1 <= steps <= 5
        assert name in VGG_WEIGHTS.keys()

        model_class = getattr(torchvision.models, name)
        weights = VGG_WEIGHTS[name]
        model = model_class(weights=weights)

        slice_indices = (
            VGG_SLICES_PRE_ACTIVATION
            if pre_activation
            else VGG_SLICES_POST_ACTIVATION
        )
        self.slices = nn.ModuleList(
            [
                model.features[start:end]
                for (start, end) in slice_indices[name][:steps]
            ]
        )
        transforms = weights.transforms()
        self.normalize_mean = transforms.mean
        self.normalize_std = transforms.std

    def forward(self, x):
        out = []
        for slice in self.slices:
            x = slice(x)
            out.append(x)
        return out

    @property
    def normalize_params(self) -> Tuple[List[float]]:
        return self.normalize_mean, self.normalize_std


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        backbone: Union[str, nn.Module],
        weights: Sequence[float] = (4.0 / 32, 8 / 32, 16 / 32, 24 / 32, 1),
        do_normalize: bool = True,
        loss_func: Union[torch.nn.Module, str] = torch.nn.L1Loss(),
        pre_activation: bool = False,
        start_after_iterations = 5000,
        cnt: int = 0
    ):
        super().__init__()

        if isinstance(loss_func, str):
            loss_func = loss_func.lower()
            if loss_func in ("l1", "mae"):
                loss_func = torch.nn.L1Loss(reduction="none")
            elif loss_func in ("l2", "mse"):
                loss_func = torch.nn.MSELoss(reduction="none")
            else:
                raise ValueError(f"Unsupported loss function '{loss_func}'.")

        if isinstance(backbone, str):
            backbone = backbone.lower()
            if backbone.startswith("vgg"):
                self.backbone = VGGPerceptualBackbone(
                    backbone, len(weights), pre_activation
                )
            else:
                assert False, f"Unsupported backbone '{backbone}'."

        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.loss_func = loss_func
        self.weights = torch.tensor(weights)
        self.do_normalize = do_normalize
        self.start_after_iterations = start_after_iterations
        self.cnt = cnt

        normalize_mean, normalize_std = self.backbone.normalize_params
        
        self.normalize = Normalize(
            mean=normalize_mean,
            std=normalize_std,
            keepdim=True,
        )

    def forward(self, input: torch.Tensor, target: torch.FloatTensor) -> torch.FloatTensor:
        
        self.cnt += 1
            
        if input.shape[1] > 3 and len(input.shape) == 4:
            input = input[:, :3, :, :]
        
        # Convert single-channel to grayscale RGB
        if input.size(3) > 3:
            input = input[:, :3, :, :]
                    
        if input.size(1) == 1:
            input = input.repeat((1, 3, 1, 1))
        if target.size(1) == 1:
            target = target.repeat((1, 3, 1, 1))

        # Normalize to backbone distribution
        if self.do_normalize:
            input = self.normalize(input)
            target = self.normalize(target)

        # Extract features
        features_input = self.backbone(input)
        with torch.no_grad():
            features_target = self.backbone(target.detach())

        # Compute loss
        loss = 0.0
        for i, weight in enumerate(self.weights):
            loss += torch.mean(self.loss_func(features_input[i], features_target[i])) * weight

        loss = loss / self.weights.sum()
        
        if self.cnt < self.start_after_iterations:
            loss *= 0
            
        return loss




if __name__ == "__main__":
    loss = PerceptualLoss("resnet50", weights=[1])
    