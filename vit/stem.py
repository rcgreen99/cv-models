import torch
from torch import nn


class Stem(nn.Module):
    """Stem (aka Patchify in this case) used in the Vision Transformer model.
    Performs patching and embedding of the input image"""

    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, input_tensor: torch.Tensor, use_conv=False) -> torch.Tensor:
        if use_conv:
            return self._conv(input_tensor)
        return self._patchify(input_tensor)

    def _patchify(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Patchify the input image
        batch_size, num_channels, height, width = input_tensor.shape
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), "Image dimensions must be divisible by the patch size"

        # Calculate the number of patches
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        patch_dim = num_channels * self.patch_size**2

        # Reshape the input tensor to patchify it
        patches = input_tensor.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            batch_size, num_patches, patch_dim
        )

        return patches

    def _conv(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Use a convolutional layer to patchify the input image
        return nn.Conv2d(
            in_channels=input_tensor.shape[1],
            out_channels=self.patch_size**2,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )(input_tensor)
