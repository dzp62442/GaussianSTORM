import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from .lpips import LPIPS


class RGBLpipsLoss(nn.Module):
    """
    Loss module that combines RGB reconstruction loss (MSE) and optional perceptual loss (LPIPS).

    Args:
        perceptual_weight (float): Weight for the perceptual loss.
        use_perceptual_loss (bool): Flag to determine whether perceptual loss is used.
        enable_perceptual_loss (bool): Initial state of perceptual loss usage.
    """

    def __init__(
        self,
        perceptual_weight=0.5,
        use_perceptual_loss=True,
        enable_perceptual_loss=True,
        weights_path=None,
        offline=False,
        perceptual_chunk_size=None,
        checkpoint_perceptual=False,
    ):
        super().__init__()

        # Initialize the perceptual loss (LPIPS) if enabled
        if enable_perceptual_loss:
            self.perceptual_loss = LPIPS(weights_path=weights_path, offline=offline).eval()
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False

        self.perceptual_weight = perceptual_weight
        self.use_perceptual_loss = use_perceptual_loss
        self.enable_perceptual_loss = enable_perceptual_loss
        self.perceptual_chunk_size = perceptual_chunk_size
        self.checkpoint_perceptual = checkpoint_perceptual

    def set_perceptual_loss(self, enable=True):
        """
        Enable or disable the perceptual loss.

        Args:
            enable (bool): Whether to enable perceptual loss.
        """
        self.enable_perceptual_loss = enable and self.use_perceptual_loss

    def forward(self, rgb, targets, valid_mask=None):
        """
        Compute the RGB reconstruction loss and (optionally) perceptual loss.

        Args:
            rgb (Tensor): Predicted RGB values with shape (..., H, W, C).
            targets (Tensor): Ground truth RGB values with shape (..., H, W, C).

        Returns:
            dict: Dictionary containing 'rgb_loss' and optionally 'perceptual_loss'.
        """
        # Rearrange input tensors to the format (batch, channels, height, width)
        rgb = rearrange(rgb, "... h w c -> (...) c h w")
        targets = rearrange(targets, "... h w c -> (...) c h w")
        if valid_mask is None:
            rgb_loss = F.mse_loss(rgb, targets)
        else:
            valid_mask = rearrange(valid_mask, "... h w -> (...) 1 h w").bool()
            selected = (rgb - targets).masked_select(valid_mask.expand_as(rgb)).square()
            rgb_loss = selected.mean() if selected.numel() else selected.sum()
        loss_dict = {"rgb_loss": rgb_loss}
        if self.enable_perceptual_loss:
            if valid_mask is not None:
                rgb = rgb.masked_fill(~valid_mask, 0)
                targets = targets.masked_fill(~valid_mask, 0)
            chunk = self.perceptual_chunk_size or len(rgb)
            values = []
            for start in range(0, len(rgb), chunk):
                pred, gt = rgb[start:start + chunk], targets[start:start + chunk]
                if self.checkpoint_perceptual and torch.is_grad_enabled() and pred.requires_grad:
                    value = checkpoint(self.perceptual_loss, pred, gt, use_reentrant=False)
                else:
                    value = self.perceptual_loss(pred, gt)
                values.append(value.reshape(-1))
            loss_dict["perceptual_loss"] = self.perceptual_weight * torch.cat(values).mean()

        return loss_dict
