# -*- coding: utf-8 -*-
# pylint: disable=E1101,W0221,R0901,R0902,R0913,R0914,E1136,C0301,C0411,C0412
"""
Loss Function
=============
*Created on 25 by bari_is*
*Copyright (C) 2025*
*For COPYING and LICENSE details, please refer to the LICENSE file*

A PyTorch module for calculating the contrastive loss used in the SatCLIP model.
This loss function computes the average of two cross-entropy losses: one for
image-to-coordinate logits and another for coordinate-to-image logits. It also
supports caching of ground-truth labels and distributed training.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class SatCLIPLoss(nn.Module):
    """
    A PyTorch module for calculating the contrastive loss used in the SatCLIP model.
    This loss function computes the average of two cross-entropy losses: one for
    image-to-coordinate logits and another for coordinate-to-image logits. It also
    supports caching of ground-truth labels and distributed training.

    Parameters
    ----------
    local_loss : bool, optional
        If True, adjusts the ground-truth labels for local loss computation in
        distributed training. Default is False.
    cache_labels : bool, optional
        If True, caches the ground-truth labels to avoid recomputation. Default is False.
    rank : int, optional
        The rank of the current process in distributed training. Default is 0.
    world_size : int, optional
        The total number of processes in distributed training. Default is 1.

    Attributes
    ----------
    local_loss : bool
        Indicates whether local loss computation is enabled.
    cache_labels : bool
        Indicates whether ground-truth labels are cached.
    rank : int
        The rank of the current process in distributed training.
    world_size : int
        The total number of processes in distributed training.
    prev_num_logits : int
        Stores the number of logits from the previous forward pass for caching purposes.
    labels : dict
        A dictionary that caches ground-truth labels for different devices.

    Methods
    -------
    get_ground_truth(device, num_logits)
        Computes or retrieves cached ground-truth labels for the given device and
        number of logits.
    forward(logits_per_image, logits_per_coord, output_dict=False)
        Computes the contrastive loss for the given logits.

    Notes
    -----
    - The loss is averaged over the two cross-entropy losses.
    - For distributed training, the ground-truth labels are adjusted based on the
      rank of the process.
    """

    def __init__(
        self,
        local_loss=False,
        cache_labels=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        """
        Generate ground-truth labels for a given device and number of logits.

        Parameters
        ----------
        device : torch.device
            The device on which the labels tensor will be created.
        num_logits : int
            The number of logits for which ground-truth labels are generated.

        Returns
        -------
        torch.Tensor
            A tensor containing the ground-truth labels.

        Notes
        -----
        - If caching is enabled (`self.cache_labels`), the labels are cached for reuse.
        - If the number of logits (`num_logits`) changes or the device is different,
          the cached labels are recalculated.
        - In a distributed setting (`self.world_size > 1`) with local loss enabled
          (`self.local_loss`), the labels are adjusted based on the rank of the process.
        """

        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(self, logits_per_image, logits_per_coord, output_dict=False):
        """
        Computes the forward pass for the loss function.

        Parameters
        ----------
        logits_per_image : torch.Tensor
            The predicted logits for the image inputs. Shape: (batch_size, num_classes).
        logits_per_coord : torch.Tensor
            The predicted logits for the coordinate inputs. Shape: (batch_size, num_classes).
        output_dict : bool, optional
            If True, returns the loss as a dictionary with the key "contrastive_loss".
            If False, returns the loss as a scalar. Default is False.

        Returns
        -------
        dict or torch.Tensor
            If `output_dict` is True, returns a dictionary with the key "contrastive_loss"
            containing the computed loss. Otherwise, returns the computed loss as a scalar.

        Notes
        -----
        The loss is computed as the average of the cross-entropy losses for the image logits
        and the coordinate logits. Ground truth labels are generated internally based on
        the batch size.
        """

        device = logits_per_image.device

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_coord, labels)) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
